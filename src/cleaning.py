"""
Cleaning + entity-resolution helpers for the advertiser-classification project.

Steps:
  1. load_and_harmonize()  - stack 6 SSP exports into one schema (dates, currency, labels)
  2. clean_name()          - rule-based normalisation of messy advertiser strings
  3. cluster_names()       - fuzzy-merge remaining variants (typos, spacing) into canonical advertisers
"""
import re
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process

# --------------------------------------------------------------------------- #
# 1. Harmonise the 6 SSP exports
# --------------------------------------------------------------------------- #
GBP_TO_USD = 1.27  # fixed-rate assumption; in production use the daily FX rate

SSP_SCHEMAS = {
    "ssp_a": dict(date="date", advertiser="advertiser", revenue="revenue_usd", device="device",
                  position="position", geo="country", date_fmt="%Y-%m-%d", fx=1.0),
    "ssp_b": dict(date="Date", advertiser="Advertiser Name", revenue="Revenue (GBP)",
                  device="Device Type", position="Ad Slot", geo="Geo",
                  date_fmt="%d/%m/%Y", fx=GBP_TO_USD),
    "ssp_c": dict(date="day", advertiser="buyer", revenue="net_revenue", device="platform",
                  position="placement", geo="market", date_fmt="%Y%m%d", fx=1.0),
    "ssp_d": dict(date="report_date", advertiser="advertiser_name", revenue="revenue",
                  device="device_category", position="ad_position", geo="geo_country",
                  date_fmt="%m-%d-%Y", fx=1.0),
    "ssp_e": dict(date="Day", advertiser="Buyer / Advertiser", revenue="Earnings (USD)",
                  device="Device", position="Placement", geo="Region",
                  date_fmt="%d %b %Y", fx=1.0),
    "ssp_f": dict(date="dt", advertiser="adv", revenue="rev", device="dev", position="pos",
                  geo="geo", date_fmt="%Y-%m-%d", fx=1.0),
}

# One lower-cased lookup covers every SSP's labels (values do not collide across SSPs)
DEVICE_MAP = {"desktop": "desktop", "desk": "desktop", "mobile": "mobile", "smartphone": "mobile",
              "mob": "mobile", "tablet": "tablet", "tab": "tablet"}
POSITION_MAP = {"above the fold": "atf", "top": "atf", "atf": "atf",
                "in-article": "in_content", "middle": "in_content", "inart": "in_content",
                "below the fold": "btf", "bottom": "btf", "btf": "btf",
                "sidebar": "sidebar", "right_rail": "sidebar", "side": "sidebar",
                "sticky footer": "sticky", "anchor": "sticky", "sticky": "sticky"}
GEO_MAP = {"us": "US", "united states": "US", "uk": "UK", "united kingdom": "UK", "gb": "UK",
           "au": "AU", "australia": "AU", "ie": "IE", "ireland": "IE",
           "row": "ROW", "rest of world": "ROW", "other": "ROW"}


def load_and_harmonize(raw_dir) -> pd.DataFrame:
    """Return one table: date, ssp, advertiser_raw, revenue_usd, device, position, geo."""
    frames = []
    for ssp, s in SSP_SCHEMAS.items():
        df = pd.read_csv(Path(raw_dir) / f"{ssp}.csv", dtype=str)
        out = pd.DataFrame({
            "date": pd.to_datetime(df[s["date"]], format=s["date_fmt"]),
            "ssp": ssp,
            "advertiser_raw": df[s["advertiser"]],
            "revenue_usd": df[s["revenue"]].astype(float) * s["fx"],
            "device": df[s["device"]].str.lower().map(DEVICE_MAP),
            "position": df[s["position"]].str.lower().map(POSITION_MAP),
            "geo": df[s["geo"]].str.lower().map(GEO_MAP),
        })
        frames.append(out)
    out = pd.concat(frames, ignore_index=True)
    unmapped = out[["device", "position", "geo"]].isna().sum()
    assert unmapped.sum() == 0, f"unmapped category labels:\n{unmapped}"
    return out


# --------------------------------------------------------------------------- #
# 2. Rule-based name cleaning
# --------------------------------------------------------------------------- #
LEGAL_SUFFIXES = {"ltd", "plc", "inc", "llc", "limited", "pty"}
CAMPAIGN_TAGS = {"uk", "us", "au", "ie", "q1", "q2", "q3", "q4", "display", "prog",
                 "2024", "2025", "retarget"}
DOMAIN_RE = re.compile(r"^[a-z0-9\-]+(\.(com|co\.uk|com\.au|ie|org|net))$")


def clean_name(raw: str) -> str:
    """Normalise one raw advertiser string: strip agency prefix, IDs, domains, campaign tags,
    legal suffixes and punctuation."""
    s = str(raw).strip()
    s = re.sub(r"^[^|]*\|\s*", "", s)          # 'GroupM | Tesco'        -> 'Tesco'
    s = re.sub(r"\s*#\d+\s*$", "", s)          # 'Tesco #48213'          -> 'Tesco'
    s = s.lower()
    if DOMAIN_RE.match(s):                     # 'tesco.co.uk'           -> 'tesco'
        s = s.split(".")[0]
    s = s.replace("'", "")                     # "sainsbury's"           -> 'sainsburys'
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()  # punctuation / underscores -> spaces
    tokens = s.split()
    changed = True
    while changed and len(tokens) > 1:         # peel trailing tags / legal suffixes in any order
        changed = False
        if tokens[-1] in CAMPAIGN_TAGS or tokens[-1] in LEGAL_SUFFIXES:
            tokens.pop()
            changed = True
    return " ".join(tokens)


def match_key(clean: str) -> str:
    """Space-free key so 'nova apex motors' and 'novaapexmotors' (from a domain) line up."""
    return clean.replace(" ", "")


# --------------------------------------------------------------------------- #
# 3. Fuzzy clustering
# --------------------------------------------------------------------------- #
def _digits(s: str) -> str:
    return "".join(ch for ch in s if ch.isdigit())


def cluster_keys(key_revenue: pd.Series, threshold: float = 90, min_len: int = 5) -> dict:
    """
    Greedy clustering of match keys.
      - Process keys from highest to lowest revenue, so the cluster representative is the
        most-seen (usually correctly spelled) variant.
      - A key joins an existing cluster if fuzz.ratio >= threshold AND the digits agree
        ('island airlines 56' must not merge with 'island airlines 57').
      - Keys shorter than `min_len` only match exactly: one typo on 'ee' or 'bt' is
        indistinguishable from a different company.
    Returns {key: representative_key}.
    """
    keys = key_revenue.sort_values(ascending=False).index.tolist()
    reps, mapping = [], {}
    for k in keys:
        if len(k) >= min_len and reps:
            hits = process.extract(k, reps, scorer=fuzz.ratio, score_cutoff=threshold, limit=5)
            hit = next((r for r, _, _ in hits if _digits(r) == _digits(k)), None)
        else:
            hit = None
        if hit is None:
            reps.append(k)
            mapping[k] = k
        else:
            mapping[k] = hit
    return mapping
