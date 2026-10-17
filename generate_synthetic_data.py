"""
Synthetic advertiser revenue data generator.

Simulates the raw exports of 6 SSPs/DSPs for an advertiser-classification project:
  - messy, inconsistent advertiser names (legal suffixes, campaign tags, domains, typos, agency prefixes)
  - different column names / date formats / currencies / category labels per SSP
  - heavily skewed (Pareto-like) revenue: a few hundred big brands + a long tail
  - seasonality and per-vertical growth so MoM / QoQ / YoY analysis is meaningful
  - device, ad position and geo (US, UK, AU, IE, ROW) dimensions

No real data is used. Brand names are public company names; all revenue is invented.

Outputs (under --out, default ./data):
  raw/ssp_a.csv ... raw/ssp_f.csv   <- what the analyst would receive
  ground_truth.csv                  <- canonical advertiser -> vertical (for evaluation only)
  name_variants_truth.csv           <- (ssp, raw name) -> canonical advertiser (to score dedupe)

Usage:
  python src/generate_synthetic_data.py --n-tail 2000 --seed 42
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

OTHER = "Other / Unclassified"

# --------------------------------------------------------------------------- #
# Taxonomy (v1) and head brands
# --------------------------------------------------------------------------- #
HEAD_BRANDS = {
    "Automotive & Manufacturing": [
        "Ford", "Toyota", "BMW", "Volkswagen", "Mercedes-Benz", "Audi", "Honda", "Nissan",
        "Hyundai", "Kia", "Tesla", "Land Rover", "Volvo", "Caterpillar", "Siemens"],
    "Finance": [
        "Barclays", "HSBC", "Lloyds Bank", "NatWest", "Santander", "Visa", "Mastercard",
        "PayPal", "Aviva", "Admiral", "Revolut", "Monzo", "American Express", "Vanguard",
        "Hargreaves Lansdown"],
    "Internet & Telecom": [
        "Vodafone", "BT", "EE", "O2", "Three", "Verizon", "AT&T", "Telstra", "Optus",
        "Microsoft", "Dropbox", "Zoom", "NordVPN", "GoDaddy", "Eir"],
    "Healthcare": [
        "Pfizer", "Bupa", "Johnson & Johnson", "Novartis", "GSK", "AstraZeneca",
        "Vitality Health", "Nuffield Health", "Medibank", "Kaiser Permanente", "Cigna",
        "Haleon", "Bayer", "Roche", "Specsavers"],
    "Apparel & Fashion": [
        "Nike", "Adidas", "Zara", "H&M", "ASOS", "Gucci", "Levi's", "Uniqlo", "Primark",
        "Next", "Burberry", "Ralph Lauren", "Puma", "Under Armour", "River Island"],
    "Food & Beverage": [
        "Coca-Cola", "Pepsi", "Nestle", "McDonald's", "KFC", "Domino's", "Heineken",
        "Cadbury", "Kellogg's", "Danone", "Starbucks", "Burger King", "Costa Coffee",
        "Red Bull", "Walkers"],
    "Travel & Tourism": [
        "Booking.com", "Expedia", "Emirates", "British Airways", "Ryanair", "easyJet",
        "Qantas", "Marriott", "Hilton", "Airbnb", "Tripadvisor", "Jet2", "Qatar Airways",
        "Hertz", "Virgin Atlantic"],
    "Entertainment & Media": [
        "Netflix", "Disney+", "Spotify", "Warner Bros", "Universal Pictures", "Paramount",
        "Amazon Prime Video", "Sony Pictures", "BBC iPlayer", "ITV", "Apple TV+", "Audible",
        "Ticketmaster", "Xbox", "PlayStation"],
    # Rule: sports-betting-led brands (Paddy Power) -> Sports; other gambling -> Other
    "Sports": [
        "Sky Sports", "Premier League", "NBA", "NFL", "UEFA", "Formula 1", "DAZN",
        "TNT Sports", "ESPN", "Wimbledon", "Cricket Australia", "Rugby Football Union",
        "UFC", "Paddy Power", "PGA Tour"],
    # Rule: supermarkets, department stores and marketplaces -> Retail & E-commerce
    "Retail & E-commerce": [
        "Amazon", "eBay", "Tesco", "Walmart", "Argos", "Currys", "IKEA", "Sainsbury's",
        "Asda", "Aldi", "Lidl", "Target", "Best Buy", "Etsy", "Temu"],
    OTHER: [
        "Bet365", "William Hill", "Oxfam", "Cancer Research UK", "Open University",
        "Rightmove", "Zoopla", "Hays Recruitment", "Greenpeace", "Shelter"],
}
VERTICALS = [v for v in HEAD_BRANDS if v != OTHER]

# --------------------------------------------------------------------------- #
# Long-tail name generation (fictional small advertisers with vertical keywords)
# --------------------------------------------------------------------------- #
MODIFIERS = ["North", "Pacific", "Atlantic", "Metro", "Prime", "Nova", "Blue", "Crown",
             "Pioneer", "Harbour", "Summit", "Evergreen", "Royal", "Urban", "Clearview"]
TAIL_POOLS = {  # (stems, suffixes)
    "Automotive & Manufacturing": (
        ["Apex", "Ironclad", "Motive", "Torque", "Vector", "Granite", "Rivet", "Drivetrain"],
        ["Motors", "Auto Group", "Industrial", "Engineering", "Tyres", "Fabrication", "Machinery"]),
    "Finance": (
        ["Capital", "Ledger", "Trust", "Sterling", "Fidelity", "Anchor", "Meridian", "Credit"],
        ["Bank", "Insurance", "Wealth", "Mortgages", "Financial", "Loans", "Investments"]),
    "Internet & Telecom": (
        ["Cloud", "Net", "Byte", "Signal", "Fibre", "Data", "Pixel", "Link"],
        ["Broadband", "Mobile", "Hosting", "Networks", "Software", "Telecom", "Digital"]),
    "Healthcare": (
        ["Vital", "Care", "Pulse", "Remedy", "Wellbeing", "Cura", "Bright", "Medi"],
        ["Clinic", "Pharma", "Health", "Dental", "Therapeutics", "Hospital Group", "Labs"]),
    "Apparel & Fashion": (
        ["Thread", "Loom", "Velvet", "Denim", "Tailor", "Atelier", "Silk", "Stitch"],
        ["Apparel", "Clothing", "Fashion", "Footwear", "Boutique", "Menswear", "Outfitters"]),
    "Food & Beverage": (
        ["Golden", "Fresh", "Harvest", "Spice", "Brew", "Olive", "Smoke", "Orchard"],
        ["Foods", "Kitchen", "Beverages", "Brewery", "Bakery", "Snacks", "Coffee Roasters"]),
    "Travel & Tourism": (
        ["Wander", "Voyage", "Horizon", "Island", "Compass", "Escape", "Journey", "Coastal"],
        ["Holidays", "Airlines", "Tours", "Resorts", "Hotels", "Cruises", "Travel"]),
    "Entertainment & Media": (
        ["Reel", "Stage", "Echo", "Spotlight", "Studio", "Encore", "Screen", "Playback"],
        ["Pictures", "Streaming", "Records", "Studios", "Media", "Games", "Productions"]),
    "Sports": (
        ["Champion", "Victory", "Athletic", "Stadium", "Peak", "Sprint", "Match", "Pitch"],
        ["FC", "Sports Club", "League", "Racing", "Fitness", "Sports Network", "Cricket Club"]),
    "Retail & E-commerce": (
        ["Bargain", "Basket", "Cart", "Market", "Outlet", "Value", "Shopfront", "Deal"],
        ["Stores", "Mart", "Marketplace", "Superstore", "Direct", "Shop", "Retail"]),
    OTHER: (
        ["Hope", "Learning", "Property", "Careers", "Community", "Legacy", "Bridge", "Open"],
        ["Foundation", "Academy", "Estates", "Recruitment", "Charity", "Homes", "Trust Fund"]),
}
TAIL_SHARE = {v: 1.0 for v in VERTICALS} | {OTHER: 0.7}

# --------------------------------------------------------------------------- #
# Revenue behaviour
# --------------------------------------------------------------------------- #
VERTICAL_SIZE = {
    "Automotive & Manufacturing": 1.3, "Finance": 1.4, "Internet & Telecom": 1.2,
    "Healthcare": 0.7, "Apparel & Fashion": 0.9, "Food & Beverage": 1.0,
    "Travel & Tourism": 1.0, "Entertainment & Media": 1.0, "Sports": 0.8,
    "Retail & E-commerce": 1.6, OTHER: 0.5,
}
GROWTH = {  # annual growth multiplier
    "Automotive & Manufacturing": 0.95, "Finance": 1.03, "Internet & Telecom": 1.05,
    "Healthcare": 1.08, "Apparel & Fashion": 0.98, "Food & Beverage": 1.00,
    "Travel & Tourism": 1.15, "Entertainment & Media": 1.10, "Sports": 1.07,
    "Retail & E-commerce": 1.12, OTHER: 1.00,
}
SEASON = {  # month 1..12 multipliers (normalised at use)
    "Retail & E-commerce": [.8, .75, .8, .85, .9, .9, .85, .9, .9, 1, 1.4, 1.7],
    "Travel & Tourism": [1.3, 1.2, 1, .9, .9, 1.1, 1.2, 1.1, .9, .8, .8, .9],
    "Sports": [1, 1, .9, .9, .8, .8, .8, 1.3, 1.3, 1.2, 1.1, 1],
    "Apparel & Fashion": [.9, .8, .9, 1, 1, .9, .9, 1, 1, 1.1, 1.3, 1.4],
    "Finance": [1.2, 1.1, 1.1, 1, .9, .9, .9, .9, 1, 1, 1, .9],
}
FLAT = [1.0] * 12

DEVICES, DEVICE_P = ["desktop", "mobile", "tablet"], [0.35, 0.55, 0.10]
POSITIONS = ["atf", "in_content", "btf", "sidebar", "sticky"]
POSITION_P = [0.30, 0.30, 0.20, 0.12, 0.08]
GEOS, GEO_BASE = ["US", "UK", "AU", "IE", "ROW"], np.array([0.35, 0.30, 0.15, 0.05, 0.15])
GBP_TO_USD = 1.27

# --------------------------------------------------------------------------- #
# Per-SSP "export formats" (each SSP labels things differently)
# --------------------------------------------------------------------------- #
LABELS = [
    dict(device={"desktop": "Desktop", "mobile": "Mobile", "tablet": "Tablet"},
         position={"atf": "Above the fold", "in_content": "In-article", "btf": "Below the fold",
                   "sidebar": "Sidebar", "sticky": "Sticky footer"},
         geo={"US": "US", "UK": "UK", "AU": "AU", "IE": "IE", "ROW": "ROW"}),
    dict(device={"desktop": "desktop", "mobile": "smartphone", "tablet": "tablet"},
         position={"atf": "top", "in_content": "middle", "btf": "bottom",
                   "sidebar": "right_rail", "sticky": "anchor"},
         geo={"US": "United States", "UK": "United Kingdom", "AU": "Australia",
              "IE": "Ireland", "ROW": "Rest of World"}),
    dict(device={"desktop": "DESK", "mobile": "MOB", "tablet": "TAB"},
         position={"atf": "ATF", "in_content": "INART", "btf": "BTF", "sidebar": "SIDE",
                   "sticky": "STICKY"},
         geo={"US": "us", "UK": "gb", "AU": "au", "IE": "ie", "ROW": "other"}),
]
SSPS = {
    "ssp_a": dict(cols=["date", "advertiser", "revenue_usd", "device", "position", "country"],
                  date_fmt="%Y-%m-%d", fx=1.0, labels=0,
                  styles={"title": .6, "upper_legal": .3, "typo": .1}),
    "ssp_b": dict(cols=["Date", "Advertiser Name", "Revenue (GBP)", "Device Type", "Ad Slot", "Geo"],
                  date_fmt="%d/%m/%Y", fx=1 / GBP_TO_USD, labels=1,
                  styles={"upper_legal": .7, "id": .3}),
    "ssp_c": dict(cols=["day", "buyer", "net_revenue", "platform", "placement", "market"],
                  date_fmt="%Y%m%d", fx=1.0, labels=2,
                  styles={"campaign": .7, "title": .3}),
    "ssp_d": dict(cols=["report_date", "advertiser_name", "revenue", "device_category",
                        "ad_position", "geo_country"],
                  date_fmt="%m-%d-%Y", fx=1.0, labels=0,
                  styles={"domain": .5, "title": .4, "typo": .1}),
    "ssp_e": dict(cols=["Day", "Buyer / Advertiser", "Earnings (USD)", "Device", "Placement", "Region"],
                  date_fmt="%d %b %Y", fx=1.0, labels=1,
                  styles={"agency": .5, "title": .3, "upper_legal": .2}),
    "ssp_f": dict(cols=["dt", "adv", "rev", "dev", "pos", "geo"],
                  date_fmt="%Y-%m-%d", fx=1.0, labels=2,
                  styles={"title": .4, "campaign": .3, "typo": .2, "upper_legal": .1}),
}
CAMPAIGN_TAGS = ["uk", "us", "au", "q1", "q2", "q3", "q4", "display", "prog", "2024", "2025", "retarget"]
AGENCIES = ["GroupM", "Publicis", "Omnicom", "WPP", "Dentsu", "Havas"]
LEGAL = ["LTD", "PLC", "INC", "LLC", "LIMITED", "PTY LTD"]
TLDS = [".com", ".co.uk", ".com.au", ".ie"]


def pick(rng, seq):
    return str(seq[rng.integers(len(seq))])


def make_variant(name: str, style: str, rng) -> str:
    """Return a messy version of a canonical advertiser name."""
    if style == "title":
        return name
    if style == "upper_legal":
        return f"{name.upper()} {pick(rng, LEGAL)}"
    if style == "campaign":
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        return f"{slug}_{pick(rng, CAMPAIGN_TAGS)}"
    if style == "domain":
        return re.sub(r"[^a-z0-9]", "", name.lower()) + pick(rng, TLDS)
    if style == "agency":
        return f"{pick(rng, AGENCIES)} | {name}"
    if style == "id":
        return f"{name} #{rng.integers(1000, 99999)}"
    if style == "typo" and len(name) > 4:
        i = int(rng.integers(1, len(name) - 2))
        if rng.random() < 0.5:
            return name[:i] + name[i + 1:]                      # dropped char
        return name[:i] + name[i + 1] + name[i] + name[i + 2:]  # swapped chars
    return name


def make_tail(rng, n_tail):
    verts = list(TAIL_POOLS)
    p = np.array([TAIL_SHARE[v] for v in verts], dtype=float)
    counts = rng.multinomial(n_tail, p / p.sum())
    used = {b for bs in HEAD_BRANDS.values() for b in bs}
    out = []
    for v, c in zip(verts, counts):
        stems, sufs = TAIL_POOLS[v]
        for _ in range(c):
            while True:
                name = f"{pick(rng, stems)} {pick(rng, sufs)}"
                if rng.random() < 0.5:
                    name = f"{pick(rng, MODIFIERS)} {name}"
                if name in used:
                    name = f"{name} {rng.integers(2, 99)}"
                if name not in used:
                    break
            used.add(name)
            out.append((name, v))
    return out


def build_advertisers(rng, n_tail):
    rows = [(b, v, True) for v, bs in HEAD_BRANDS.items() for b in bs]
    rows += [(n, v, False) for n, v in make_tail(rng, n_tail)]
    adv = pd.DataFrame(rows, columns=["canonical_advertiser", "vertical", "is_head"])
    mu = np.where(adv.is_head, 12.0, 8.0)          # head ~ $160k/yr median, tail ~ $3k/yr
    sigma = np.where(adv.is_head, 0.9, 1.3)
    size = adv.vertical.map(VERTICAL_SIZE).values
    adv["base_annual_revenue_usd"] = rng.lognormal(mu, sigma) * size
    return adv


def generate_rows(adv, rng, start_year, years):
    months = pd.period_range(f"{start_year}-01", periods=12 * years, freq="M")
    names = list(SSPS)
    rows = []
    for a in adv.itertuples(index=False):
        season = np.array(SEASON.get(a.vertical, FLAT), dtype=float)
        season /= season.mean()
        n_ssp = int(rng.integers(2, 5)) if a.is_head else int(rng.choice([1, 2], p=[.75, .25]))
        my_ssps = list(rng.choice(names, size=n_ssp, replace=False))
        shares = rng.dirichlet(np.ones(n_ssp))
        geo_w = rng.dirichlet(GEO_BASE * 6)
        p_active = 0.95 if a.is_head else 0.60      # tail runs campaign bursts
        for mi, m in enumerate(months):
            if rng.random() > p_active:
                continue
            month_rev = (a.base_annual_revenue_usd / 12 * season[m.month - 1]
                         * GROWTH[a.vertical] ** (mi / 12) * rng.lognormal(0, 0.25))
            for ssp, share in zip(my_ssps, shares):
                n = int(rng.integers(1, 4 if a.is_head else 3))
                parts = rng.dirichlet(np.ones(n)) * month_rev * share
                styles = SSPS[ssp]["styles"]
                for part in parts:
                    style = str(rng.choice(list(styles), p=list(styles.values())))
                    rows.append((
                        pd.Timestamp(m.year, m.month, int(rng.integers(1, m.days_in_month + 1))),
                        ssp, make_variant(a.canonical_advertiser, style, rng),
                        a.canonical_advertiser, part,
                        str(rng.choice(DEVICES, p=DEVICE_P)),
                        str(rng.choice(POSITIONS, p=POSITION_P)),
                        str(rng.choice(GEOS, p=geo_w)),
                    ))
    return pd.DataFrame(rows, columns=["date", "ssp", "advertiser_raw", "canonical_advertiser",
                                       "revenue_usd", "device", "position", "geo"])


def write_outputs(adv, df, out: Path):
    raw_dir = out / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for ssp, cfg in SSPS.items():
        sub = df[df.ssp == ssp]
        lab = LABELS[cfg["labels"]]
        exp = pd.DataFrame({
            cfg["cols"][0]: sub.date.dt.strftime(cfg["date_fmt"]),
            cfg["cols"][1]: sub.advertiser_raw,
            cfg["cols"][2]: (sub.revenue_usd * cfg["fx"]).round(2),
            cfg["cols"][3]: sub.device.map(lab["device"]),
            cfg["cols"][4]: sub.position.map(lab["position"]),
            cfg["cols"][5]: sub.geo.map(lab["geo"]),
        }).sample(frac=1, random_state=0)           # shuffle like a real export
        exp.to_csv(raw_dir / f"{ssp}.csv", index=False)

    adv[["canonical_advertiser", "vertical", "is_head"]].to_csv(out / "ground_truth.csv", index=False)
    (df[["ssp", "advertiser_raw", "canonical_advertiser"]].drop_duplicates()
       .to_csv(out / "name_variants_truth.csv", index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--n-tail", type=int, default=2000, help="number of long-tail advertisers")
    ap.add_argument("--start-year", type=int, default=2024)
    ap.add_argument("--years", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("data"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    adv = build_advertisers(rng, args.n_tail)
    df = generate_rows(adv, rng, args.start_year, args.years)
    write_outputs(adv, df, args.out)

    tot = df.revenue_usd.sum()
    head = df[df.canonical_advertiser.isin(adv[adv.is_head].canonical_advertiser)].revenue_usd.sum()
    print(f"advertisers: {len(adv):,} | rows: {len(df):,} | raw name variants: "
          f"{df.advertiser_raw.nunique():,} | total revenue: ${tot:,.0f}")
    print(f"head brands ({adv.is_head.sum()}) = {head / tot:.0%} of revenue")
    print((df.groupby(df.canonical_advertiser.map(adv.set_index('canonical_advertiser').vertical))
             .revenue_usd.sum().div(tot).sort_values(ascending=False).map("{:.1%}".format)).to_string())


if __name__ == "__main__":
    main()
