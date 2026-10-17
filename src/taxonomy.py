"""The 10 reporting verticals (v1) plus an 'Other' bucket, and the boundary rules behind them."""

OTHER = "Other / Unclassified"

VERTICALS = [
    "Automotive & Manufacturing",
    "Finance",
    "Internet & Telecom",
    "Healthcare",
    "Apparel & Fashion",
    "Food & Beverage",
    "Travel & Tourism",
    "Entertainment & Media",
    "Sports",
    "Retail & E-commerce",
]
LABELS = VERTICALS + [OTHER]

# Principle: classify by WHAT THE ADVERTISER SELLS, not where the ad runs or the sales channel.
BOUNDARY_RULES = [
    "Supermarkets, department stores and marketplaces (Tesco, Walmart, Amazon, eBay) -> Retail & E-commerce",
    "Food & Beverage is for producers and restaurant/food brands (Nestle, McDonald's)",
    "Fashion retailers (ASOS, H&M) -> Apparel & Fashion: product category outweighs the channel",
    "Sports apparel (Nike, Adidas) -> Apparel & Fashion; sports media, leagues, clubs -> Sports",
    "Sports-betting-led brands (Paddy Power) -> Sports; other gambling (Bet365, William Hill) -> Other",
    "Streaming and media -> Entertainment & Media; ISPs and tech platforms -> Internet & Telecom",
    "Charities, education, property, recruitment and anything unclear -> Other / Unclassified",
]
