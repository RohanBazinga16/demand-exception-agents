"""
Synthetic data generator for the Demand Exception Agents project.

Company: "Kaveri Foods Pvt Ltd" — a FICTIONAL Indian FMCG firm. No real company,
retailer, or person is described. No client data was used.

Outputs (written next to this file):
  item_master.csv      10 items (SKU x pack) with service class, shelf life, price
  demand_history.csv   40 IWLs (Item-Week-Location) x 104 weeks of shipments
  future_actuals.csv   the next 12 weeks (W105-W116)  -> EVALUATION ONLY.
                       No tool or agent ever reads this file.
  event_notes.csv      20 market / commercial notes (the mini-RAG corpus),
                       including distractors, an off-location note, and a
                       cancellation that does NOT update the original note.
  ground_truth.csv     the planted exception per IWL, its true cause, the
                       expected treatment bucket and the explaining note (if any).

Run:  python data/generate_data.py      (deterministic, seed=42)
"""
from __future__ import annotations

import os
from datetime import date, timedelta

import numpy as np
import pandas as pd

SEED = 42
N_HIST = 104
N_FUT = 12
HERE = os.path.dirname(os.path.abspath(__file__))
FIRST_WEEK = date(2026, 9, 28) - timedelta(weeks=N_HIST - 1)  # W001 = 2024-10-07

ITEMS = [
    # item_id, sku, pack, category, service_class, shelf_life_days, price_inr, base_units
    ("OAT500", "Rolled Oats", "500g", "Breakfast", "A", 270, 165, 1200),
    ("OAT1K", "Rolled Oats", "1kg", "Breakfast", "A", 270, 299, 600),
    ("MUS400", "Fruit Muesli", "400g", "Breakfast", "B", 240, 275, 500),
    ("MUS750", "Fruit Muesli", "750g", "Breakfast", "B", 240, 470, 300),
    ("GRA6", "Granola Bar", "6-pack", "Snacks", "A", 120, 180, 800),
    ("GRA12", "Granola Bar", "12-pack", "Snacks", "A", 120, 340, 350),
    ("POP60", "Masala Popcorn", "60g", "Snacks", "A", 90, 30, 1500),
    ("POP150", "Masala Popcorn", "150g", "Snacks", "A", 90, 65, 700),
    ("CAK250", "Eggless Cake Mix", "250g", "Baking", "C", 180, 140, 90),
    ("CAK500", "Eggless Cake Mix", "500g", "Baking", "C", 180, 250, 55),
]
LOCATIONS = {  # DC code -> (name, volume factor)
    "MUM": ("Mumbai DC", 1.00),
    "DEL": ("Delhi DC", 0.90),
    "BLR": ("Bengaluru DC", 0.70),
    "KOL": ("Kolkata DC", 0.50),
}
INTERMITTENT = {"CAK250-KOL", "CAK500-KOL", "CAK500-BLR"}


def week_start(t: int) -> date:
    return FIRST_WEEK + timedelta(weeks=t - 1)


def seasonal_factor(category: str, d: date) -> float:
    """Annual pattern; festive (Oct-Nov) bump for Snacks/Baking, winter lift for Breakfast."""
    doy = d.timetuple().tm_yday
    if category == "Breakfast":
        return 1.0 + 0.10 * np.cos(2 * np.pi * (doy - 15) / 365.25)  # peaks mid-Jan
    festive = np.exp(-0.5 * ((doy - 305) / 14.0) ** 2)  # ~1 Nov
    amp = 0.45 if category == "Snacks" else 0.70
    return 1.0 + 0.05 * np.sin(2 * np.pi * doy / 365.25) + amp * festive


def planted_multiplier(iwl: str, t: int, d: date) -> float:
    """Planted exception patterns. t runs 1..116 (history + future)."""
    m = 1.0
    if iwl == "OAT1K-MUM" and t >= 97:          # new listing -> level shift up
        m *= 1.40
    if iwl == "MUS400-DEL":                      # competitor launch -> level shift down
        if t >= 100:
            m *= 0.62
        elif t >= 96:
            m *= 0.70
    if iwl == "GRA12-MUM" and t >= 98:           # level shift down, NO note explains it
        m *= 0.70
    if iwl == "GRA6-BLR":                        # past promo spike + pantry-loading dip
        if t in (101, 102):
            m *= 2.20
        elif t in (103, 104):
            m *= 0.85
    if iwl == "POP150-MUM" and t in (106, 107, 108):   # upcoming promo (real)
        m *= 1.60
    # POP150-DEL: promo was planned, then cancelled -> nothing happens.
    if iwl == "OAT500-KOL" and t >= 95:          # price rise on 500g -> mix moves to 1kg
        m *= max(0.70, 1 - 0.10 * (t - 94))
    if iwl == "OAT1K-KOL" and t >= 95:
        m *= min(1.30, 1 + 0.10 * (t - 94))
    if iwl == "MUS750-MUM" and t >= 93:          # monthly order cycle -> weekly phasing
        m *= 1.60 if d.day <= 7 else 0.82
    if iwl == "GRA12-DEL" and t > 92:            # slow erosion, no event
        m *= 1 - 0.018 * (t - 92)
    return m


def generate():
    rng = np.random.default_rng(SEED)
    rows = []
    for item_id, sku, pack, cat, svc, shelf, price, base in ITEMS:
        for loc, (_, lf) in LOCATIONS.items():
            iwl = f"{item_id}-{loc}"
            lvl = base * lf * rng.uniform(0.9, 1.1)
            for t in range(1, N_HIST + N_FUT + 1):
                d = week_start(t)
                mu = lvl * seasonal_factor(cat, d) * planted_multiplier(iwl, t, d)
                if iwl in INTERMITTENT:
                    # zero-inflated, lumpy institutional orders
                    if rng.random() < 0.45:
                        y = 0
                    else:
                        y = rng.poisson(mu * 1.6) * (3 if rng.random() < 0.12 else 1)
                else:
                    y = max(0, rng.normal(mu, mu * 0.08))
                rows.append(dict(iwl_id=iwl, item_id=item_id, location=loc, week=t,
                                 week_start=d.isoformat(), units=int(round(y))))
    df = pd.DataFrame(rows)
    hist = df[df.week <= N_HIST].reset_index(drop=True)
    fut = df[df.week > N_HIST].reset_index(drop=True)

    items = pd.DataFrame(
        [dict(item_id=i, sku=s, pack=p, category=c, service_class=sv,
              shelf_life_days=sl, unit_price_inr=pr) for i, s, p, c, sv, sl, pr, _ in ITEMS])

    W = lambda t: f"W{t:03d} (w/c {week_start(t).isoformat()})"  # noqa: E731
    notes = [
        ("EV01", 95, "listing", "MUM", "OAT1K", 97, None, "active",
         "MetroMart Mumbai listing gain - Rolled Oats 1kg",
         f"MetroMart (modern trade chain) has added Rolled Oats 1kg to its Mumbai planogram across 120 stores. "
         f"First orders ship {W(97)}. Listing is permanent; expect a sustained step-up in Mumbai DC volume for the 1kg pack."),
        ("EV02", 95, "competitor", "DEL", "MUS400", 96, None, "active",
         "GrainCo launches value muesli in Delhi NCR",
         f"Competitor GrainCo launched a 400g fruit muesli at an 18% lower shelf price in Delhi NCR from {W(96)}. "
         f"Early trade feedback suggests share loss for our Fruit Muesli 400g in the north."),
        ("EV03", 99, "promo", "BLR", "GRA6", 101, 102, "active",
         "Bengaluru buy-one-get-one on Granola Bar 6-pack",
         f"BOGO promotion on Granola Bar 6-pack at Bengaluru key accounts, {W(101)} to {W(102)} only. "
         f"One-off; no follow-on activity planned. Expect pantry loading and a short dip afterwards."),
        ("EV04", 100, "promo", "MUM", "POP150", 106, 108, "active",
         "Festive promotion - Masala Popcorn 150g, Mumbai",
         f"Diwali-season 25% off promotion on Masala Popcorn 150g across Mumbai modern trade, {W(106)} to {W(108)}. "
         f"Confirmed with retailer; POS material printed. Category team expects roughly +60% uplift in promo weeks."),
        ("EV05", 99, "promo", "DEL", "POP150", 106, 108, "active",
         "Festive promotion - Masala Popcorn 150g, Delhi",
         f"Planned Diwali-season 25% off promotion on Masala Popcorn 150g in Delhi NCR modern trade, {W(106)} to {W(108)}. "
         f"Category team expects roughly +60% uplift in promo weeks."),
        ("EV06", 103, "promo_update", "DEL", "POP150", None, None, "active",
         "Update: Delhi popcorn activity",
         "Retailer budget for the Delhi NCR festive window has been reallocated to beverages. "
         "The planned Masala Popcorn 150g activity will not go ahead this season."),
        ("EV07", 93, "price", "KOL", "OAT500", 95, None, "active",
         "Price increase - Rolled Oats 500g, East",
         f"List price of Rolled Oats 500g rises 12% in the East region from {W(95)}; 1kg price unchanged. "
         f"Trade expects shoppers to trade up to the 1kg pack, which is now better value per kg."),
        ("EV08", 91, "customer", "MUM", "MUS750", 93, None, "active",
         "MetroMart Mumbai moves to monthly replenishment",
         f"From {W(93)}, MetroMart Mumbai places one consolidated order at the start of each month for Fruit Muesli 750g "
         f"instead of weekly orders. Monthly volume unchanged; weekly pattern will be lumpy."),
        ("EV09", 88, "promo", "MUM", "OAT500", 90, 91, "active",
         "Mumbai oats sampling drive",
         f"In-store sampling for Rolled Oats 500g in Mumbai, {W(90)} to {W(91)}. Small activity, minimal volume impact expected."),
        ("EV10", 100, "article", "ALL", "OAT", None, None, "active",
         "Industry report: oats demand rising nationally",
         "An industry tracker reports that oats consumption in urban India grew in double digits this year, "
         "driven by health positioning. Growth is broad-based across metros. No company-specific activity."),
        ("EV11", 60, "customer", "KOL", "CAK", None, None, "active",
         "Cake mix demand in the East is institutional",
         "Most Eggless Cake Mix volume through Kolkata DC goes to small bakeries and caterers who order irregularly "
         "and in bulk. Weekly demand is lumpy with many zero weeks; aggregate monthly volume is stable."),
        ("EV12", 68, "supply", "DEL", "GRA12", 70, 71, "active",
         "Delhi DC stock-out - Granola Bar 12-pack",
         f"Packaging shortage caused a stock-out of Granola Bar 12-pack at Delhi DC in {W(70)} to {W(71)}. Resolved."),
        ("EV13", 92, "seasonal", "ALL", "ALL", 103, 112, "active",
         "Festive season outlook",
         "Festive season (Navratri to Diwali) typically lifts Snacks and Baking volumes across all regions in "
         "October-November. Baseline forecasts already include normal seasonality."),
        ("EV14", 78, "supply", "BLR", "ALL", 80, 80, "active",
         "Bengaluru flooding - delivery disruption",
         f"Heavy rain disrupted secondary deliveries from Bengaluru DC in {W(80)}. Volumes recovered the following week."),
        ("EV15", 102, "customer", "DEL", "GRA12", 110, None, "active",
         "Planogram review scheduled - Delhi snacks bay",
         f"A key Delhi account will review its snacks planogram in {W(110)}. No change to ranging yet."),
        ("EV16", 100, "competitor", "DEL", "MUS400", 100, None, "active",
         "GrainCo deepens price cut",
         f"GrainCo cut its 400g muesli price by a further 7% in Delhi NCR from {W(100)}, widening the gap to 25%."),
        ("EV17", 101, "promo", "MUM", "OAT500", 103, 104, "active",
         "Rolled Oats 500g weekend offer - Mumbai only",
         f"Weekend price-off on Rolled Oats 500g in Mumbai modern trade, {W(103)} to {W(104)}. Mumbai only."),
        ("EV18", 97, "supply", "ALL", "POP", None, None, "active",
         "Corn input cost update",
         "Corn input costs rose 6% this quarter. No list-price change planned for popcorn this year."),
        ("EV19", 96, "customer", "BLR", "ALL", None, None, "active",
         "Quick-commerce growth in Bengaluru",
         "Quick-commerce platforms now account for a growing share of Bengaluru orders. Order sizes are smaller and more frequent."),
        ("EV20", 90, "listing", "KOL", "GRA6", 120, None, "active",
         "Possible listing - Granola Bar 6-pack, Kolkata",
         "Early-stage discussion with a Kolkata chain about listing Granola Bar 6-pack next year. Nothing agreed."),
    ]
    ev = pd.DataFrame(notes, columns=["note_id", "published_week", "type", "location", "item_scope",
                                      "effective_from_week", "effective_to_week", "status", "title", "text"])
    ev["published_date"] = ev.published_week.apply(lambda t: week_start(int(t)).isoformat())

    gt = []
    planted = {
        "OAT1K-MUM": ("LEVEL_SHIFT_UP", "Action-Required", "EV01", "relevel"),
        "MUS400-DEL": ("LEVEL_SHIFT_DOWN", "Action-Required", "EV02|EV16", "relevel"),
        "GRA12-MUM": ("LEVEL_SHIFT_DOWN", "Action-Required", "", "relevel"),
        "GRA6-BLR": ("ONE_OFF_SPIKE", "Action-Required", "EV03", "cleanse_spike"),
        "POP150-MUM": ("UPCOMING_PROMO", "Action-Required", "EV04", "promo_uplift"),
        "POP150-DEL": ("NONE", "Low-Touch", "EV06", "no_change"),
        "OAT500-KOL": ("PACK_MIX_SHIFT", "Action-Required", "EV07", "relevel"),
        "OAT1K-KOL": ("PACK_MIX_SHIFT", "Action-Required", "EV07", "relevel"),
        "MUS750-MUM": ("WEEKLY_PHASING", "Monitor", "EV08", "rephase_weekly"),
        "CAK250-KOL": ("STRUCTURAL_NOISE", "Structurally-Noisy", "EV11", "aggregate"),
        "CAK500-KOL": ("STRUCTURAL_NOISE", "Structurally-Noisy", "EV11", "aggregate"),
        "CAK500-BLR": ("STRUCTURAL_NOISE", "Structurally-Noisy", "", "aggregate"),
        "GRA12-DEL": ("BIAS_DRIFT", "Monitor", "", "no_change"),
        "OAT500-BLR": ("NONE", "Low-Touch", "", "no_change"),
    }
    notes_on = {"GRA6-BLR": "One-time miss, but the spike leaked into the baseline: cleanse it, do not chase it.",
                "POP150-DEL": "Trap: EV05 (promo planned) was never withdrawn; EV06 cancels it in different words.",
                "OAT500-BLR": "Distractor: EV10 (national article) and EV17 (Mumbai-only offer) look relevant but are not.",
                "GRA12-MUM": "Gap: a real step-down with no explaining note. Correct diagnosis is 'unexplained'.",
                "CAK500-BLR": "Lumpy demand; EV11 covers Kolkata only."}
    for iwl in hist.iwl_id.unique():
        cause, bucket, evid, action = planted.get(iwl, ("NONE", "Low-Touch", "", "no_change"))
        gt.append(dict(iwl_id=iwl, planted=iwl in planted, true_cause=cause, expected_bucket=bucket,
                       explaining_notes=evid, correct_action=action, comment=notes_on.get(iwl, "")))
    gt = pd.DataFrame(gt)

    items.to_csv(os.path.join(HERE, "item_master.csv"), index=False)
    hist.to_csv(os.path.join(HERE, "demand_history.csv"), index=False)
    fut.to_csv(os.path.join(HERE, "future_actuals.csv"), index=False)
    ev.to_csv(os.path.join(HERE, "event_notes.csv"), index=False)
    gt.to_csv(os.path.join(HERE, "ground_truth.csv"), index=False)
    print(f"history {hist.shape}, future {fut.shape}, notes {len(ev)}, planted {gt.planted.sum()}")


if __name__ == "__main__":
    generate()
