"""Downloads NHANES and writes data/nhanes_intake.csv.

NHANES 2017-2018 (US CDC / NCHS, public domain). Three files joined on SEQN:
demographics, body measures, and the day-1 dietary recall.

Filters to adults 18-80 with complete records and plausible intake (800-5000
kcal) - NHANES includes non-response and partial recalls that would otherwise
come through as zeros.

See the README for why this replaced the original dataset.

    python scripts/build_dataset.py
"""

import io
import ssl
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT = DATA_DIR / "nhanes_intake.csv"

BASE_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/"

FILES = {
    "DEMO_J.xpt": ["SEQN", "RIAGENDR", "RIDAGEYR"],
    "BMX_J.xpt": ["SEQN", "BMXHT", "BMXWT", "BMXBMI"],
    "DR1TOT_J.xpt": ["SEQN", "DR1TKCAL", "DR1TPROT", "DR1TCARB", "DR1TTFAT", "DR1TSUGR"],
}

COLUMN_NAMES = {
    "RIDAGEYR": "age_years",
    "BMXHT": "height_cm",
    "BMXWT": "weight_kg",
    "BMXBMI": "bmi",
    "DR1TKCAL": "energy_kcal",
    "DR1TPROT": "protein_g",
    "DR1TCARB": "carbs_g",
    "DR1TTFAT": "fat_g",
    "DR1TSUGR": "sugar_g",
}


def _download(filename: str) -> pd.DataFrame:
    request = urllib.request.Request(
        BASE_URL + filename, headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(
        request, timeout=180, context=ssl.create_default_context()
    ) as response:
        payload = response.read()

    print(f"  {filename}: {len(payload) / 1048576:.1f} MB")
    return pd.read_sas(io.BytesIO(payload), format="xport")


def build() -> pd.DataFrame:
    print(f"Downloading NHANES 2017-2018 from {BASE_URL}")
    frames = {
        name: _download(name)[columns] for name, columns in FILES.items()
    }

    merged = (
        frames["DEMO_J.xpt"]
        .merge(frames["BMX_J.xpt"], on="SEQN")
        .merge(frames["DR1TOT_J.xpt"], on="SEQN")
    )
    print(f"\nJoined on SEQN: {len(merged)} respondents")

    merged = merged[merged["RIDAGEYR"].between(18, 80)]
    merged = merged.dropna()
    merged = merged[
        merged["DR1TKCAL"].between(800, 5000)
        & (merged["DR1TPROT"] > 0)
        & (merged["DR1TCARB"] > 0)
        & (merged["DR1TTFAT"] > 0)
    ]

    # RIAGENDR is 1 = male, 2 = female.
    merged["SEX_MALE"] = (merged["RIAGENDR"] == 1).astype(int)
    merged = merged.drop(columns=["RIAGENDR", "SEQN"]).rename(columns=COLUMN_NAMES)

    print(f"After filtering to usable adult records: {len(merged)}")
    print(
        f"Energy range: {merged['energy_kcal'].min():.0f}"
        f" - {merged['energy_kcal'].max():.0f} kcal"
    )
    return merged.reset_index(drop=True)


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    dataset = build()
    dataset.to_csv(OUTPUT, index=False)
    print(f"\nWrote {OUTPUT} ({len(dataset)} rows, {len(dataset.columns)} columns)")
