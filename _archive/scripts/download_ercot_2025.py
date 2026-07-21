"""
Download ERCOT 2025 Data via the Official ERCOT Public API
==========================================================

Fetches:
  1. DAM Settlement Point Prices  (NP4-190-CD)  — Hourly, LZ_ERCOT
  2. RTM Settlement Point Prices  (NP6-905-CD)  — 15-min SCED, LZ_ERCOT
  3. Wind Power Production         (NP4-732-CD)  — Hourly, ACTUAL_SYSTEM_WIDE
  4. Solar Power Production        (NP4-737-CD)  — Hourly, ACTUAL_SYSTEM_WIDE

All data for Jan 1, 2025 – Dec 31, 2025.

PREREQUISITES — You MUST register at https://apiexplorer.ercot.com first:
  1. Create an account (free)
  2. Subscribe to the "ERCOT Public API" product
  3. Copy your Primary Subscription Key
  4. Set three environment variables before running:
       export ERCOT_API_USERNAME="your_username"
       export ERCOT_API_PASSWORD="your_password"
       export ERCOT_API_SUBSCRIPTION_KEY="your_subscription_key"

Time resolutions of downloaded data:
  - DAM LMPs:         1 Hour
  - RTM LMPs (SCED):  15 Minutes
  - Wind Production:  1 Hour
  - Solar Production: 1 Hour
"""

import os
import sys
import json
import time
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# ── Configuration ─────────────────────────────────────────────────────────────

ERCOT_TOKEN_URL = (
    "https://ercotb2c.b2clogin.com/ercotb2c.onmicrosoft.com/"
    "B2C_1_PUBAPI-ROPC-FLOW/oauth2/v2.0/token"
)
ERCOT_API_BASE  = "https://api.ercot.com/api/public-reports"
CLIENT_ID       = "fec253ea-0d06-4272-a5e6-b478baeecd70"

# Reports to download
REPORTS = {
    "dam_spp": {
        "name": "DAM Settlement Point Prices (Hourly) — All Load Zones & Hubs",
        "path": "/np4-190-cd/dam_stlmnt_pnt_prices",
        # No server-side filter — download all LZ/HB points, filter client-side
        "filter_field": None,
        "filter_value": None,
        # Keep only load zones (LZ_*) and hub averages (HB_*)
        "client_filter_col": 2,   # column index for settlementPoint
        "client_filter_fn": lambda v: v.startswith("LZ_") or v.startswith("HB_"),
        "filename": "ercot_dam_lmp_2025.csv",
    },
    "rtm_spp": {
        "name": "RTM Settlement Point Prices — SCED (15-min) — All Load Zones & Hubs",
        "path": "/np6-905-cd/spp_node_zone_hub",
        "filter_field": None,
        "filter_value": None,
        "client_filter_col": 3,   # column index for settlementPoint in RTM
        "client_filter_fn": lambda v: v.startswith("LZ_") or v.startswith("HB_"),
        "filename": "ercot_rtm_lmp_2025.csv",
    },
    "wind": {
        "name": "Wind Power Production (Hourly, System-Wide)",
        "path": "/np4-732-cd/wpp_hrly_avrg_actl_fcast",
        "filter_field": None,
        "filter_value": None,
        "client_filter_col": None,
        "client_filter_fn": None,
        "filename": "ercot_wind_production_2025.csv",
    },
    "solar": {
        "name": "Solar Power Production (Hourly, System-Wide)",
        "path": "/np4-737-cd/spp_hrly_avrg_actl_fcast",
        "filter_field": None,
        "filter_value": None,
        "client_filter_col": None,
        "client_filter_fn": None,
        "filename": "ercot_solar_production_2025.csv",
    },
}

YEAR       = 2025
PAGE_SIZE  = 10000    # Max records per page
MAX_PAGES  = 200      # Safety cap per month chunk


# ── Authentication ────────────────────────────────────────────────────────────

def get_credentials():
    """Read ERCOT API credentials from env vars, or prompt interactively."""
    import getpass
    username = os.environ.get("ERCOT_API_USERNAME")
    password = os.environ.get("ERCOT_API_PASSWORD")
    sub_key  = os.environ.get("ERCOT_API_SUBSCRIPTION_KEY")

    if not sub_key:
        print("Enter your ERCOT API credentials (from apiexplorer.ercot.com):")
        sub_key = input("  Subscription Key: ").strip()
    if not username:
        username = input("  ERCOT Account Email (username): ").strip()
    if not password:
        password = getpass.getpass("  ERCOT Account Password (hidden): ")

    return username, password, sub_key


def get_id_token(username: str, password: str) -> str:
    """Authenticate via ERCOT B2C ROPC flow and return an ID token."""
    print("Authenticating with ERCOT API...")
    payload = {
        "username":      username,
        "password":      password,
        "grant_type":    "password",
        "scope":         f"openid {CLIENT_ID} offline_access",
        "client_id":     CLIENT_ID,
        "response_type": "id_token",
    }
    resp = requests.post(ERCOT_TOKEN_URL, data=payload, timeout=30)
    resp.raise_for_status()
    token = resp.json().get("id_token")
    if not token:
        print(f"ERROR: Token response missing 'id_token'. Full response:\n{resp.json()}")
        sys.exit(1)
    print("  ✓ Authentication successful (token valid ~1 hour)")
    return token


# ── Data Fetching ─────────────────────────────────────────────────────────────

def fetch_report(
    report_cfg: dict,
    date_from: str,
    date_to: str,
    id_token: str,
    sub_key: str,
) -> pd.DataFrame:
    """Fetch one ERCOT report for a date range, handling pagination."""

    url = ERCOT_API_BASE + report_cfg["path"]
    headers = {
        "Authorization":              f"Bearer {id_token}",
        "Ocp-Apim-Subscription-Key":  sub_key,
    }
    params = {
        "deliveryDateFrom": date_from,
        "deliveryDateTo":   date_to,
        "size":             PAGE_SIZE,
    }
    # Add optional server-side filter
    if report_cfg.get("filter_field") and report_cfg.get("filter_value"):
        params[report_cfg["filter_field"]] = report_cfg["filter_value"]

    # Get field names from first call
    field_names = None
    all_records = []
    page = 1

    while page <= MAX_PAGES:
        params["page"] = page
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=60)
        except requests.exceptions.RequestException as e:
            print(f"    [WARN] Request error on page {page}: {e}")
            break

        if resp.status_code == 401:
            print("    [ERROR] 401 Unauthorized — token may have expired.")
            return pd.DataFrame()
        if resp.status_code == 429:
            print("    [WARN] Rate limited (429). Sleeping 10s...")
            time.sleep(10)
            continue
        if resp.status_code != 200:
            print(f"    [WARN] HTTP {resp.status_code} on page {page}. Body: {resp.text[:200]}")
            break

        data = resp.json()
        records = data.get("data", [])
        if not records:
            break

        # Capture field names from first page
        if field_names is None:
            field_names = [f["name"] for f in data.get("fields", [])]

        # Apply client-side filter if configured
        col_idx = report_cfg.get("client_filter_col")
        fn      = report_cfg.get("client_filter_fn")
        if col_idx is not None and fn is not None:
            records = [r for r in records if fn(r[col_idx])]

        all_records.extend(records)

        # Check if there are more pages
        total = data.get("_meta", {}).get("totalRecords", 0)
        fetched_so_far = (page) * PAGE_SIZE  # approximate
        print(f"    page {page}: +{len(records)} kept  (page fetched, total API records: {total})")
        if fetched_so_far >= total:
            break
        page += 1
        time.sleep(0.3)  # be polite to the API

    if not all_records:
        return pd.DataFrame()
    df = pd.DataFrame(all_records, columns=field_names if field_names else None)
    return df


def download_full_year(report_key: str, report_cfg: dict, id_token: str, sub_key: str, output_dir: Path):
    """Download a full year of data for one report, month by month."""

    print(f"\n{'='*60}")
    print(f"  {report_cfg['name']}")
    print(f"  Endpoint: {report_cfg['path']}")
    print(f"{'='*60}")

    monthly_dfs = []

    for month in range(1, 13):
        date_from = f"{YEAR}-{month:02d}-01"
        # Last day of month
        if month == 12:
            date_to = f"{YEAR}-12-31"
        else:
            next_month = datetime(YEAR, month + 1, 1)
            last_day = next_month - timedelta(days=1)
            date_to = last_day.strftime("%Y-%m-%d")

        print(f"\n  [{report_key}] {date_from} → {date_to}")
        df = fetch_report(report_cfg, date_from, date_to, id_token, sub_key)
        if df.empty:
            print(f"    ⚠ No data returned for this month")
        else:
            print(f"    ✓ {len(df)} records")
            monthly_dfs.append(df)

    if monthly_dfs:
        full_df = pd.concat(monthly_dfs, ignore_index=True)
        out_path = output_dir / report_cfg["filename"]
        full_df.to_csv(out_path, index=False)
        print(f"\n  ✅ Saved {len(full_df):,} records → {out_path}")
        return full_df
    else:
        print(f"\n  ❌ No data collected for {report_key}")
        return pd.DataFrame()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Output directory
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir.parent / "data" / "ercot"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  ERCOT 2025 Data Downloader")
    print("  Using Official ERCOT Public API (api.ercot.com)")
    print(f"  Output: {output_dir}")
    print("=" * 60)

    # Get credentials and authenticate
    username, password, sub_key = get_credentials()
    id_token = get_id_token(username, password)

    # Download each report
    results = {}
    for key, cfg in REPORTS.items():
        results[key] = download_full_year(key, cfg, id_token, sub_key, output_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("  DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"  {'Report':<45} {'Records':>10}  {'Resolution'}")
    print(f"  {'-'*75}")
    resolutions = {
        "dam_spp": "1 Hour",
        "rtm_spp": "15 Minutes",
        "wind":    "1 Hour",
        "solar":   "1 Hour",
    }
    for key, cfg in REPORTS.items():
        n = len(results[key]) if not results[key].empty else 0
        res = resolutions.get(key, "?")
        print(f"  {cfg['name']:<45} {n:>10,}  {res}")
    print(f"\n  Files saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
