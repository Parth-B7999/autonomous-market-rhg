"""Fetch and clean ERCOT LZ_SOUTH DAM + RTM LMPs and PV data.

Generates synthetic data with realistic LZ_SOUTH patterns and caches to
data/processed/synthetic_week.npz for use by downstream scripts.
"""
from __future__ import annotations

import argparse

import yaml

from amrhg.data import load_or_generate_ercot_data


def main(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data = load_or_generate_ercot_data(cfg)

    sim_cfg = cfg["simulation"]
    n_days = sim_cfg["n_days"]
    n_rtm = n_days * 288
    n_dam = n_days * 24

    print(f"ERCOT LZ_SOUTH synthetic data — {sim_cfg['start_date']}, {n_days} days")
    print(f"  RTM LMP  ({n_rtm:5d} points): ${data['rtm_lmp'].min():.1f}–${data['rtm_lmp'].max():.1f}/MWh")
    print(f"  DAM LMP  ({n_dam:5d} points): ${data['dam_lmp'].min():.1f}–${data['dam_lmp'].max():.1f}/MWh")
    print(f"  PV CF     ({n_rtm:5d} points): {data['pv_capacity_factor'].min():.3f}–{data['pv_capacity_factor'].max():.3f}")
    print("  Saved to data/processed/synthetic_week.npz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate/cache ERCOT data")
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    main(args.config)
