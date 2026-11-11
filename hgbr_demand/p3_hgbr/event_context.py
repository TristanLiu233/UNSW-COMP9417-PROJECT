"""Descriptive overlays of pre-specified AEMO event windows (not used in modelling)."""
import argparse
from pathlib import Path

import pandas as pd

EVENTS = (
    ("2021-10-01", "Five-minute settlement begins",
     "https://www.aemo.com.au/initiatives/major-programs/past-major-programs/five-minute-settlement"),
    ("2022-06-15", "NEM spot market suspension begins",
     "https://aemo.com.au/newsroom/media-release/aemo-suspends-nem-wholesale-market"),
    ("2023-03-17", "NSW market suspension",
     "https://www.aemo.com.au/energy-systems/electricity/national-electricity-market-nem/nem-events-and-reports/market-event-reports"),
    ("2026-03-23", "NSW market suspension",
     "https://www.aemo.com.au/energy-systems/electricity/national-electricity-market-nem/nem-events-and-reports/market-event-reports"),
)


def overlay(replay_dir):
    daily = pd.read_csv(replay_dir / "daily_monitors.csv", parse_dates=["market_day"]).set_index("market_day")
    alarms = pd.read_csv(replay_dir / "drift_alarms.csv", parse_dates=["observed_day"])
    records = []
    for stamp, title, url in EVENTS:
        day = pd.Timestamp(stamp)
        delta = (alarms.observed_day - day).dt.days
        before = daily.loc[day - pd.Timedelta(days=30):day-pd.Timedelta(days=1)]
        after = daily.loc[day:day+pd.Timedelta(days=29)]
        result = {"event_day": day.date(), "event": title, "source_url": url,
                  "mean_frozen_daily_mape_before_pct": before.frozen_mape_pct.mean(),
                  "mean_frozen_daily_mape_from_event_pct": after.frozen_mape_pct.mean(),
                  "change_pp": after.frozen_mape_pct.mean()-before.frozen_mape_pct.mean()}
        for detector in ("input_only", "frozen_error_cusum"):
            nearby = alarms.loc[(alarms.detector == detector) & delta.between(-30,29)]
            result[f"{detector}_alerts_in_window"] = "; ".join(nearby.observed_day.dt.strftime("%Y-%m-%d"))
        records.append(result)
    pd.DataFrame(records).to_csv(replay_dir / "event_context_descriptive_only.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay_dir", type=Path)
    overlay(parser.parse_args().replay_dir)
