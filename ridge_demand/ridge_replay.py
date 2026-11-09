from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "nem_weather_halfhourly_2010_2026.csv"


def main():
    data = pd.read_csv(DATA_PATH, parse_dates=["timestamp"], index_col="timestamp")
    data = data.sort_index()

    # 时间戳是半小时区间的结束时间。先减 30 分钟，再取市场日。
    interval_start = data.index - pd.Timedelta(minutes=30)
    data["market_day"] = interval_start.normalize()
    data["slot"] = interval_start.hour * 2 + interval_start.minute // 30

    print(f"读入 {len(data):,} 条半小时记录")
    print(data[["market_day", "slot", "nsw_demand", "nsw_price"]].head())
    print(data[["market_day", "slot", "nsw_demand", "nsw_price"]].tail())


if __name__ == "__main__":
    main()