import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE.parent / "data" / "nem_weather_halfhourly_2010_2026.csv"
DEFAULT_OUTPUT = HERE / "outputs" / "day_type_clustering"

SLOTS_PER_DAY = 48
TRAIN_YEARS = (2010, 2011, 2012, 2013)
K_VALUES = range(2, 11)
PCA_VARIANCE = 0.95
SEED = 42
N_INIT = 20


def load_daily_profiles(csv_path):
    data = pd.read_csv(csv_path, usecols=["timestamp", "nsw_demand"])
    if data.empty:
        raise ValueError("输入 CSV 没有数据。")

    interval_end = pd.to_datetime(data["timestamp"], errors="raise")
    if interval_end.isna().any():
        raise ValueError("CSV 中存在空时间戳。")
    if not interval_end.eq(interval_end.dt.floor("30min")).all():
        raise ValueError("CSV 中存在未对齐到半小时的时间戳。")

    # 时间戳是区间结束时间。午夜 00:00 属于前一个市场日的最后一档。
    interval_start = interval_end - pd.Timedelta(minutes=30)
    data["market_day"] = interval_start.dt.normalize()
    data["slot"] = interval_start.dt.hour * 2 + interval_start.dt.minute // 30

    if data.duplicated(["market_day", "slot"]).any():
        raise ValueError("同一个市场日内存在重复档位。")

    daily = (
        data.pivot(index="market_day", columns="slot", values="nsw_demand")
        .sort_index()
        .reindex(columns=range(SLOTS_PER_DAY))
    )

    expected_days = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    missing_days = expected_days.difference(daily.index)
    if len(missing_days):
        raise ValueError(f"缺少完整市场日，首个缺失日为 {missing_days[0].date()}。")
    if not np.isfinite(daily.to_numpy(dtype=float)).all():
        raise ValueError("有市场日缺少需求档位，或需求值无效。")

    return daily


def select_k_by_year(train):
    rows = []

    for validation_year in TRAIN_YEARS:
        is_validation = train.index.year == validation_year
        fit_profiles = train.loc[~is_validation]
        validation_profiles = train.loc[is_validation]

        # 每折用另外三年拟合 PCA，折内所有 k 使用同一套 PCA。
        pca = PCA(n_components=PCA_VARIANCE, svd_solver="full")
        fit_scores = pca.fit_transform(fit_profiles)
        validation_scores = pca.transform(validation_profiles)

        for k in K_VALUES:
            model = KMeans(n_clusters=k, n_init=N_INIT, random_state=SEED)
            model.fit(fit_scores)
            labels = model.predict(validation_scores)

            counts = np.bincount(labels, minlength=k)
            occupied = np.count_nonzero(counts)
            score = np.nan
            if 2 <= occupied < len(labels):
                score = silhouette_score(validation_scores, labels)

            rows.append({
                "k": k,
                "validation_year": validation_year,
                "fit_days": len(fit_profiles),
                "validation_days": len(validation_profiles),
                "pca_components": pca.n_components_,
                "validation_silhouette": score,
                "occupied_clusters": occupied,
                "smallest_cluster_days": counts.min(),
            })

    folds = pd.DataFrame(rows)
    yearly_scores = folds.pivot(
        index="k",
        columns="validation_year",
        values="validation_silhouette",
    )
    yearly_scores.columns = [f"silhouette_{year}" for year in yearly_scores.columns]

    summary = yearly_scores.join(folds.groupby("k").agg(
        valid_years=("validation_silhouette", "count"),
        mean_silhouette=("validation_silhouette", "mean"),
        std_silhouette=("validation_silhouette", "std"),
        min_occupied_clusters=("occupied_clusters", "min"),
        min_cluster_days=("smallest_cluster_days", "min"),
    ))

    # 四轮都能评分的候选按平均分排名。同分时选较小的 k。
    eligible = summary.loc[summary["valid_years"].eq(len(TRAIN_YEARS))]
    ranked = eligible.reset_index().sort_values(
        ["mean_silhouette", "k"], ascending=[False, True]
    )
    if ranked.empty:
        raise ValueError("没有任何 k 能在四个留出年份中都计算轮廓系数。")

    best_k = int(ranked.iloc[0]["k"])
    summary["selected"] = summary.index == best_k
    return best_k, folds, summary


def save_profile_plot(train, train_labels, typical_profiles, path):
    hours = np.arange(SLOTS_PER_DAY) / 2
    n_types = len(typical_profiles)
    n_columns = min(n_types, 3)
    n_rows = (n_types + n_columns - 1) // n_columns

    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(5 * n_columns, 3.5 * n_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )

    for ax, (day_type, mean_curve) in zip(axes.flat, typical_profiles.iterrows()):
        members = train.loc[train_labels.eq(day_type)]
        color = f"C{(day_type - 1) % 10}"

        ax.plot(
            hours,
            members.to_numpy().T,
            color=color,
            alpha=0.035,
            linewidth=0.6,
        )
        ax.axhline(1, color="0.5", linestyle="--", linewidth=0.8)
        ax.plot(hours, mean_curve.to_numpy(), color=color, linewidth=2.8)
        ax.set(
            title=f"Type {day_type} · {len(members):,} training days",
            xlim=(0, 23.5),
            xticks=np.arange(0, 24, 3),
        )

    for ax in list(axes.flat)[n_types:]:
        ax.set_visible(False)

    fig.suptitle(f"NSW demand profiles by day type, 2010–2013 (k={n_types})")
    fig.supxlabel("Market time (interval start, AEST)")
    fig.supylabel("Demand / daily mean")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_yearly_shares_plot(yearly_shares, last_day, path):
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)

    for day_type, shares in yearly_shares.items():
        ax.plot(
            shares.index,
            shares * 100,
            marker="o",
            color=f"C{(day_type - 1) % 10}",
            label=f"Type {day_type}",
        )

    ax.set(
        title="Annual share of NSW demand day types",
        xlabel=f"Market year (data through {last_day:%Y-%m-%d})",
        ylabel="Share of observed days (%)",
        ylim=(0, 100),
        xticks=yearly_shares.index,
    )
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Select NSW demand day types using leave-one-year-out validation."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    daily = load_daily_profiles(args.data)
    daily_mean = daily.mean(axis=1)
    if (daily_mean <= 0).any():
        raise ValueError("日均需求必须大于 0，才能按日均值归一化。")

    normalized = daily.div(daily_mean, axis=0)
    train = normalized.loc[normalized.index.year.isin(TRAIN_YEARS)]
    expected_train_days = pd.date_range("2010-01-01", "2013-12-31", freq="D")
    if not train.index.equals(expected_train_days):
        raise ValueError("数据必须覆盖完整的 2010–2013 年训练期。")

    best_k, folds, summary = select_k_by_year(train)

    # 选定 k 后，用完整四年拟合最终 PCA 和 K-means。
    pca = PCA(n_components=PCA_VARIANCE, svd_solver="full")
    train_scores = pca.fit_transform(train)
    model = KMeans(n_clusters=best_k, n_init=N_INIT, random_state=SEED)
    model.fit(train_scores)

    # 用固定模型给所有可用日期分配日型，包括已有的 11 月和 12 月。
    all_scores = pca.transform(normalized)
    labels = pd.Series(
        model.predict(all_scores) + 1,
        index=daily.index,
        name="day_type",
    )

    assignments = labels.to_frame()
    assignments["daily_mean_mw"] = daily_mean
    assignments["distance_to_centroid"] = model.transform(all_scores).min(axis=1)

    train_labels = labels.loc[train.index]
    typical_profiles = train.groupby(train_labels).mean()

    years = pd.Index(labels.index.year, name="year")
    yearly_counts = pd.crosstab(years, labels.to_numpy(), colnames=["day_type"])
    yearly_shares = yearly_counts.div(yearly_counts.sum(axis=1), axis=0)

    pca_tag = f"pca{100 * PCA_VARIANCE:g}"
    search_suffix = f"loyo_2010_2013_k{K_VALUES.start}_to_k{K_VALUES.stop - 1}_{pca_tag}"
    model_suffix = f"loyo_k{best_k}_{pca_tag}"
    args.output.mkdir(parents=True, exist_ok=True)

    folds.to_csv(args.output / f"k_selection_folds_{search_suffix}.csv", index=False)
    summary.to_csv(args.output / f"k_selection_summary_{search_suffix}.csv")
    assignments.to_csv(args.output / f"day_type_assignments_{model_suffix}.csv")
    typical_profiles.to_csv(args.output / f"typical_day_profiles_{model_suffix}.csv")
    yearly_shares.to_csv(args.output / f"yearly_day_type_shares_{model_suffix}.csv")

    save_profile_plot(
        train,
        train_labels,
        typical_profiles,
        args.output / f"day_type_profile_overlays_{model_suffix}.png",
    )
    save_yearly_shares_plot(
        yearly_shares,
        daily.index.max(),
        args.output / f"yearly_day_type_shares_{model_suffix}.png",
    )

    print(summary.round(4).to_string())
    print(f"\nSelected k: {best_k}")
    print(f"Final PCA components: {pca.n_components_}")
    print(f"Final PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    print(f"Training days: {len(train):,}")
    print(f"Total labelled days: {len(assignments):,}")
    print(f"Outputs: {args.output}")


if __name__ == "__main__":
    main()