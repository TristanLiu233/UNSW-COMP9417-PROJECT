"""Select day types using chronological forward validation."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


HERE = Path(__file__).resolve().parent
DEFAULT_DATA = (
    HERE.parent / "data" / "nem_weather_halfhourly_2010_2026.csv"
)
DEFAULT_OUTPUT = HERE / "outputs" / "day_type_clustering"

SLOTS_PER_DAY = 48
TRAIN_YEARS = (2010, 2011, 2012, 2013)
VALIDATION_YEARS = (2011, 2012, 2013)
K_VALUES = range(2, 11)
PCA_VARIANCE = 0.95
SEED = 42
N_INIT = 20


def load_daily_profiles(csv_path):
    data = pd.read_csv(
        csv_path, usecols=["timestamp", "nsw_demand"]
    )
    if data.empty:
        raise ValueError("输入CSV没有数据。")

    ends = pd.to_datetime(data["timestamp"], errors="raise")
    if ends.isna().any():
        raise ValueError("存在空时间戳。")
    if not ends.eq(ends.dt.floor("30min")).all():
        raise ValueError("存在未对齐到半小时的时间戳。")

    # 午夜00:00属于前一个市场日最后一档。
    starts = ends - pd.Timedelta(minutes=30)
    data["market_day"] = starts.dt.normalize()
    data["slot"] = starts.dt.hour * 2 + starts.dt.minute // 30
    data["nsw_demand"] = pd.to_numeric(
        data["nsw_demand"], errors="raise"
    )

    if data.duplicated(["market_day", "slot"]).any():
        raise ValueError("同一市场日存在重复档位。")

    daily = (
        data.pivot(
            index="market_day",
            columns="slot",
            values="nsw_demand",
        )
        .sort_index()
        .reindex(columns=range(SLOTS_PER_DAY))
    )

    expected = pd.date_range(
        daily.index.min(), daily.index.max(), freq="D"
    )
    if not daily.index.equals(expected):
        raise ValueError("数据存在缺失市场日。")
    if not np.isfinite(daily.to_numpy(dtype=float)).all():
        raise ValueError("存在缺失档位或无效需求。")
    if (daily.to_numpy() <= 0).any():
        raise ValueError("需求应为正数。")

    return daily


def select_k_forward(train):
    rows = []

    for validation_year in VALIDATION_YEARS:
        # 只用验证年份之前的数据拟合。
        fit_profiles = train.loc[
            train.index.year < validation_year
        ]
        validation_profiles = train.loc[
            train.index.year == validation_year
        ]

        if fit_profiles.empty or validation_profiles.empty:
            raise ValueError("前向验证缺少训练或验证数据。")
        if fit_profiles.index.max() >= validation_profiles.index.min():
            raise ValueError("训练日期必须早于验证日期。")

        # PCA也必须只在本折训练数据上拟合。
        pca = PCA(
            n_components=PCA_VARIANCE,
            svd_solver="full",
        )
        fit_scores = pca.fit_transform(fit_profiles)
        validation_scores = pca.transform(validation_profiles)

        for k in K_VALUES:
            model = KMeans(
                n_clusters=k,
                n_init=N_INIT,
                random_state=SEED,
            )
            model.fit(fit_scores)
            labels = model.predict(validation_scores)

            counts = np.bincount(labels, minlength=k)
            occupied = int(np.count_nonzero(counts))

            score = np.nan
            if 2 <= occupied < len(labels):
                score = silhouette_score(
                    validation_scores, labels
                )

            rows.append({
                "k": k,
                "fit_first_day": fit_profiles.index.min(),
                "fit_last_day": fit_profiles.index.max(),
                "validation_year": validation_year,
                "fit_days": len(fit_profiles),
                "validation_days": len(validation_profiles),
                "pca_components": pca.n_components_,
                "pca_variance_explained": (
                    pca.explained_variance_ratio_.sum()
                ),
                "validation_silhouette": score,
                "occupied_clusters": occupied,
                "smallest_cluster_days": int(counts.min()),
            })

    folds = pd.DataFrame(rows)

    yearly_scores = folds.pivot(
        index="k",
        columns="validation_year",
        values="validation_silhouette",
    )
    yearly_scores.columns = [
        f"silhouette_{year}" for year in yearly_scores.columns
    ]

    summary = yearly_scores.join(
        folds.groupby("k").agg(
            valid_years=("validation_silhouette", "count"),
            mean_silhouette=("validation_silhouette", "mean"),
            std_silhouette=("validation_silhouette", "std"),
            min_occupied_clusters=("occupied_clusters", "min"),
            min_cluster_days=("smallest_cluster_days", "min"),
        )
    )

    # 三折都可评分；按平均轮廓系数选k，同分选较小k。
    eligible = summary.loc[
        summary["valid_years"].eq(len(VALIDATION_YEARS))
    ]
    ranked = eligible.reset_index().sort_values(
        ["mean_silhouette", "k"],
        ascending=[False, True],
    )
    if ranked.empty:
        raise ValueError("没有候选k能在全部前向验证折中评分。")

    best_k = int(ranked.iloc[0]["k"])
    summary["selected"] = summary.index == best_k
    return best_k, folds, summary


def save_profile_plot(train, train_labels, typical, path):
    hours = np.arange(SLOTS_PER_DAY) / 2
    n_types = len(typical)
    n_columns = min(n_types, 3)
    n_rows = (n_types + n_columns - 1) // n_columns

    fig, axes = plt.subplots(
        n_rows, n_columns,
        figsize=(5 * n_columns, 3.5 * n_rows),
        sharex=True, sharey=True,
        squeeze=False,
        constrained_layout=True,
    )

    for ax, (day_type, mean_curve) in zip(
        axes.flat, typical.iterrows()
    ):
        members = train.loc[train_labels.eq(day_type)]
        color = f"C{(day_type - 1) % 10}"

        ax.plot(
            hours, members.to_numpy().T,
            color=color, alpha=0.035, linewidth=0.6,
        )
        ax.axhline(
            1, color="0.5", linestyle="--", linewidth=0.8
        )
        ax.plot(
            hours, mean_curve.to_numpy(),
            color=color, linewidth=2.8,
        )
        ax.set(
            title=f"Type {day_type}: {len(members):,} training days",
            xlim=(0, 23.5),
            xticks=np.arange(0, 24, 3),
        )

    for ax in list(axes.flat)[n_types:]:
        ax.set_visible(False)

    fig.suptitle(
        f"NSW training demand profiles, 2010-2013 (k={n_types})"
    )
    fig.supxlabel("Market time (interval start, AEST)")
    fig.supylabel("Demand / daily mean")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_yearly_shares_plot(shares, last_day, path):
    fig, ax = plt.subplots(
        figsize=(10, 5), constrained_layout=True
    )

    for day_type in shares.columns:
        ax.plot(
            shares.index, shares[day_type] * 100,
            marker="o", label=f"Type {day_type}",
            color=f"C{(day_type - 1) % 10}",
        )

    ax.set(
        title="Annual shares under fixed training-period clusters",
        xlabel=f"Market year (data through {last_day:%Y-%m-%d})",
        ylabel="Share of observed days (%)",
        ylim=(0, 100),
        xticks=shares.index,
    )
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=DEFAULT_DATA
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT
    )
    args = parser.parse_args()

    daily = load_daily_profiles(args.data)
    daily_mean = daily.mean(axis=1)
    normalized = daily.div(daily_mean, axis=0)

    train = normalized.loc[
        normalized.index.year.isin(TRAIN_YEARS)
    ]
    expected_train = pd.date_range(
        "2010-01-01", "2013-12-31", freq="D"
    )
    if not train.index.equals(expected_train):
        raise ValueError("训练期必须完整覆盖2010-2013年。")

    best_k, folds, summary = select_k_forward(train)

    # 最终模型仍只在完整2010-2013年拟合一次。
    pca = PCA(
        n_components=PCA_VARIANCE, svd_solver="full"
    )
    train_scores = pca.fit_transform(train)
    model = KMeans(
        n_clusters=best_k,
        n_init=N_INIT,
        random_state=SEED,
    )
    model.fit(train_scores)

    # 后续年份只转换和分配类别，不重新拟合。
    all_scores = pca.transform(normalized)
    labels = pd.Series(
        model.predict(all_scores) + 1,
        index=daily.index,
        name="day_type",
    )

    assignments = labels.to_frame()
    assignments["daily_mean_mw"] = daily_mean
    assignments["distance_to_centroid"] = (
        model.transform(all_scores).min(axis=1)
    )

    train_labels = labels.loc[train.index]
    typical = train.groupby(train_labels).mean()

    years = pd.Index(labels.index.year, name="year")
    counts = pd.crosstab(
        years, labels.to_numpy(), colnames=["day_type"]
    ).reindex(columns=range(1, best_k + 1), fill_value=0)
    shares = counts.div(counts.sum(axis=1), axis=0)

    pca_tag = f"pca{100 * PCA_VARIANCE:g}"
    search_tag = (
        f"forward_2010_2013_k{K_VALUES.start}"
        f"_to_k{K_VALUES.stop - 1}_{pca_tag}"
    )
    model_tag = f"forward_k{best_k}_{pca_tag}"

    args.output.mkdir(parents=True, exist_ok=True)

    folds.to_csv(
        args.output / f"k_selection_folds_{search_tag}.csv",
        index=False,
    )
    summary.to_csv(
        args.output / f"k_selection_summary_{search_tag}.csv"
    )
    assignments.to_csv(
        args.output / f"day_type_assignments_{model_tag}.csv"
    )
    typical.to_csv(
        args.output / f"typical_day_profiles_{model_tag}.csv"
    )
    shares.to_csv(
        args.output / f"yearly_day_type_shares_{model_tag}.csv"
    )

    save_profile_plot(
        train, train_labels, typical,
        args.output / f"day_type_profile_overlays_{model_tag}.png",
    )
    save_yearly_shares_plot(
        shares, daily.index.max(),
        args.output / f"yearly_day_type_shares_{model_tag}.png",
    )

    print("Forward validation:")
    print("2010 -> 2011")
    print("2010-2011 -> 2012")
    print("2010-2012 -> 2013")
    print(summary.round(4).to_string())
    print(f"\nSelected k: {best_k}")
    print(f"Final PCA components: {pca.n_components_}")
    print(
        "Final PCA explained variance: "
        f"{pca.explained_variance_ratio_.sum():.2%}"
    )
    print(f"Training days: {len(train):,}")
    print(f"Total labelled days: {len(assignments):,}")
    print(f"Outputs: {args.output}")

    # 若k仍为2，可检查与旧版标签和距离是否一致。
    old_path = (
        args.output
        / f"day_type_assignments_loyo_k{best_k}_{pca_tag}.csv"
    )
    if old_path.exists():
        old = pd.read_csv(
            old_path, index_col="market_day", parse_dates=True
        )
        if old.index.equals(assignments.index):
            same_labels = np.array_equal(
                old["day_type"], assignments["day_type"]
            )
            same_distances = np.allclose(
                old["distance_to_centroid"],
                assignments["distance_to_centroid"],
                rtol=0, atol=1e-10,
            )
            print(f"Labels identical to old result: {same_labels}")
            print(
                "Distances identical to old result: "
                f"{same_distances}"
            )


if __name__ == "__main__":
    main()