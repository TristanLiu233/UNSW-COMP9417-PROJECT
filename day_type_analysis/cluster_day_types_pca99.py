import cluster_day_types as experiment


if __name__ == "__main__":
    experiment.PCA_VARIANCE = 0.99
    experiment.DEFAULT_OUTPUT = experiment.HERE / "outputs_loyo_pca99"
    experiment.main()