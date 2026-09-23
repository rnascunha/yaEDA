from yaeda.charts import EDAChartGenerator


def test_multi_dataset_distribution_and_drift_charts(classification_df):
    cg = EDAChartGenerator()
    train_df = classification_df
    test_df = classification_df.sample(100, random_state=42).copy()
    test_df["feat_cat"] = "NewCategory"  # Inject unseen category

    datasets = {"Train": train_df, "Test": test_df}

    # 1. Test numeric comparison
    b64_num = cg.plot_feature_distribution_comparison(datasets, "feat_num1", is_numeric=True)
    assert len(b64_num) > 100

    # 2. Test categorical comparison with unseen category
    b64_cat = cg.plot_feature_distribution_comparison(datasets, "feat_cat", is_numeric=False)
    assert len(b64_cat) > 100

    # 3. Test summary card with secondary dataset and target
    b64_card = cg.plot_feature_summary_card(
        df=train_df,
        feature="feat_num1",
        target="target",
        secondary_dfs={"Test": test_df},
    )
    assert len(b64_card) > 100

    # 4. Test summary card without target
    b64_card_notarget = cg.plot_feature_summary_card(
        df=train_df,
        feature="feat_num1",
        target=None,
        secondary_dfs={"Test": test_df},
    )
    assert len(b64_card_notarget) > 100
