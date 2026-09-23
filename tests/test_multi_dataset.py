from yaeda import TabularEDA


def test_multi_dataset_comparison_dashboard(classification_df, tmp_path):
    train_df = classification_df
    # Create test dataframe with missingness drift and novel category
    test_df = classification_df.sample(120, random_state=42).copy()
    test_df = test_df.drop(columns=["target"])
    test_df.loc[:30, "feat_num1"] = None  # Higher missing rate
    test_df["feat_cat"] = test_df["feat_cat"].replace({"TypeA": "UnseenTypeX"})

    eda = TabularEDA(
        df=(train_df, "Train"),
        target="target",
        secondary_dfs=[(test_df, "Test")],
    )

    assert eda.has_secondary is True
    assert eda.has_target is True

    # 1. Multi dataset profiling check
    multi_stats = eda.multi_stats
    assert "Test" in multi_stats.comparisons
    test_comps = multi_stats.comparisons["Test"]

    cat_comp = next(c for c in test_comps if c.feature == "feat_cat")
    assert len(cat_comp.unseen_categories) > 0
    assert "UnseenTypeX" in cat_comp.unseen_categories

    # 2. HTML Generation check
    html_path = tmp_path / "multi_dashboard.html"
    html_content = eda.to_html(output=html_path)

    assert html_path.exists()
    assert "tab-comparison" in html_content
    assert "Train vs. Test" in html_content
    assert "UnseenTypeX" in html_content

    # 3. Markdown Generation check
    md_path = tmp_path / "multi_report.md"
    md_content = eda.to_markdown(output=md_path)

    assert md_path.exists()
    assert "Secondary Dataset Comparisons" in md_content
    assert "Train vs. `Test`" in md_content


def test_multi_dataset_without_target(classification_df, tmp_path):
    df1 = classification_df.drop(columns=["target"])
    df2 = df1.sample(80, random_state=42).copy()

    eda = TabularEDA(
        df=(df1, "Cohort_A"),
        target=None,
        secondary_dfs={"Cohort_B": df2},
    )

    assert eda.has_secondary is True
    assert eda.has_target is False

    html_path = tmp_path / "comparison_no_target.html"
    html_content = eda.to_html(output=html_path)

    assert html_path.exists()
    assert "tab-comparison" in html_content
    assert "tab-features" in html_content
    # Target-dependent tabs must be absent
    assert "tab-golden" not in html_content
    assert "tab-interactions" not in html_content
