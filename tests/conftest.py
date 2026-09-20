import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


@pytest.fixture
def classification_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 250
    x1 = np.random.normal(0, 1, n)
    x2 = x1 * 0.92 + np.random.normal(0, 0.15, n)
    x3 = np.random.exponential(1.5, n)
    cat_feature = np.random.choice(["TypeA", "TypeB", "TypeC"], size=n, p=[0.5, 0.3, 0.2])

    logits = 1.4 * x1 - 1.1 * x3 + (cat_feature == "TypeB") * 1.5
    prob = 1 / (1 + np.exp(-logits))
    target = (prob > 0.5).astype(int)

    df = pd.DataFrame(
        {
            "feat_num1": x1,
            "feat_collinear": x2,
            "feat_skewed": x3,
            "feat_cat": cat_feature,
            "target": target,
        }
    )

    df.loc[:10, "feat_num1"] = np.nan
    df.loc[15, "feat_skewed"] = 45.0
    df.loc[20:35, "feat_num1"] = 0.0
    return df


@pytest.fixture
def regression_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 200
    x1 = np.random.uniform(10, 100, n)
    x2 = np.random.normal(5, 2, n)
    cat = np.random.choice(["Small", "Medium", "Large"], size=n)
    y = 3.0 * x1 - 2.5 * x2 + np.random.normal(0, 2, n)

    df = pd.DataFrame(
        {
            "feat_num1": x1,
            "feat_num2": x2,
            "feat_cat": cat,
            "target_price": y,
        }
    )
    df.loc[:5, "feat_num2"] = np.nan
    return df


@pytest.fixture
def fitted_binary_model(classification_df) -> tuple[RandomForestClassifier, np.ndarray, np.ndarray]:
    # Impute missing values so prediction array length matches the full 250 rows of classification_df
    X = classification_df[["feat_num1", "feat_collinear", "feat_skewed"]].copy()
    X["feat_num1"] = X["feat_num1"].fillna(X["feat_num1"].median())
    y = classification_df["target"]

    model = RandomForestClassifier(n_estimators=10, max_depth=4, random_state=42)
    model.fit(X, y)
    y_pred = model.predict(X)  # Exactly 250 elements
    y_prob = model.predict_proba(X)[:, 1]  # Exactly 250 elements
    return model, y_pred, y_prob


@pytest.fixture
def fitted_regression_model(regression_df) -> tuple[RandomForestRegressor, np.ndarray]:
    # Impute missing values so prediction array length matches the full 200 rows of regression_df
    X = regression_df[["feat_num1", "feat_num2"]].copy()
    X["feat_num2"] = X["feat_num2"].fillna(X["feat_num2"].median())
    y = regression_df["target_price"]

    model = RandomForestRegressor(n_estimators=10, max_depth=4, random_state=42)
    model.fit(X, y)
    y_pred = model.predict(X)  # Exactly 200 elements
    return model, y_pred
