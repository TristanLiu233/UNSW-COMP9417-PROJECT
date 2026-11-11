"""The four estimators used by both the controlled comparisons and tuning."""

from dataclasses import dataclass

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ModelSpec:
    task: str
    default: float
    grid: tuple


SPECS = {
    "ridge": ModelSpec("demand", 1, (1, 100, 10000)),
    "boost_demand": ModelSpec("demand", 31, (7, 15, 31)),
    "logistic": ModelSpec("price", 1, (0.01, 0.1, 1)),
    "boost_price": ModelSpec("price", 31, (7, 15, 31)),
}
TARGET = {"demand": "nsw_demand_mw", "price": "nsw_price_up"}
PRIMARY = {"demand": "MAPE", "price": "BalAcc"}


def make_estimator(name, param):
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=param))
    if name == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(C=param, max_iter=500, random_state=17))
    estimator = {"boost_demand": HistGradientBoostingRegressor,
                 "boost_price": HistGradientBoostingClassifier}[name]
    loss = "squared_error" if SPECS[name].task == "demand" else "log_loss"
    return estimator(loss=loss, max_leaf_nodes=int(param), max_iter=100,
                     learning_rate=0.1, early_stopping=False, random_state=17)


def predict_raw(name, param, X_train, y_train, X_valid):
    model = make_estimator(name, param).fit(X_train, y_train)
    if SPECS[name].task == "price":
        return model.predict_proba(X_valid)[:, 1]
    return model.predict(X_valid)
