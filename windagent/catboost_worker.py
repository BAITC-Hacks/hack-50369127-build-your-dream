"""Isolated native-CatBoost inference. Never deserializes Python/joblib models.

Input/output are JSON over stdin/stdout, suitable for a separate Python 3.12 venv.
Only this worker needs CatBoost; the web server keeps its stdlib-only runtime.
"""
import json
import sys


def main():
    from catboost import CatBoostRegressor, Pool

    request = json.load(sys.stdin)
    model = CatBoostRegressor()
    model.load_model(request["model_path"], format="cbm")
    names = list(model.feature_names_)
    if names != request["feature_names"]:
        raise ValueError("Native model feature names/order do not match the audited feature contract")
    data = Pool(request["features"], feature_names=names, cat_features=[names.index("turbine_id")])
    prediction = model.predict(data, thread_count=4)
    json.dump({"predictions": prediction.tolist(), "tree_count": model.tree_count_,
               "feature_names": names}, sys.stdout, allow_nan=False)


if __name__ == "__main__":
    main()
