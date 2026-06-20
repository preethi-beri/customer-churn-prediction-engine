import joblib
import json
import pandas as pd

model = joblib.load(
    "models/churn_model.pkl"
)

with open(
    "models/feature_names.json",
    "r"
) as f:
    feature_names = json.load(f)


def get_top_risk_factors():

    try:

        coefficients = abs(
            model.coef_[0]
        )

        importance_df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": coefficients
            }
        )

        top_features = (
            importance_df
            .sort_values(
                "importance",
                ascending=False
            )
            .head(5)
        )

        return list(
            top_features["feature"]
        )

    except Exception as e:

        print("Error:", e)

        return []