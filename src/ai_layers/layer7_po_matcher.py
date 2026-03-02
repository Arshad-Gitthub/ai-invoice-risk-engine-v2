import numpy as np

import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor

import joblib

CATEGORY_PRIOR_TOLERANCE = {"Raw Material":7.0,"Logistics":6.0,"Packaging":4.0,"Spare Parts":5.0,"General Expenses":2.0}

VENDOR_ENCODE = {"V001":1,"V002":2,"V003":3,"V004":4,"V005":5,"V999":0}

CATEGORY_ENCODE = {"Raw Material":1,"Logistics":2,"Packaging":3,"Spare Parts":4,"General Expenses":5}

class POToleranceLearner:

    def __init__(self):

        self.regressor = GradientBoostingRegressor(n_estimators=150, learning_rate=0.08,

                                                    max_depth=4, min_samples_leaf=5, random_state=42)

        self.is_fitted = False

        self.fallback = CATEGORY_PRIOR_TOLERANCE.copy()

    def _encode_features(self, df_or_dict):

        if isinstance(df_or_dict, dict):

            d = df_or_dict

            return np.array([[VENDOR_ENCODE.get(d.get("vendor_id",""),0),

                               CATEGORY_ENCODE.get(d.get("category",""),0),

                               float(d.get("po_amount",10000)),

                               float(d.get("payment_terms",30)),

                               float(d.get("vendor_risk_base",0.05))]])

        return np.column_stack([

            df_or_dict["vendor_id"].map(VENDOR_ENCODE).fillna(0),

            df_or_dict["category"].map(CATEGORY_ENCODE).fillna(0),

            df_or_dict["po_amount"].fillna(10000),

            df_or_dict["payment_terms"].fillna(30),

            df_or_dict["vendor_risk_base"].fillna(0.05),

        ])

    def fit(self, df):

        normal = df[df["is_anomaly"] == 0] if "is_anomaly" in df.columns else df

        needed = ["vendor_id","category","po_amount","payment_terms","vendor_risk_base","amount_variance_pct"]

        if not all(c in normal.columns for c in needed): return self

        X = self._encode_features(normal)

        y = normal["amount_variance_pct"].abs().clip(0, 50).values

        if len(X) >= 20:

            self.regressor.fit(X, y)

            self.is_fitted = True

        return self

    def score(self, invoice):

        actual_var = abs(float(invoice.get("amount_variance_pct", 0)))

        if self.is_fitted:

            X = self._encode_features(invoice)

            learned = float(max(0.5, self.regressor.predict(X)[0]))

        else:

            cat = invoice.get("category","General Expenses")

            learned = CATEGORY_PRIOR_TOLERANCE.get(cat, 5.0)

        excess = max(0.0, actual_var - learned)

        risk = float(np.clip(excess / max(learned * 3, 1.0), 0.0, 1.0))

        return {

            "risk_score": round(risk, 4),

            "actual_variance": round(actual_var, 2),

            "learned_tolerance": round(learned, 2),

            "excess_variance": round(excess, 2),

            "detail": f"Actual={actual_var:.1f}% vs learned tol={learned:.1f}%",

        }
 
