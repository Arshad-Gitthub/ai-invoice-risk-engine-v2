import numpy as np

import pandas as pd

from sklearn.ensemble import IsolationForest

from sklearn.cluster import DBSCAN

from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler

import joblib, os

FEATURE_COLS = ["invoice_amount","po_amount","amount_variance_pct","quantity",

                "unit_price","line_vs_invoice_pct","days_to_due","is_month_end",

                "is_friday","vendor_risk_base","processing_hour"]

class EnsembleAnomalyDetector:

    ENSEMBLE_WEIGHTS = {"isolation_forest": 0.40, "dbscan": 0.25, "mlp_nn": 0.35}

    def __init__(self):

        self.scaler = StandardScaler()

        self.iso_forest = IsolationForest(n_estimators=300, contamination=0.08, max_samples="auto", random_state=42)

        self.dbscan = DBSCAN(eps=1.5, min_samples=5, metric="euclidean")

        self.mlp = MLPClassifier(hidden_layer_sizes=(64,32,16), activation="relu", solver="adam",

                                  learning_rate_init=0.001, max_iter=500, random_state=42)

        self.dbscan_labels_ = None

        self.dbscan_X_train_ = None

        self.is_fitted = False

    def _extract(self, df):

        if isinstance(df, pd.DataFrame):

            missing = [c for c in FEATURE_COLS if c not in df.columns]

            for c in missing: df[c] = 0

            return df[FEATURE_COLS].fillna(0).values

        return df

    def fit(self, df):

        X_raw = self._extract(df)

        X = self.scaler.fit_transform(X_raw)

        self.iso_forest.fit(X)

        normal_mask = df["is_anomaly"].values == 0 if "is_anomaly" in df.columns else np.ones(len(X), dtype=bool)

        X_normal = X[normal_mask]

        self.dbscan.fit(X_normal)

        self.dbscan_X_train_ = X_normal

        self.dbscan_labels_ = self.dbscan.labels_

        if "is_anomaly" in df.columns and df["is_anomaly"].sum() >= 5:

            self.mlp.fit(X, df["is_anomaly"].values)

        else:

            iso_labels = (self.iso_forest.predict(X) == -1).astype(int)

            self.mlp.fit(X, iso_labels)

        self.is_fitted = True

        return self

    def _iso_risk(self, x):

        score = float(self.iso_forest.decision_function(x.reshape(1,-1))[0])

        return float(np.clip(0.5 - score, 0.0, 1.0))

    def _dbscan_risk(self, x):

        if self.dbscan_X_train_ is None: return 0.5

        dists = np.linalg.norm(self.dbscan_X_train_ - x, axis=1)

        avg_dist = float(np.mean(np.sort(dists)[:5]))

        return float(np.clip(avg_dist / 3.0, 0.0, 1.0))

    def _mlp_risk(self, x):

        proba = self.mlp.predict_proba(x.reshape(1,-1))[0]

        return float(proba[1]) if len(proba) > 1 else 0.0

    def score(self, invoice):

        row = pd.DataFrame([invoice])

        X_raw = self._extract(row)

        X = self.scaler.transform(X_raw)[0]

        iso_risk = self._iso_risk(X)

        dbscan_risk = self._dbscan_risk(X)

        mlp_risk = self._mlp_risk(X)

        w = self.ENSEMBLE_WEIGHTS

        ensemble = round(iso_risk*w["isolation_forest"] + dbscan_risk*w["dbscan"] + mlp_risk*w["mlp_nn"], 4)

        votes = {"IsolationForest": iso_risk, "DBSCAN": dbscan_risk, "MLP_NN": mlp_risk}

        return {

            "ensemble_risk": ensemble,

            "model_votes": {"isolation_forest": round(iso_risk,4),

                            "dbscan": round(dbscan_risk,4),

                            "mlp_nn": round(mlp_risk,4)},

            "dominant_model": max(votes, key=votes.get),

        }
 
