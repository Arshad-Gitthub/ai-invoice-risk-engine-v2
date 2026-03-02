import numpy as np

from sklearn.ensemble import RandomForestClassifier

from sklearn.calibration import CalibratedClassifierCV

import joblib

THRESHOLDS = {"AUTO_POST": 0.28, "HOLD": 0.58}

class MetaLearner:

    def __init__(self):

        base_rf = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=3,

                                          class_weight={0:1,1:10}, random_state=42)

        self.clf = CalibratedClassifierCV(base_rf, cv=3, method="isotonic")

        self.is_fitted = False

    def _build_meta_features(self, signals):

        s1 = signals.get("nlp_risk",0); s2 = signals.get("dedup_risk",0)

        s3 = signals.get("iso_forest_risk",0); s4 = signals.get("dbscan_risk",0)

        s5 = signals.get("mlp_risk",0); s6 = signals.get("vendor_profile_risk",0)

        s7 = signals.get("po_tolerance_risk",0)

        all_scores = [s1,s2,s3,s4,s5,s6,s7]

        ens_avg = np.mean([s3,s4,s5])

        return np.array([[s1,s2,s3,s4,s5,s6,s7, s1*ens_avg, s6*s7,

                           float(np.max(all_scores)),

                           float(sum(1 for s in all_scores if s > 0.5))]])

    def fit(self, signal_rows, labels):

        X = np.vstack([self._build_meta_features(r) for r in signal_rows])

        y = np.array(labels)

        if len(X) >= 20 and len(set(y)) > 1:

            self.clf.fit(X, y)

            self.is_fitted = True

        return self

    def predict(self, signals):

        X = self._build_meta_features(signals)

        if self.is_fitted:

            proba = self.clf.predict_proba(X)[0]

            risk_score = float(proba[1])

        else:

            weights = [0.10,0.15,0.18,0.12,0.15,0.18,0.12]

            signal_vals = [signals.get(k,0) for k in

                           ["nlp_risk","dedup_risk","iso_forest_risk","dbscan_risk",

                            "mlp_risk","vendor_profile_risk","po_tolerance_risk"]]

            risk_score = float(np.dot(weights, signal_vals))

        risk_score = round(risk_score, 4)

        if risk_score < THRESHOLDS["AUTO_POST"]:

            decision = "AUTO_POST"

        elif risk_score < THRESHOLDS["HOLD"]:

            decision = "REVIEW"

        else:

            decision = "HOLD"

        signal_items = [("NLP Email", signals.get("nlp_risk",0)),

                        ("Ensemble Anomaly", max(signals.get("iso_forest_risk",0), signals.get("mlp_risk",0))),

                        ("Vendor Profile", signals.get("vendor_profile_risk",0)),

                        ("PO Tolerance", signals.get("po_tolerance_risk",0))]

        top_drivers = sorted(signal_items, key=lambda x: x[1], reverse=True)[:2]

        drivers_str = ", ".join(f"{n}={v:.2f}" for n,v in top_drivers if v > 0.1)

        return {

            "final_risk_score": risk_score,

            "risk_pct": f"{risk_score*100:.1f}%",

            "decision": decision,

            "confidence": "HIGH" if abs(risk_score - 0.43) > 0.15 else "MEDIUM",

            "top_drivers": drivers_str or "All signals low",

            "meta_model": "RandomForest+Calibrated" if self.is_fitted else "WeightedFallback",

        }
 
