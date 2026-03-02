import numpy as np

import pandas as pd

from scipy.spatial.distance import mahalanobis

from scipy.linalg import pinv

VENDOR_FEATURES = ["invoice_amount","quantity","unit_price","amount_variance_pct","days_to_due"]

class VendorBehaviorProfiler:

    def __init__(self):

        self.profiles = {}

        self.global_profile = {}

        self.is_fitted = False

    def _build_profile(self, data):

        if len(data) < 3: return None

        mean = np.mean(data, axis=0)

        cov = np.cov(data.T)

        cov_inv = pinv(cov + np.eye(len(mean)) * 1e-6)

        std = np.std(data, axis=0) + 1e-9

        return {"mean": mean, "cov_inv": cov_inv, "std": std, "n": len(data),

                "min": np.min(data, axis=0), "max": np.max(data, axis=0)}

    def fit(self, df):

        normal = df[df["is_anomaly"] == 0] if "is_anomaly" in df.columns else df

        feats = [c for c in VENDOR_FEATURES if c in normal.columns]

        X_all = normal[feats].fillna(0).values

        self.global_profile = self._build_profile(X_all)

        self.global_profile["features"] = feats

        for v_id, grp in normal.groupby("vendor_id"):

            X_v = grp[feats].fillna(0).values

            if len(X_v) >= 3:

                profile = self._build_profile(X_v)

                if profile:

                    total = len(df[df["vendor_id"] == v_id])

                    anoms = len(df[(df["vendor_id"] == v_id) & (df.get("is_anomaly", pd.Series(0)) == 1)])

                    profile["trust"] = max(0.0, 1.0 - (anoms / max(total, 1)) * 4)

                    profile["features"] = feats

                    self.profiles[str(v_id)] = profile

        self.is_fitted = True

        return self

    def _extract_vector(self, invoice, features):

        return np.array([float(invoice.get(f, 0)) for f in features])

    def score(self, invoice):

        v_id = str(invoice.get("vendor_id", "UNKNOWN"))

        profile = self.profiles.get(v_id, self.global_profile)

        if not profile:

            return {"risk_score": 0.5, "mahal_dist": None, "vendor_trust": 0.5, "detail": "No profile"}

        feats = profile["features"]

        x = self._extract_vector(invoice, feats)

        try:

            dist = mahalanobis(x, profile["mean"], profile["cov_inv"])

        except Exception:

            dist = float(np.mean(np.abs(x - profile["mean"]) / profile["std"]))

        risk = float(np.clip(dist / 4.0, 0.0, 1.0))

        trust = profile.get("trust", 1.0)

        adjusted_risk = float(np.clip(risk * (2.0 - trust), 0.0, 1.0))

        return {

            "risk_score": round(adjusted_risk, 4),

            "raw_risk": round(risk, 4),

            "mahal_dist": round(dist, 4),

            "vendor_trust": round(trust, 4),

            "vendor_n": profile["n"],

            "detail": f"Mahalanobis distance={dist:.2f} (n={profile['n']}, trust={trust:.2f})"

        }

    def update_online(self, invoice, is_clean):

        if not is_clean: return

        v_id = str(invoice.get("vendor_id","UNKNOWN"))

        if v_id not in self.profiles: return

        p = self.profiles[v_id]

        x = self._extract_vector(invoice, p["features"])

        n = p["n"]

        p["mean"] = (p["mean"] * n + x) / (n + 1)

        p["n"] = n + 1
 
