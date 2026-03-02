import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.pipeline import Pipeline

import joblib

class NLPEmailClassifier:

    LABEL_NAMES = {0: "NORMAL", 1: "URGENT", 2: "SUSPICIOUS"}

    def __init__(self):

        self.pipeline = Pipeline([

            ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=500, sublinear_tf=True, stop_words="english")),

            ("clf", GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42))

        ])

        self.classes_ = []

        self.is_fitted = False

    def fit(self, texts, labels):

        self.pipeline.fit(texts, labels)

        self.classes_ = sorted(set(labels))

        self.is_fitted = True

        return self

    def predict(self, text):

        proba = self.pipeline.predict_proba([text])[0]

        risk_map = {0: 0.0, 1: 0.4, 2: 1.0}

        label_idx = int(np.argmax(proba))

        actual_label = int(self.classes_[label_idx]) if self.classes_ else label_idx

        conf = float(proba[label_idx])

        risk_score = sum(float(proba[i]) * risk_map.get(int(cls), 0.5) for i, cls in enumerate(self.classes_))

        return {

            "label": actual_label,

            "label_name": self.LABEL_NAMES.get(actual_label, "NORMAL"),

            "confidence": round(conf, 4),

            "risk_score": round(risk_score, 4),

        }

    def save(self, path):

        joblib.dump({"pipeline": self.pipeline, "classes": self.classes_}, path)

    def load(self, path):

        d = joblib.load(path)

        self.pipeline = d["pipeline"]

        self.classes_ = d["classes"]

        self.is_fitted = True

        return self
 
