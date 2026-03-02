import numpy as np

from difflib import SequenceMatcher

from datetime import datetime

import hashlib

def _str_sim(a, b):

    return SequenceMatcher(None, str(a).upper(), str(b).upper()).ratio()

def _num_sim(a, b):

    if max(abs(a), abs(b)) < 1e-9:

        return 1.0

    return max(0.0, 1.0 - abs(a - b) / max(abs(a), abs(b)))

class AIDuplicateDetector:

    WEIGHTS = np.array([0.30, 0.25, 0.20, 0.15, 0.07, 0.03])

    def __init__(self):

        self.threshold = 0.88

        self.registry = []

        self._exact_hashes = set()

    def _fingerprint(self, inv):

        k = f"{inv['invoice_number']}|{inv['invoice_amount']}|{inv.get('vendor_id','')}"

        return hashlib.sha256(k.encode()).hexdigest()

    def _similarity_vector(self, a, b):

        def norm(s): return "".join(c for c in str(s).upper() if c.isalnum())

        try:

            d1 = datetime.strptime(a.get("invoice_date","2000-01-01"), "%Y-%m-%d")

            d2 = datetime.strptime(b.get("invoice_date","2000-01-01"), "%Y-%m-%d")

            date_sim = max(0.0, 1.0 - abs((d1-d2).days) / 30.0)

        except:

            date_sim = 0.5

        return np.array([

            _str_sim(norm(a["invoice_number"]), norm(b["invoice_number"])),

            _num_sim(a["invoice_amount"], b["invoice_amount"]),

            _str_sim(a.get("po_number",""), b.get("po_number","")),

            1.0 if a.get("vendor_id") == b.get("vendor_id") else 0.0,

            date_sim,

            _num_sim(a.get("quantity",0), b.get("quantity",0)),

        ])

    def _composite_score(self, vec):

        return float(np.dot(vec, self.WEIGHTS))

    def fit(self, invoices):

        if len(invoices) < 10:

            return self

        scores = []

        sample = invoices[:min(100, len(invoices))]

        for i in range(len(sample)):

            for j in range(i+1, min(i+10, len(sample))):

                vec = self._similarity_vector(sample[i], sample[j])

                scores.append(self._composite_score(vec))

        if scores:

            self.threshold = min(0.95, float(np.percentile(scores, 99)) + 0.01)

        return self

    def check(self, new_inv):

        fp = self._fingerprint(new_inv)

        if fp in self._exact_hashes:

            return {"is_duplicate": True, "match_type": "EXACT",

                    "match_score": 1.0, "risk_score": 1.0,

                    "detail": "Identical invoice_number + amount + vendor"}

        best_score, best_inv, best_vec = 0.0, None, None

        for stored in self.registry[-500:]:

            vec = self._similarity_vector(new_inv, stored)

            score = self._composite_score(vec)

            if score > best_score:

                best_score, best_inv, best_vec = score, stored, vec

        if best_score >= self.threshold:

            dims = ["inv_num","amount","po_num","vendor","date","qty"]

            top_dim = dims[int(np.argmax(best_vec * self.WEIGHTS))]

            return {"is_duplicate": True, "match_type": "AI_SIMILARITY",

                    "match_score": round(best_score, 4), "risk_score": round(best_score, 4),

                    "detail": f"AI similarity={best_score:.3f} (threshold={self.threshold:.3f}). Strongest: {top_dim}"}

        self._exact_hashes.add(fp)

        self.registry.append(new_inv)

        return {"is_duplicate": False, "match_type": "NONE",

                "match_score": round(best_score, 4), "risk_score": 0.0,

                "detail": f"No match (best score={best_score:.3f} < threshold={self.threshold:.3f})"}
 
