import sys

import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import pandas as pd

from data.abc_dataset import generate_dataset, LIVE_INVOICES

from src.ai_layers.layer1_nlp_parser import NLPEmailClassifier

from src.ai_layers.layer2_ai_dedup import AIDuplicateDetector

from src.ai_layers.layer3_ensemble_anomaly import EnsembleAnomalyDetector

from src.ai_layers.layer6_vendor_profiler import VendorBehaviorProfiler

from src.ai_layers.layer7_po_matcher import POToleranceLearner

from src.ai_layers.layer8_meta_learner import MetaLearner

from src.ai_layers.layer9_continuous_learning import ContinuousLearner

def bar(score, width=20):

    n = int(float(score) * width)

    return "#" * n + "." * (width - n)

def header(title):

    print("\n" + "=" * 68)

    print(f"  {title}")

    print("=" * 68)

def run_pipeline():

    header("ABC COMPANY - 90% AI Invoice Processing System")

    print("\n[*] Generating ABC Company invoice history (500 invoices)...")

    df = generate_dataset(n=500)

    print(f"    Total: {len(df)} | Normal: {(df.is_anomaly==0).sum()} | Anomalies: {(df.is_anomaly==1).sum()}")

    print(f"    Anomaly types: {df[df.is_anomaly==1]['anomaly_type'].value_counts().to_dict()}")

    header("TRAINING ALL 9 AI LAYERS")

    print("\n  Layer 1 - NLP Email Classifier...")

    nlp = NLPEmailClassifier()

    nlp.fit(df["email_text"].tolist(), df["email_urgency"].tolist())

    print(f"  [OK] Trained on {len(df)} email texts")

    print("\n  Layer 2 - AI Duplicate Detector...")

    dedup = AIDuplicateDetector()

    dedup.fit(df.to_dict("records"))

    print(f"  [OK] Learned similarity threshold: {dedup.threshold:.3f}")

    print("\n  Layers 3+4+5 - Ensemble Anomaly Detector...")

    ensemble = EnsembleAnomalyDetector()

    ensemble.fit(df)

    print(f"  [OK] IsolationForest: {ensemble.iso_forest.n_estimators} trees")

    print("\n  Layer 6 - Vendor Behavioral Profiler...")

    vendor_profiler = VendorBehaviorProfiler()

    vendor_profiler.fit(df)

    print(f"  [OK] Vendor profiles built:")

    for v, p in vendor_profiler.profiles.items():

        print(f"       {v}: n={p['n']} | trust={p['trust']:.2f} | mean_amt={p['mean'][0]:,.0f} AED")

    print("\n  Layer 7 - PO Tolerance Learner...")

    po_learner = POToleranceLearner()

    po_learner.fit(df)

    print(f"  [OK] Trained on {len(df[df.is_anomaly==0])} normal invoices")

    print("\n  Layer 8 - Meta-Learner...")

    meta = MetaLearner()

    meta_signals, meta_labels = [], []

    for _, row in df.iterrows():

        inv = row.to_dict()

        nlp_r = nlp.predict(inv.get("email_text", ""))

        ens_r = ensemble.score(inv)

        vend_r = vendor_profiler.score(inv)

        po_r = po_learner.score(inv)

        meta_signals.append({

            "nlp_risk": nlp_r["risk_score"],

            "dedup_risk": 0.0,

            "iso_forest_risk": ens_r["model_votes"]["isolation_forest"],

            "dbscan_risk": ens_r["model_votes"]["dbscan"],

            "mlp_risk": ens_r["model_votes"]["mlp_nn"],

            "vendor_profile_risk": vend_r["risk_score"],

            "po_tolerance_risk": po_r["risk_score"],

        })

        meta_labels.append(int(inv.get("is_anomaly", 0)))

    meta.fit(meta_signals, meta_labels)

    print(f"  [OK] Meta-learner trained")

    print("\n  Layer 9 - Continuous Learning Engine...")

    learner = ContinuousLearner()

    print(f"  [OK] PSI drift threshold={learner.DRIFT_THRESHOLD}")

    print("\n[OK] ALL 9 AI LAYERS TRAINED SUCCESSFULLY\n")

    header("PROCESSING 6 LIVE ABC INVOICES - MARCH 2026")

    results = []

    risk_scores = []

    for invoice in LIVE_INVOICES:

        print(f"\n{'─'*68}")

        print(f"Invoice: {invoice['invoice_number']}  |  {invoice['vendor_name']}")

        print(f"  Amount: AED {invoice['invoice_amount']:>12,.2f}  |  PO: AED {invoice['po_amount']:>12,.2f}  |  Variance: {invoice['amount_variance_pct']:+.1f}%")

        print(f"{'─'*68}")

        nlp_r = nlp.predict(invoice.get("email_text", ""))

        print(f"\n  L1 NLP   [{bar(nlp_r['risk_score'])}] {nlp_r['risk_score']:.3f}  -> {nlp_r['label_name']}")

        dup_r = dedup.check(invoice)

        dedup_risk = 1.0 if dup_r["is_duplicate"] else dup_r["match_score"]

        print(f"  L2 DEDUP [{bar(dedup_risk)}] {dedup_risk:.3f}  -> {dup_r['match_type']}")

        if dup_r["is_duplicate"]:

            print(f"\n  [REJECTED] Duplicate detected.")

            results.append({"invoice_number": invoice["invoice_number"],

                "vendor": invoice["vendor_name"], "amount": invoice["invoice_amount"],

                "variance_pct": invoice["amount_variance_pct"], "final_risk": 1.0,

                "decision": "REJECTED_DUPLICATE", "top_drivers": "Duplicate", "confidence": "HIGH"})

            continue

        ens_r = ensemble.score(invoice)

        iso_risk = ens_r["model_votes"]["isolation_forest"]

        dbscan_risk = ens_r["model_votes"]["dbscan"]

        mlp_risk = ens_r["model_votes"]["mlp_nn"]

        print(f"  L3 ISO   [{bar(iso_risk)}]  {iso_risk:.3f}")

        print(f"  L4 DBSCAN[{bar(dbscan_risk)}] {dbscan_risk:.3f}")

        print(f"  L5 MLP   [{bar(mlp_risk)}]  {mlp_risk:.3f}")

        vend_r = vendor_profiler.score(invoice)

        print(f"  L6 VENDOR[{bar(vend_r['risk_score'])}] {vend_r['risk_score']:.3f}")

        po_r = po_learner.score(invoice)

        print(f"  L7 PO    [{bar(po_r['risk_score'])}] {po_r['risk_score']:.3f}  -> Actual={po_r['actual_variance']:.1f}% | Tol={po_r['learned_tolerance']:.1f}%")

        signals = {

            "nlp_risk": nlp_r["risk_score"], "dedup_risk": 0.0,

            "iso_forest_risk": iso_risk, "dbscan_risk": dbscan_risk,

            "mlp_risk": mlp_risk, "vendor_profile_risk": vend_r["risk_score"],

            "po_tolerance_risk": po_r["risk_score"],

        }

        meta_r = meta.predict(signals)

        final_decision = meta_r["decision"]

        final_risk = meta_r["final_risk_score"]

        risk_scores.append(final_risk)

        icons = {"AUTO_POST": "[AUTO]", "REVIEW": "[REVIEW]", "HOLD": "[HOLD]"}

        icon = icons.get(final_decision, "[?]")

        print(f"\n  {icon} FINAL DECISION: {final_decision} ({final_risk*100:.1f}% risk)")

        results.append({"invoice_number": invoice["invoice_number"],

            "vendor": invoice["vendor_name"], "amount": invoice["invoice_amount"],

            "variance_pct": invoice["amount_variance_pct"], "final_risk": final_risk,

            "decision": final_decision, "top_drivers": meta_r["top_drivers"],

            "confidence": meta_r["confidence"]})

    drift = learner.check_drift(risk_scores)

    header("PROCESSING SUMMARY - ABC COMPANY")

    print(f"\n  {'Invoice':<20} {'Vendor':<26} {'Amount':>12}  {'Risk':>5}  {'Decision'}")

    print(f"  {'─'*20} {'─'*26} {'─'*12}  {'─'*5}  {'─'*22}")

    auto_post = review = hold = dup = 0

    for r in results:

        score_s = f"{r['final_risk']:.3f}" if r["decision"] != "REJECTED_DUPLICATE" else "  N/A"

        icons = {"AUTO_POST":"[OK]","REVIEW":"[??]","HOLD":"[!!]","REJECTED_DUPLICATE":"[DUP]"}

        icon = icons.get(r["decision"],"")

        print(f"  {r['invoice_number']:<20} {r['vendor'][:25]:<26} {r['amount']:>12,.0f}  {score_s:>5}  {icon} {r['decision']}")

        if r["decision"] == "AUTO_POST": auto_post += 1

        elif r["decision"] == "REVIEW": review += 1

        elif r["decision"] == "HOLD": hold += 1

        elif r["decision"] == "REJECTED_DUPLICATE": dup += 1

    protected = sum(r["amount"] for r in results if r["decision"] in ("HOLD","REJECTED_DUPLICATE","REVIEW"))

    print(f"\n  [OK]  Auto-posted:        {auto_post}")

    print(f"  [??]  Review queue:       {review}")

    print(f"  [!!]  Held/escalated:     {hold}")

    print(f"  [DUP] Duplicate rejected: {dup}")

    print(f"\n  AED protected: {protected:,.2f}")

    print(f"  Model drift: PSI={drift['psi']:.4f} - {drift['status']}")

    print(f"\n{'='*68}")

    print(f"  Pipeline complete - 9 AI layers executed successfully")

    print(f"{'='*68}\n")

    return results

if __name__ == "__main__":

    run_pipeline()
 
