"""Credit Card Fraud Detection. Usage: python fraud_detection.py [train.csv] [test.csv]
Works with Kaggle fraudTrain/fraudTest.csv or creditcard.csv (single file)."""
import sys, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

def prep(df):
    df = df.copy()
    tgt = "is_fraud" if "is_fraud" in df else "Class"
    if "trans_date_trans_time" in df:
        t = pd.to_datetime(df["trans_date_trans_time"])
        df["hour"] = t.dt.hour
        if "dob" in df: df["age"] = t.dt.year - pd.to_datetime(df["dob"]).dt.year
    drop = ["Unnamed: 0","cc_num","zip","unix_time","trans_num","first","last","street","city",
            "state","job","merchant","dob","trans_date_trans_time"]
    df = df.drop(columns=[c for c in drop if c in df])
    for c in ("category","gender"):
        if c in df: df = pd.get_dummies(df, columns=[c], drop_first=True)
    y = df.pop(tgt)
    return df.select_dtypes(include=[np.number, bool]).astype(float), y

args = sys.argv[1:] or ["fraudTrain.csv", "fraudTest.csv"]
tr = pd.read_csv(args[0])
if len(args) > 1:
    te = pd.read_csv(args[1])
    Xtr, ytr = prep(tr); Xte, yte = prep(te)
    Xte = Xte.reindex(columns=Xtr.columns, fill_value=0)
else:
    X, y = prep(tr)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

sc = StandardScaler().fit(Xtr)
Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(max_depth=12, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1, class_weight="balanced_subsample",
                                            min_samples_leaf=2, random_state=42),
}
for n, m in models.items():
    m.fit(Xtr_s, ytr)
    p = m.predict(Xte_s)
    print(f"\n=== {n} ===  ROC-AUC: {roc_auc_score(yte, m.predict_proba(Xte_s)[:,1]):.4f}")
    print(classification_report(yte, p, target_names=["legit","fraud"], digits=4))
