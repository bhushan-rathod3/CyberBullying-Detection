import pandas as pd, numpy as np, re, os, warnings, joblib
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, ConfusionMatrixDisplay

KAGGLE, SYNTH = "cyberbullying_tweets.csv", "comments.csv"
HARDNEG = "hard_negatives.csv"      # optional: Gemini-made benign tweets about identity groups (same 2 columns)
GOLD    = "gold_to_label.csv"        # optional: real tweets YOU labelled (columns: tweet_text, cyberbullying_type)

# ---- ONE cleaning function for ALL sources (so style differences are not a free clue) ----
def clean(t):
    t = str(t).lower()
    t = re.sub(r"http\S+|www\S+", " ", t); t = re.sub(r"@\w+", " ", t); t = re.sub(r"^rt ", " ", t)
    t = re.sub(r"#", " ", t)                      # keep the hashtag WORD, drop the symbol
    return " ".join(re.sub(r"[^a-z\s']", " ", t).split())   # keeps 'not', "n't" etc. (no stopword removal)

def load(path):
    d = pd.read_csv(path)[["tweet_text", "cyberbullying_type"]].dropna()
    d = d.drop_duplicates("tweet_text", keep=False)
    d["clean"] = d.tweet_text.map(clean)
    return d.drop_duplicates("clean", keep=False).reset_index(drop=True)

def split(d): return train_test_split(d, test_size=0.2, stratify=d.cyberbullying_type, random_state=42)

sets = {"Kaggle": load(KAGGLE), "Gemini": load(SYNTH)}
if os.path.exists(HARDNEG): sets["HardNeg"] = load(HARDNEG)
parts = {k: split(v) if len(v) > 50 else (v, v.iloc[:0]) for k, v in sets.items()}
train = {k: p[0] for k, p in parts.items()}; test = {k: p[1] for k, p in parts.items() if len(p[1])}
if os.path.exists(GOLD): g = load(GOLD); test["GOLD (real, hand-labelled)"] = g

def make(): return Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
    ("clf", LogisticRegression(max_iter=3000, C=5, class_weight="balanced"))])

models = {"Kaggle only": ["Kaggle"], "Gemini only": ["Gemini"], "COMBINED": list(train)}
fitted = {}
rows = []
SMALL_SET_WEIGHT = 10   # Gemini / hard-negative rows are few but clean -> count each as 10 rows (tested: 1,5,10,20,40)
for name, srcs in models.items():
    tr = pd.concat([train[s].assign(src=s) for s in srcs])
    w = np.where(tr.src == "Kaggle", 1.0, SMALL_SET_WEIGHT)
    m = make(); m.fit(tr.clean, tr.cyberbullying_type, clf__sample_weight=w); fitted[name] = m
    for tname, te in test.items():
        p = m.predict(te.clean)
        rows.append(dict(model=name, tested_on=tname, n=len(te), acc=round(accuracy_score(te.cyberbullying_type, p), 3),
                         macro_f1=round(f1_score(te.cyberbullying_type, p, average="macro"), 3)))
R = pd.DataFrame(rows)
print("=== every model on every test set (the cross-dataset table) ===")
print(R.pivot(index="model", columns="tested_on", values="acc").to_string())

final = fitted["COMBINED"]
for tname, te in test.items():
    p = final.predict(te.clean); print(f"\n--- COMBINED model on {tname} ---"); print(classification_report(te.cyberbullying_type, p, digits=3, zero_division=0))
te = test["Kaggle"]; ConfusionMatrixDisplay.from_predictions(te.cyberbullying_type, final.predict(te.clean), xticks_rotation=45, cmap="Blues")
plt.tight_layout(); plt.savefig("confusion_kaggle.png", dpi=110); plt.close()
joblib.dump(final, "model.joblib"); print("\nsaved model.joblib")

# ---- explanation: which words pushed the model toward the predicted class? ----
def predict_explain(text, k=5):
    tf, clf = final.named_steps["tfidf"], final.named_steps["clf"]
    x = tf.transform([clean(text)]); idx = x.nonzero()[1]; vocab = tf.get_feature_names_out()
    proba = clf.predict_proba(x)[0]; order = proba.argsort()[::-1]; top = clf.classes_[order[0]]
    contrib = x[0, idx].toarray().ravel() * clf.coef_[order[0]][idx]; best = np.argsort(contrib)[::-1][:k]
    sure = proba[order[0]] >= 0.6 and len(idx) >= 3     # not-sure rule: low confidence or <3 known words
    return {"verdict": top if sure else "UNSURE (" + top + "?)", "prediction": top, "confidence": round(float(proba[order[0]]), 3),
            "runner_up": (clf.classes_[order[1]], round(float(proba[order[1]]), 3)),
            "key_words": [(vocab[idx[j]], round(float(contrib[j]), 2)) for j in best if contrib[j] > 0],
            "words_seen_by_model": len(idx)}

if __name__ == "__main__":
    for t in ["boomers need to step down from office already", "can someone help me configure a docker container?",
              "go back to your country nobody wants u here", "my muslim friend is the kindest person i know",
              "women shouldn't work in engineering fields", "school :-/"]:
        print(t, "->", predict_explain(t))