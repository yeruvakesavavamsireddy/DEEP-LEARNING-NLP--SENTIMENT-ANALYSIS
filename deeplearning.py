import warnings
warnings.filterwarnings("ignore")
 
import numpy  as np
import pandas as pd
import re, string
 
from sklearn.pipeline           import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network     import MLPClassifier
from sklearn.model_selection    import train_test_split, learning_curve, cross_val_score
from sklearn.metrics            import (classification_report, confusion_matrix,
                                        accuracy_score, roc_auc_score, roc_curve)
from sklearn.linear_model       import LogisticRegression
from sklearn.ensemble           import GradientBoostingClassifier
 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
 
 
POS_TEMPLATES = [
    "This {noun} is absolutely {adj} and I love it so much!",
    "I had a {adj} experience, would definitely recommend to everyone.",
    "The quality is {adj} and the customer service was excellent.",
    "Amazing {adj} product, totally exceeded all my expectations.",
    "Really {adj} results, very happy with this {noun} overall.",
    "Outstanding {noun}, the features are {adj} and work perfectly.",
    "Brilliant and {adj}, nothing short of spectacular performance.",
    "The {adj} quality blew me away, highly recommended to all.",
    "Five stars! This {noun} is {adj} in every single way.",
    "Incredible {noun}, everything about it is {adj} and wonderful.",
]
NEG_TEMPLATES = [
    "This {noun} is absolutely {adj} and I hate using it.",
    "I had a {adj} experience, would not recommend this to anyone.",
    "The quality is {adj} and the customer service was terrible.",
    "Terrible {adj} {noun}, did not meet any of my expectations.",
    "Really {adj} results, very disappointed with this {noun} overall.",
    "Poor performance, the features are {adj} and barely work at all.",
    "Awful and {adj}, nothing short of a complete disappointment.",
    "The {adj} quality shocked me, I would not recommend it.",
    "One star. This {noun} is {adj} in every single way.",
    "Terrible {noun}, everything about it is {adj} and dreadful.",
]
POS_ADJS = ["fantastic","wonderful","incredible","superb","brilliant",
            "amazing","outstanding","excellent","magnificent","terrific",
            "delightful","impressive","phenomenal","splendid","top-notch"]
NEG_ADJS = ["horrible","terrible","dreadful","awful","disgusting",
            "disappointing","atrocious","pathetic","abysmal","defective",
            "useless","broken","unreliable","substandard","appalling"]
NOUNS    = ["product","service","item","purchase","experience","device","app"]
 
 
def generate_dataset(n=3000):
    np.random.seed(42)
    texts, labels = [], []
    for _ in range(n//2):
        t = np.random.choice(POS_TEMPLATES)
        texts.append(t.format(adj=np.random.choice(POS_ADJS), noun=np.random.choice(NOUNS)))
        labels.append(1)
    for _ in range(n//2):
        t = np.random.choice(NEG_TEMPLATES)
        texts.append(t.format(adj=np.random.choice(NEG_ADJS), noun=np.random.choice(NOUNS)))
        labels.append(0)
    df = pd.DataFrame({"text": texts, "label": labels}).sample(frac=1, random_state=42)
    return df.reset_index(drop=True)
 
 
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
    return re.sub(r"\s+", " ", text).strip()
 
 
def build_models():
    mlp = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1,2), sublinear_tf=True, min_df=2)),
        ("clf",   MLPClassifier(hidden_layer_sizes=(256,128,64), activation="relu",
                                solver="adam", alpha=0.001, learning_rate="adaptive",
                                max_iter=200, early_stopping=True, validation_fraction=0.1,
                                random_state=42, verbose=False))
    ])
    lr = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1,2), sublinear_tf=True)),
        ("clf",   LogisticRegression(C=1.0, max_iter=500, random_state=42))
    ])
    gb = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2), sublinear_tf=True)),
        ("clf",   GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42))
    ])
    return {"MLP (Deep Network)": mlp, "Logistic Regression": lr, "Gradient Boosting": gb}
 
 
def train_and_evaluate(df):
    df["text_clean"] = df["text"].apply(clean_text)
    X, y = df["text_clean"], df["label"]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42,stratify=y)
 
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
    models  = build_models()
    results = {}
    for name, pipe in models.items():
        print(f"\n  Training: {name} ...")
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:,1]
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        cm  = confusion_matrix(y_test, y_pred)
        print(f"  Accuracy: {acc:.4f}  AUC: {auc:.4f}")
        print(classification_report(y_test, y_pred, target_names=["Negative","Positive"]))
        results[name] = {"pipeline":pipe,"acc":acc,"auc":auc,"y_pred":y_pred,"y_prob":y_prob,"cm":cm}
 
    best = max(results, key=lambda k: results[k]["auc"])
    print(f"\n  Best model: {best}")
    return results, X_train, X_test, y_train, y_test, best
 
 
def visualise(results, X_train, X_test, y_train, y_test, best_name):
    fig = plt.figure(figsize=(18,12))
    fig.suptitle("TASK 2 — Deep Learning NLP: Sentiment Analysis", fontsize=16, fontweight="bold")
    gs = gridspec.GridSpec(2,3,figure=fig,hspace=0.40,wspace=0.35)
 
    # Model Comparison
    ax1 = fig.add_subplot(gs[0,0])
    names = list(results.keys())
    accs  = [results[n]["acc"] for n in names]
    aucs  = [results[n]["auc"] for n in names]
    x=np.arange(len(names)); w=0.35
    ax1.bar(x-w/2,accs,w,label="Accuracy",color="#2196F3")
    ax1.bar(x+w/2,aucs,w,label="AUC-ROC", color="#4CAF50")
    ax1.set_xticks(x); ax1.set_xticklabels([n.split("(")[0].strip()[:6] for n in names],fontsize=8)
    ax1.set_ylim(0.5,1.05); ax1.set_title("Model Comparison"); ax1.legend(); ax1.grid(axis="y",alpha=0.3)
 
    # Confusion Matrix
    ax2 = fig.add_subplot(gs[0,1])
    cm=results[best_name]["cm"]
    im=ax2.imshow(cm,cmap="Blues")
    ax2.set_xticks([0,1]); ax2.set_yticks([0,1])
    ax2.set_xticklabels(["Neg","Pos"]); ax2.set_yticklabels(["Neg","Pos"])
    ax2.set_title(f"Confusion Matrix\n({best_name.split('(')[0].strip()})")
    ax2.set_xlabel("Predicted"); ax2.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax2.text(j,i,str(cm[i,j]),ha="center",va="center",fontsize=14,
                     color="white" if cm[i,j]>cm.max()/2 else "black")
    plt.colorbar(im,ax=ax2)
 
    # ROC Curves
    ax3 = fig.add_subplot(gs[0,2])
    cols=["#2196F3","#4CAF50","#FF9800"]
    for (name,res),c in zip(results.items(),cols):
        fpr,tpr,_=roc_curve(y_test,res["y_prob"])
        ax3.plot(fpr,tpr,color=c,lw=2,label=f"{name.split('(')[0].strip()[:12]} ({res['auc']:.3f})")
    ax3.plot([0,1],[0,1],"k--",lw=1.5)
    ax3.set_xlabel("FPR"); ax3.set_ylabel("TPR"); ax3.set_title("ROC Curves")
    ax3.legend(fontsize=8); ax3.grid(alpha=0.3)
 
    # Learning Curve
    ax4 = fig.add_subplot(gs[1,0])
    print("  Computing learning curve (MLP)...")
    ts,tr,vs = learning_curve(results["MLP (Deep Network)"]["pipeline"],X_train,y_train,
                               train_sizes=np.linspace(0.1,1.0,8),cv=5,scoring="accuracy",n_jobs=-1)
    ax4.fill_between(ts,tr.mean(1)-tr.std(1),tr.mean(1)+tr.std(1),alpha=0.2,color="#2196F3")
    ax4.fill_between(ts,vs.mean(1)-vs.std(1),vs.mean(1)+vs.std(1),alpha=0.2,color="#FF5722")
    ax4.plot(ts,tr.mean(1),"o-",color="#2196F3",lw=2,label="Train")
    ax4.plot(ts,vs.mean(1),"o-",color="#FF5722",lw=2,label="Validation")
    ax4.set_xlabel("Samples"); ax4.set_ylabel("Accuracy")
    ax4.set_title("MLP Learning Curve"); ax4.legend(); ax4.grid(alpha=0.3)
 
    # Score Distribution
    ax5 = fig.add_subplot(gs[1,1])
    prob=results[best_name]["y_prob"]
    ax5.hist(prob[y_test==0],bins=30,alpha=0.7,color="#F44336",label="Negative")
    ax5.hist(prob[y_test==1],bins=30,alpha=0.7,color="#4CAF50",label="Positive")
    ax5.axvline(0.5,color="black",linestyle="--")
    ax5.set_title("Prediction Score Distribution")
    ax5.set_xlabel("P(Positive)"); ax5.legend(); ax5.grid(alpha=0.3)
 
    # Per-class metrics
    ax6 = fig.add_subplot(gs[1,2])
    cm2=results[best_name]["cm"]
    pn=cm2[0,0]/(cm2[0,0]+cm2[1,0]+1e-9); pp=cm2[1,1]/(cm2[1,1]+cm2[0,1]+1e-9)
    rn=cm2[0,0]/(cm2[0,0]+cm2[0,1]+1e-9); rp=cm2[1,1]/(cm2[1,1]+cm2[1,0]+1e-9)
    xp=np.arange(2)
    ax6.bar(xp-0.2,[pn,pp],0.35,label="Precision",color="#2196F3")
    ax6.bar(xp+0.2,[rn,rp],0.35,label="Recall",   color="#FF9800")
    ax6.set_xticks(xp); ax6.set_xticklabels(["Negative","Positive"])
    ax6.set_ylim(0,1.15); ax6.set_title("Precision & Recall by Class")
    ax6.legend(); ax6.grid(axis="y",alpha=0.3)
 
    plt.savefig("/mnt/user-data/outputs/task2_deep_learning_results.png",dpi=150,bbox_inches="tight")
    plt.close()
    print("  Visualisation saved -> task2_deep_learning_results.png")
 
 
def demo_predict(pipeline, texts):
    cleaned = [clean_text(t) for t in texts]
    probs   = pipeline.predict_proba(cleaned)[:,1]
    print("\n  Demo Predictions:")
    for text,prob in zip(texts,probs):
        label = "POSITIVE" if prob>=0.5 else "NEGATIVE"
        conf  = prob if prob>=0.5 else 1-prob
        print(f"  [{label} {conf:.1%}] {text[:70]}")
 
 
def main():
    print("\n" + "="*70)
    print("  CODTECH INTERNSHIP — TASK 2: DEEP LEARNING NLP SENTIMENT")
    print("="*70)
 
    print("\n[1/4] Generating dataset...")
    df = generate_dataset(3000)
    print(f"  {len(df)} samples generated")
 
    print("\n[2/4] Training models...")
    results, X_train, X_test, y_train, y_test, best_name = train_and_evaluate(df)
 
    print("\n[3/4] Saving visualisations...")
    visualise(results, X_train, X_test, y_train, y_test, best_name)
 
    print("\n[4/4] Demo inference...")
    demo_predict(results[best_name]["pipeline"],[
        "This is an absolutely fantastic and wonderful product I love it!",
        "Terrible experience, horrible quality, never buying this again.",
        "Really good service, the team was very responsive.",
        "Worst purchase ever, completely broken and awful support.",
        "Exceeded all expectations, brilliant product highly recommended!",
    ])
 
    b = results[best_name]
    print(f"\n  Best: {best_name} | Accuracy={b['acc']:.4f} | AUC={b['auc']:.4f}")
    print("  Task 2 Complete!")
 
 
if __name__ == "__main__":
    main()
