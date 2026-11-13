import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay

st.set_page_config(page_title="Customer Churn Analysis", page_icon="📊", layout="wide")
st.title("📊 Customer Churn Analysis")
st.caption("Explore subscription cancellations, customer engagement, and retention opportunities.")

uploaded = st.sidebar.file_uploader("Upload your churn CSV (optional)", type=["csv"])
df = pd.read_csv(uploaded) if uploaded else pd.read_csv("customer_churn.csv")
if "churn" not in df.columns:
    st.error("Dataset must contain a 'churn' column (0 = retained, 1 = churned).")
    st.stop()

if df["churn"].dtype == "object":
    df["churn"] = df["churn"].astype(str).str.lower().map({"yes": 1, "true": 1, "1": 1, "no": 0, "false": 0, "0": 0})
df = df.dropna(subset=["churn"])
df["churn"] = df["churn"].astype(int)

total = len(df)
churn_rate = df["churn"].mean() * 100
retained = int((df["churn"] == 0).sum())
churned = int((df["churn"] == 1).sum())
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total customers", f"{total:,}")
c2.metric("Churned customers", f"{churned:,}")
c3.metric("Retained customers", f"{retained:,}")
c4.metric("Churn rate", f"{churn_rate:.1f}%")

st.subheader("Dataset preview")
st.dataframe(df.head(12), use_container_width=True)
left, right = st.columns(2)
with left:
    st.subheader("Churn distribution")
    counts = df["churn"].map({0: "Retained", 1: "Churned"}).value_counts()
    fig, ax = plt.subplots()
    ax.bar(counts.index, counts.values)
    ax.set_ylabel("Customers")
    st.pyplot(fig)
    plt.close(fig)
with right:
    st.subheader("Churn by contract type")
    if "contract_type" in df.columns:
        rate = df.groupby("contract_type")["churn"].mean().mul(100).sort_values()
        fig, ax = plt.subplots()
        ax.bar(rate.index.astype(str), rate.values)
        ax.set_ylabel("Churn rate (%)")
        ax.tick_params(axis="x", rotation=15)
        st.pyplot(fig)
        plt.close(fig)

numeric = df.select_dtypes(include="number").drop(columns=["churn"], errors="ignore")
if not numeric.empty:
    st.subheader("Customer behaviour vs churn")
    feature = st.selectbox("Choose a numeric feature", numeric.columns)
    fig, ax = plt.subplots()
    groups = [df.loc[df["churn"] == v, feature].dropna() for v in [0, 1]]
    ax.boxplot(groups, labels=["Retained", "Churned"])
    ax.set_ylabel(feature)
    st.pyplot(fig)
    plt.close(fig)

st.subheader("Key findings")
if "contract_type" in df.columns:
    rates = df.groupby("contract_type")["churn"].mean().mul(100).sort_values(ascending=False)
    st.write(f"- Highest observed churn by contract: **{rates.index[0]}** ({rates.iloc[0]:.1f}%).")
for col, label, unit in [("support_tickets", "Average support tickets", ""), ("weekly_usage_hours", "Average weekly usage", " hours")]:
    if col in df.columns:
        means = df.groupby("churn")[col].mean()
        if len(means) == 2:
            st.write(f"- {label}: **{means.loc[1]:.2f}{unit}** for churned vs **{means.loc[0]:.2f}{unit}** for retained customers.")

st.subheader("Churn prediction model")
st.caption("Baseline Logistic Regression. Results depend on the dataset and do not guarantee future customer behaviour.")
X = df.drop(columns=["churn", "customer_id"], errors="ignore")
y = df["churn"]
if y.nunique() < 2 or len(df) < 30:
    st.info("At least 30 rows and both churn classes are required to train the model.")
else:
    categorical = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numerical = X.select_dtypes(include="number").columns.tolist()
    transformers = []
    if numerical:
        transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numerical))
    if categorical:
        transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical))
    preprocess = ColumnTransformer(transformers)
    model = Pipeline([("prep", preprocess), ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=stratify)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{accuracy_score(y_test, pred):.2f}")
    m2.metric("Precision", f"{precision_score(y_test, pred, zero_division=0):.2f}")
    m3.metric("Recall", f"{recall_score(y_test, pred, zero_division=0):.2f}")
    m4.metric("F1 score", f"{f1_score(y_test, pred, zero_division=0):.2f}")
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax, colorbar=False)
    st.pyplot(fig)
    plt.close(fig)

st.subheader("Retention recommendations")
st.markdown("""
1. **Engagement outreach:** contact customers whose product usage has declined.
2. **Support improvement:** review repeated support requests and resolve recurring issues.
3. **Contract incentives:** explain the benefits of longer-term plans without pressuring customers.
4. **Personalized offers:** test relevant offers with at-risk customer groups.
5. **Measure results:** compare churn rates before and after each retention campaign.
""")
st.info("This repository includes a simulated example dataset for learning. Replace it with an authorized, anonymized business dataset for real analysis.")
