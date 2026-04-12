import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from collections import defaultdict
import matplotlib.pyplot as plt

st.set_page_config(page_title="Hybrid Recommender", layout="wide")

# ==========================================
# LOAD DATA
# ==========================================
@st.cache_data
def load_data():
    df = pd.read_csv('smartphones.csv')
    df = df.drop_duplicates(subset=['model']).reset_index(drop=True)

    df = df.fillna('')
    df['avg_rating'] = pd.to_numeric(df['avg_rating'], errors='coerce').fillna(3.0)

    df['content_features'] = (
        df['brand_name'] + " " +
        df['os'] + " " +
        df['processor_brand'] + " " +
        df['ram_capacity'].astype(str) + "GB " +
        df['internal_memory'].astype(str) + "GB"
    ).str.lower()

    return df

df_items = load_data()

# ==========================================
# CONTENT-BASED MODEL
# ==========================================
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_items['content_features'])
content_sim = cosine_similarity(tfidf_matrix)

content_sim_df = pd.DataFrame(
    content_sim,
    index=df_items['model'],
    columns=df_items['model']
)

# ==========================================
# CREATE MOCK USER RATINGS
# ==========================================
np.random.seed(42)

users = 50
models = df_items['model'].sample(150, random_state=42).tolist()

ratings = []
for u in range(1, users+1):
    for m in np.random.choice(models, np.random.randint(5, 20), replace=False):
        ratings.append({
            "user": u,
            "model": m,
            "rating": float(np.random.randint(1, 6))
        })

df_ratings = pd.DataFrame(ratings)

# ==========================================
# TRAIN TEST SPLIT (IMPORTANT FIX)
# ==========================================
train, test = train_test_split(df_ratings, test_size=0.2, random_state=42)

# USER-ITEM MATRIX
user_item = train.pivot_table(index='user', columns='model', values='rating').fillna(0)
user_sim = cosine_similarity(user_item)

user_sim_df = pd.DataFrame(user_sim, index=user_item.index, columns=user_item.index)

# ==========================================
# PREDICTION FUNCTIONS
# ==========================================
def predict_cb(user, item):
    user_data = train[train['user'] == user]
    
    if user_data.empty:
        return 3.0  # cold start

    num, den = 0, 0
    for _, row in user_data.iterrows():
        sim = content_sim_df.loc[item, row['model']]
        num += sim * row['rating']
        den += sim

    return num / den if den > 0 else 3.0


def predict_cf(user, item):
    if item not in user_item.columns:
        return 3.0

    sim_users = user_sim_df[user].drop(user)
    item_ratings = user_item[item]

    mask = item_ratings > 0
    if not mask.any():
        return 3.0

    return np.dot(sim_users[mask], item_ratings[mask]) / sim_users[mask].sum()


def predict_hybrid(user, item):
    cb = predict_cb(user, item)
    cf = predict_cf(user, item)

    # REQUIRED FORMULA
    return (0.5 * cb) + (0.5 * cf)

# ==========================================
# EVALUATION FUNCTIONS
# ==========================================
def evaluate_rmse():
    y_true, y_cb, y_cf, y_hyb = [], [], [], []

    for _, row in test.iterrows():
        u, m, r = row['user'], row['model'], row['rating']

        cb = predict_cb(u, m)
        cf = predict_cf(u, m)
        hyb = predict_hybrid(u, m)

        y_true.append(r)
        y_cb.append(cb)
        y_cf.append(cf)
        y_hyb.append(hyb)

    return (
        mean_squared_error(y_true, y_cb, squared=False),
        mean_squared_error(y_true, y_cf, squared=False),
        mean_squared_error(y_true, y_hyb, squared=False),
    )


def precision_recall_f1(k=5):
    users = test['user'].unique()[:10]

    precision, recall = [], []

    for u in users:
        user_test = test[test['user'] == u]

        relevant = user_test[user_test['rating'] >= 4]['model'].tolist()

        scores = []
        for m in df_items['model']:
            score = predict_hybrid(u, m)
            scores.append((m, score))

        top_k = sorted(scores, key=lambda x: x[1], reverse=True)[:k]
        top_k_items = [i[0] for i in top_k]

        hits = len(set(top_k_items) & set(relevant))

        precision.append(hits / k)
        recall.append(hits / len(relevant) if relevant else 1)

    p = np.mean(precision)
    r = np.mean(recall)
    f1 = 2 * p * r / (p + r) if (p + r) else 0

    return p, r, f1

# ==========================================
# STREAMLIT UI
# ==========================================
st.title("📱 Hybrid Smartphone Recommender")

tab1, tab2 = st.tabs(["Recommendations", "Evaluation"])

# =========================
# TAB 1
# =========================
with tab1:
    user_id = st.slider("Select User", 1, 50, 1)

    results = []
    for m in df_items['model']:
        score = predict_hybrid(user_id, m)
        results.append((m, score))

    top = sorted(results, key=lambda x: x[1], reverse=True)[:5]

    st.subheader("Top 5 Recommendations")
    st.table(pd.DataFrame(top, columns=["Smartphone", "Score"]))

# =========================
# TAB 2
# =========================
with tab2:
    if st.button("Run Evaluation"):
        rmse_cb, rmse_cf, rmse_hyb = evaluate_rmse()
        p, r, f1 = precision_recall_f1()

        st.subheader("RMSE Comparison")
        st.write(f"Content-Based RMSE: {rmse_cb:.3f}")
        st.write(f"Collaborative RMSE: {rmse_cf:.3f}")
        st.write(f"Hybrid RMSE: {rmse_hyb:.3f}")

        st.subheader("Top-K Metrics (K=5)")
        st.write(f"Precision: {p:.3f}")
        st.write(f"Recall: {r:.3f}")
        st.write(f"F1 Score: {f1:.3f}")