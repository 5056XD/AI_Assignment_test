import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Hybrid AI Recommender", layout="wide")

# ==========================================
# 1. LOAD & PREPROCESS DATA
# ==========================================
@st.cache_data
def load_data():
    df = pd.read_csv('smartphones.csv')
    df = df.drop_duplicates(subset=['model']).reset_index(drop=True)
    
    df['avg_rating'] = df['avg_rating'].fillna(df['avg_rating'].mean())
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('')
        else:
            df[col] = df[col].fillna(0)
            
    scaler = MinMaxScaler(feature_range=(0, 5))
    df['normalized_avg_rating'] = scaler.fit_transform(df[['avg_rating']])
    
    df['content_features'] = (
         df['brand_name'].astype(str) + ' ' + 
         df['os'].astype(str) + ' ' + 
         df['processor_brand'].astype(str) + ' ' +
         df['ram_capacity'].astype(str) + 'gb ' + 
         df['internal_memory'].astype(str) + 'gb'
    ).fillna('').str.lower()
    
    return df

df_items = load_data()

# ==========================================
# 2. AI ENGINES SETUP
# ==========================================
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_items['content_features'])
content_sim_matrix = cosine_similarity(tfidf_matrix)
content_sim_df = pd.DataFrame(content_sim_matrix, index=df_items['model'], columns=df_items['model'])

np.random.seed(42)
sample_phones = df_items['model'].sample(150, random_state=42).tolist()
mock_ratings = []
for u_id in range(1, 51):
    n = np.random.randint(5, 20)
    for p in np.random.choice(sample_phones, n, replace=False):
        mock_ratings.append({'user_id': u_id, 'model': p, 'rating': float(np.random.randint(1, 6))})
df_ratings = pd.DataFrame(mock_ratings)

user_item_matrix = df_ratings.pivot_table(index='user_id', columns='model', values='rating').fillna(0)
user_sim_df = pd.DataFrame(cosine_similarity(user_item_matrix), index=user_item_matrix.index, columns=user_item_matrix.index)

def predict_cb(u_id, item):
    u_ratings = df_ratings[df_ratings['user_id'] == u_id]
    num, den = 0, 0
    for _, row in u_ratings.iterrows():
        sim = content_sim_df.loc[item, row['model']]
        num += sim * row['rating']
        den += sim
    return num / den if den > 0 else 3.0

def predict_cf(u_id, item):
    if item not in user_item_matrix.columns: return 3.0
    sim_users = user_sim_df[u_id].drop(u_id)
    item_rats = user_item_matrix[item].drop(u_id)
    mask = item_rats > 0
    if not mask.any(): return 3.0
    return np.dot(sim_users[mask], item_rats[mask]) / sim_users[mask].sum()

# ==========================================
# 3. MAIN DASHBOARD
# ==========================================
st.title("📱 Hybrid Smartphone Recommender")

tab1, tab2 = st.tabs(["🎯 Live Recommendations", "📊 System Evaluation"])

# --- TAB 1: RECOMMENDATIONS ---
with tab1:
    selected_user = st.slider("Select User ID to generate recommendations:", 1, 50, 1)
    st.subheader(f"Top 5 Recommendations for User {selected_user}")
    
    with st.spinner("Calculating..."):
        all_models = df_items['model'].unique()
        rated = df_ratings[df_ratings['user_id'] == selected_user]['model'].tolist()
        unrated = [m for m in all_models if m not in rated]
        
        results = []
        for m in unrated[:100]: 
            cb = predict_cb(selected_user, m)
            cf = predict_cf(selected_user, m)
            glob = df_items.loc[df_items['model'] == m, 'normalized_avg_rating'].values[0]
            final = (0.5 * cf) + (0.3 * cb) + (0.2 * glob)
            
            results.append({
                'Smartphone': m, 
                'Hybrid Score': round(final, 2), 
                'Content': round(cb, 2), 
                'Collab': round(cf, 2)
            })
        
        recs = pd.DataFrame(results).sort_values(by='Hybrid Score', ascending=False).head(5)
        st.dataframe(recs, use_container_width=True)

# --- TAB 2: METRICS VS K (YOUR EXACT CODE) ---
with tab2:
    st.subheader("📊 Evaluation Metrics vs K")

    if st.button("Generate Evaluation Chart"):
        with st.spinner("Running calculations for K=1 to 10..."):
            k_values = list(range(1, 11))
            precision_scores = []
            recall_scores = []
            f1_scores = []

            # 1. Pre-calculate user data for speed
            sample_users = df_ratings['user_id'].unique()[:15] 
            user_relevant = defaultdict(list)
            user_recs = defaultdict(list)

            for u in sample_users:
                u_ratings = df_ratings[df_ratings['user_id'] == u]
                for _, row in u_ratings.iterrows():
                    m = row['model']
                    if row['rating'] >= 4.0:
                        user_relevant[u].append(m)
                    
                    cb = predict_cb(u, m)
                    cf = predict_cf(u, m)
                    glob = df_items.loc[df_items['model'] == m, 'normalized_avg_rating'].values[0] if m in df_items['model'].values else 0
                    hyb = (0.5 * cf) + (0.3 * cb) + (0.2 * glob)
                    user_recs[u].append((m, hyb))

            # 2. Run the K-Loop
            for k in k_values:
                k_prec, k_rec = [], []
                for u in sample_users:
                    sorted_recs = sorted(user_recs[u], key=lambda x: x[1], reverse=True)[:k]
                    top_k_items = [item for item, score in sorted_recs]
                    rel_items = user_relevant[u]
                    
                    hits = len(set(top_k_items).intersection(set(rel_items)))
                    k_prec.append(hits / k)
                    k_rec.append(hits / len(rel_items) if len(rel_items) > 0 else 1)
                
                avg_p = np.mean(k_prec)
                avg_r = np.mean(k_rec)
                f1 = 2 * (avg_p * avg_r) / (avg_p + avg_r) if (avg_p + avg_r) > 0 else 0
                
                precision_scores.append(avg_p)
                recall_scores.append(avg_r)
                f1_scores.append(f1)

            # ==============================
            # PLOT (Your Code)
            # ==============================
            fig, ax = plt.subplots()

            ax.plot(k_values, precision_scores, marker='o', label='Precision')
            ax.plot(k_values, recall_scores, marker='s', label='Recall')
            ax.plot(k_values, f1_scores, marker='^', label='F1-score')

            ax.set_xlabel("K (Top Recommendations)")
            ax.set_ylabel("Score")
            ax.set_title("Evaluation Metrics vs K")
            ax.legend()

            st.pyplot(fig)

            # ==============================
            # TABLE (Your Code)
            # ==============================
            result_df = pd.DataFrame({
                'K': k_values,
                'Precision': precision_scores,
                'Recall': recall_scores,
                'F1-score': f1_scores
            })

            st.dataframe(result_df, use_container_width=True)

            # ==============================
            # AVERAGE METRICS (Your Code)
            # ==============================
            avg_precision = np.mean(precision_scores)
            avg_recall = np.mean(recall_scores)
            avg_f1 = np.mean(f1_scores)

            st.subheader("📌 Average Evaluation Metrics")

            col1, col2, col3 = st.columns(3)

            col1.metric("Avg Precision", f"{avg_precision:.4f}")
            col2.metric("Avg Recall", f"{avg_recall:.4f}")
            col3.metric("Avg F1-score", f"{avg_f1:.4f}")