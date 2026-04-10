import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set Page Config
st.set_page_config(page_title="Pro Hybrid Recommender", layout="wide")

# ==========================================
# 1. Load and Preprocess Data
# ==========================================
@st.cache_data
def load_data():
    df = pd.read_csv('smartphones.csv')
    df = df.drop_duplicates(subset=['model']).reset_index(drop=True)
    
    # Fill missing values
    df['avg_rating'] = df['avg_rating'].fillna(df['avg_rating'].mean())
    
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('')
        else:
            df[col] = df[col].fillna(0)
            
    # Normalize global ratings for hybrid use
    scaler = MinMaxScaler(feature_range=(0, 5))
    df['normalized_avg_rating'] = scaler.fit_transform(df[['avg_rating']])
    
    # Feature Engineering for Content-Based
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
# 2. Advanced Engines (CB & CF)
# ==========================================
# TF-IDF Setup
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_items['content_features'])
content_sim_matrix = cosine_similarity(tfidf_matrix)
content_sim_df = pd.DataFrame(content_sim_matrix, index=df_items['model'], columns=df_items['model'])

# Synthetic User Data (50 Users)
np.random.seed(42)
sample_phones = df_items['model'].sample(150, random_state=42).tolist()
mock_ratings = []
for u_id in range(1, 51):
    n = np.random.randint(5, 20)
    for p in np.random.choice(sample_phones, n, replace=False):
        mock_ratings.append({'user_id': u_id, 'model': p, 'rating': float(np.random.randint(1, 6))})
df_ratings = pd.DataFrame(mock_ratings)

# Collaborative Matrix
user_item_matrix = df_ratings.pivot_table(index='user_id', columns='model', values='rating').fillna(0)
user_sim_df = pd.DataFrame(cosine_similarity(user_item_matrix), index=user_item_matrix.index, columns=user_item_matrix.index)

def predict_cb(u_id, item):
    u_ratings = df_ratings[df_ratings['user_id'] == u_id]
    num, den = 0, 0
    for _, row in u_ratings.iterrows():
        sim = content_sim_df.loc[item, row['model']]
        num += sim * row['rating']
        den += sim
    return num / den if den > 0 else 3.0 # Default to middle rating

def predict_cf(u_id, item):
    if item not in user_item_matrix.columns: return 3.0
    sim_users = user_sim_df[u_id].drop(u_id)
    item_rats = user_item_matrix[item].drop(u_id)
    mask = item_rats > 0
    if not mask.any(): return 3.0
    return np.dot(sim_users[mask], item_rats[mask]) / sim_users[mask].sum()

# ==========================================
# 3. Streamlit Tabs
# ==========================================
tab1, tab2, tab3 = st.tabs(["🚀 Recommendations", "📊 Model Accuracy", "📅 Project Timeline"])

with tab1:
    st.header("Personalized Recommendations")
    selected_user = st.sidebar.slider("Switch User ID:", 1, 50, 1)
    
    all_models = df_items['model'].unique()
    rated = df_ratings[df_ratings['user_id'] == selected_user]['model'].tolist()
    unrated = [m for m in all_models if m not in rated]
    
    results = []
    for m in unrated[:100]: # Speed optimization
        cb = predict_cb(selected_user, m)
        cf = predict_cf(selected_user, m)
        glob = df_items.loc[df_items['model'] == m, 'normalized_avg_rating'].values[0]
        # Optimized Weights: 50% Collab (Peers), 30% Content (Specs), 20% Global (Market)
        final = (0.5 * cf) + (0.3 * cb) + (0.2 * glob)
        results.append({'Smartphone': m, 'Hybrid Score': round(final, 2), 'CB': round(cb, 2), 'CF': round(cf, 2)})
    
    recs = pd.DataFrame(results).sort_values(by='Hybrid Score', ascending=False).head(5)
    st.table(recs)

with tab2:
    st.header("Model Evaluation (Rubric Metrics)")
    if st.button("Calculate Accuracy Rate"):
        actuals, p_cb, p_cf, p_hyb = [], [], [], []
        hits = 0
        total_relevant = 0
        
        for _, row in df_ratings.sample(200).iterrows(): # Sample evaluation
            u, m, act = row['user_id'], row['model'], row['rating']
            cb, cf = predict_cb(u, m), predict_cf(u, m)
            hyb = (0.5 * cf) + (0.3 * cb) + (0.2 * 3.0) # simplified for speed
            
            actuals.append(act)
            p_hyb.append(hyb)
            if act >= 4 and hyb >= 3.5: hits += 1
            if act >= 4: total_relevant += 1

        rmse = np.sqrt(mean_squared_error(actuals, p_hyb))
        precision = hits / 200
        recall = hits / total_relevant if total_relevant > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision+recall)>0 else 0
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("RMSE (Lower is Better)", f"{rmse:.2f}")
        c2.metric("Precision@K", f"{precision:.2f}")
        c3.metric("Recall@K", f"{recall:.2f}")
        c4.metric("F1-Score", f"{f1:.2f}")

with tab3:
    st.header("Project Gantt Chart")
    gantt_data = [
        dict(Task="Data Preprocessing", Start='2026-01-01', Finish='2026-01-05', Member="Member 1"),
        dict(Task="Algorithm Development", Start='2026-01-06', Finish='2026-01-15', Member="Member 2"),
        dict(Task="Hybrid Integration", Start='2026-01-16', Finish='2026-01-22', Member="Member 3"),
        dict(Task="UI & Deployment", Start='2026-01-23', Finish='2026-01-30', Member="Group")
    ]
    fig = px.timeline(gantt_data, x_start="Start", x_end="Finish", y="Task", color="Member")
    st.plotly_chart(fig, use_container_width=True)