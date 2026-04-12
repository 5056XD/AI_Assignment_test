import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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
    st.header("📈 Evaluation Metrics vs K")
    st.write("This graph illustrates the trade-off between Precision and Recall as the number of recommended items (K) increases.")
    
    if st.button("Generate Metrics vs K Graph (Takes 10-15 seconds)"):
        with st.spinner("Calculating metrics for K=1 through 10..."):
            k_values = list(range(1, 11))
            precisions = []
            recalls = []
            f1_scores = []
            
            # Use a sample of users for speed in the Streamlit app
            sample_users = df_ratings['user_id'].unique()[:15] 
            
            # Pre-calculate scores for speed
            user_relevant_items = defaultdict(list)
            user_recommended_items = defaultdict(list)
            
            for u in sample_users:
                u_ratings = df_ratings[df_ratings['user_id'] == u]
                for _, row in u_ratings.iterrows():
                    m = row['model']
                    act = row['rating']
                    
                    if act >= 4.0:
                        user_relevant_items[u].append(m)
                        
                    # Calculate hybrid score
                    cb = predict_cb(u, m)
                    cf = predict_cf(u, m)
                    glob = df_items.loc[df_items['model'] == m, 'normalized_avg_rating'].values[0]
                    hyb = (0.5 * cf) + (0.3 * cb) + (0.2 * glob)
                    
                    user_recommended_items[u].append((m, hyb))
            
            # Loop through different values of K
            for k in k_values:
                k_prec, k_rec = [], []
                for u in sample_users:
                    recs = sorted(user_recommended_items[u], key=lambda x: x[1], reverse=True)[:k]
                    top_k_items = [item for item, score in recs]
                    rel_items = user_relevant_items[u]
                    
                    hits = len(set(top_k_items).intersection(set(rel_items)))
                    k_prec.append(hits / k)
                    k_rec.append(hits / len(rel_items) if len(rel_items) > 0 else 1)
                
                avg_prec = np.mean(k_prec)
                avg_rec = np.mean(k_rec)
                f1 = 2 * (avg_prec * avg_rec) / (avg_prec + avg_rec) if (avg_prec + avg_rec) > 0 else 0
                
                precisions.append(avg_prec)
                recalls.append(avg_rec)
                f1_scores.append(f1)
            
            # ----------------------------------------
            # Draw the Plotly Line Graph
            # ----------------------------------------
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(x=k_values, y=precisions, mode='lines+markers', name='Precision@K', line=dict(color='blue', width=3)))
            fig.add_trace(go.Scatter(x=k_values, y=recalls, mode='lines+markers', name='Recall@K', line=dict(color='green', width=3)))
            fig.add_trace(go.Scatter(x=k_values, y=f1_scores, mode='lines+markers', name='F1-Score', line=dict(color='red', width=3, dash='dash')))
            
            fig.update_layout(
                title="Performance Metrics Across Different Top-K Recommendations",
                xaxis_title="Number of Recommendations (K)",
                yaxis_title="Score (0.0 to 1.0)",
                xaxis=dict(tickmode='linear', dtick=1),
                yaxis=dict(range=[0, 1.05]), # Keep Y axis scaled properly
                hovermode="x unified",
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            
            st.plotly_chart(fig, use_container_width=True)