import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. Load Data & Prepare Features
# ==========================================
df_items = pd.read_csv('smartphones.csv')
df_items = df_items.drop_duplicates(subset=['model']).reset_index(drop=True)

# Handle missing 'avg_rating' by filling it with the mean rating of all phones
mean_rating = df_items['avg_rating'].mean()
df_items['avg_rating'].fillna(mean_rating, inplace=True)

# Normalize avg_rating to a 0-5 scale so it matches our other scores
scaler = MinMaxScaler(feature_range=(0, 5))
df_items['normalized_avg_rating'] = scaler.fit_transform(df_items[['avg_rating']])

# Handle other missing values safely based on their data type
for col in df_items.columns:
    if df_items[col].dtype == 'object':  # If it's a text column
        df_items[col] = df_items[col].fillna('')
    else:  # If it's a number column
        df_items[col] = df_items[col].fillna(0)
df_items['content_features'] = (
    df_items['brand_name'].astype(str) + ' ' +
    df_items['os'].astype(str) + ' ' +
    df_items['processor_brand'].astype(str) + ' ' +
    df_items['ram_capacity'].astype(str) + 'GB RAM ' +
    df_items['internal_memory'].astype(str) + 'GB Storage'
)

# ==========================================
# 2. Generate Synthetic User Interactions
# ==========================================
# (Still required because Collaborative Filtering needs user-to-user mappings)
np.random.seed(42)
sample_phones = df_items['model'].sample(200, random_state=42).tolist()
df_subset = df_items[df_items['model'].isin(sample_phones)].reset_index(drop=True)

mock_ratings = []
for user_id in range(1, 51):
    num_ratings = np.random.randint(10, 30)
    rated_phones = np.random.choice(sample_phones, num_ratings, replace=False)
    for phone in rated_phones:
        mock_ratings.append({'user_id': user_id, 'model': phone, 'rating': float(np.random.randint(1, 6))})

df_ratings = pd.DataFrame(mock_ratings)

# ==========================================
# 3. Content-Based & Collaborative Setup
# ==========================================
# Content-Based (Cosine Similarity)
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_subset['content_features'])
content_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
content_sim_df = pd.DataFrame(content_sim_matrix, index=df_subset['model'], columns=df_subset['model'])

def get_content_prediction(user_id, item_id, df_ratings, sim_matrix):
    user_ratings = df_ratings[df_ratings['user_id'] == user_id]
    num, den = 0, 0
    for _, row in user_ratings.iterrows():
        if row['model'] in sim_matrix.columns and item_id in sim_matrix.columns:
            sim = sim_matrix.loc[item_id, row['model']]
            num += sim * row['rating']
            den += sim
    return num / den if den != 0 else 0

# Collaborative (User-Item Matrix)
user_item_matrix = df_ratings.pivot_table(index='user_id', columns='model', values='rating').fillna(0)
user_sim_matrix = cosine_similarity(user_item_matrix)
user_sim_df = pd.DataFrame(user_sim_matrix, index=user_item_matrix.index, columns=user_item_matrix.index)

def get_collab_prediction(user_id, item_id, user_item_matrix, user_sim_df):
    if item_id not in user_item_matrix.columns or user_id not in user_sim_df.index: return 0
    sim_users = user_sim_df[user_id].drop(user_id)
    item_rats = user_item_matrix[item_id].drop(user_id)
    mask = item_rats > 0
    sim_users, item_rats = sim_users[mask], item_rats[mask]
    return np.dot(sim_users, item_rats) / sim_users.sum() if sim_users.sum() != 0 else 0

# ==========================================
# 4. NEW: 3-Way Hybrid Scoring System
# ==========================================
def get_hybrid_recommendations(user_id, top_n=5):
    all_items = df_subset['model'].unique()
    rated_items = df_ratings[df_ratings['user_id'] == user_id]['model'].tolist()
    unrated_items = [item for item in all_items if item not in rated_items]
    
    predictions = []
    
    for item in unrated_items:
        # 1. Get Content Score
        cb_score = get_content_prediction(user_id, item, df_ratings, content_sim_df)
        
        # 2. Get Collaborative Score
        cf_score = get_collab_prediction(user_id, item, user_item_matrix, user_sim_df)
        
        # 3. Get Global Popularity (Your avg_rating column!)
        # We extract the normalized avg rating for this specific phone
        global_avg_score = df_subset.loc[df_subset['model'] == item, 'normalized_avg_rating'].values[0]
        
        # ADVANCED HYBRID FORMULA: Weighting all three!
        final_score = (0.4 * cb_score) + (0.4 * cf_score) + (0.2 * global_avg_score)
        
        predictions.append({
            'Smartphone': item,
            'Final Score': round(final_score, 2),
            'CB Score': round(cb_score, 2),
            'CF Score': round(cf_score, 2),
            'Global Rating': round(global_avg_score, 2)
        })
        
    # Sort and return as DataFrame
    df_preds = pd.DataFrame(predictions)
    return df_preds.sort_values(by='Final Score', ascending=False).head(top_n)

# ==========================================
# STREAMLIT UI: Run the System and Display
# ==========================================
st.title("📱 Hybrid Smartphone Recommender")
st.write("This system uses Content-Based Filtering, Collaborative Filtering, and Global Popularity to recommend smartphones.")

# Let the user pick a User ID to simulate
selected_user = st.sidebar.slider("Select User ID to generate recommendations for:", min_value=1, max_value=50, value=1)

st.subheader(f"Top 5 Recommendations for User {selected_user}")

# Generate the recommendations
with st.spinner("Calculating hybrid scores..."):
    recommendations_df = get_hybrid_recommendations(user_id=selected_user, top_n=5)
    
    # Display the dataframe on the webpage!
    st.dataframe(recommendations_df, use_container_width=True)

# --- Optional: Show Evaluation Metrics ---
st.subheader("Model Evaluation Metrics")
if st.button("Run Evaluation (Takes a few seconds)"):
    with st.spinner("Running evaluation..."):
        # We need to modify your evaluate_model slightly to return values instead of printing them
        # Let's do a quick inline evaluation display:
        
        actuals, preds_cb, preds_cf, preds_hybrid = [], [], [], []
        user_relevant_items = defaultdict(list)
        user_recommended_items = defaultdict(list)

        for _, row in df_ratings.iterrows():
            u = row['user_id']
            i = row['model']
            actual_rating = row['rating']
            
            cb = get_content_prediction(u, i, df_ratings, content_sim_df)
            cf = get_collab_prediction(u, i, user_item_matrix, user_sim_df)
            global_avg = df_subset.loc[df_subset['model'] == i, 'normalized_avg_rating'].values[0] if i in df_subset['model'].values else 0
            
            hybrid = (0.4 * cb) + (0.4 * cf) + (0.2 * global_avg)
            
            actuals.append(actual_rating)
            preds_cb.append(cb)
            preds_cf.append(cf)
            preds_hybrid.append(hybrid)
            
            if actual_rating >= 4.0:
                user_relevant_items[u].append(i)
            user_recommended_items[u].append((i, hybrid))

        rmse_cb = np.sqrt(mean_squared_error(actuals, preds_cb))
        rmse_cf = np.sqrt(mean_squared_error(actuals, preds_cf))
        rmse_hybrid = np.sqrt(mean_squared_error(actuals, preds_hybrid))
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Content-Based RMSE", f"{rmse_cb:.4f}")
        col2.metric("Collaborative RMSE", f"{rmse_cf:.4f}")
        col3.metric("Hybrid RMSE", f"{rmse_hybrid:.4f}")