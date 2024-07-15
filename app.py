import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the datasets
items = pd.read_csv('items.csv')

# Create a TF-IDF vectorizer to convert features into numerical vectors
vectorizer = TfidfVectorizer()

# Fit the vectorizer to the features and transform them into vectors
features_vectors = vectorizer.fit_transform(items['item_id'].astype(str) + ' ' + items['year_of_sale'].astype(str))

# Calculate the cosine similarity between all pairs of items
similarity_matrix = cosine_similarity(features_vectors, features_vectors)

def get_content_based_recommendations(item_id, num_recommendations=5):
    # Check if the item_id exists in the DataFrame
    if item_id not in items['item_id'].values:
        return f"Item ID {item_id} does not exist in the dataset."
    
    # Get the index of the item
    idx = items[items['item_id'] == item_id].index[0]

    # Get the similarity scores for all items
    sim_scores = list(enumerate(similarity_matrix[idx]))

    # Sort the items by similarity score
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the top N recommendations
    top_n = sim_scores[1:num_recommendations+1]

    # Return the recommended items with items, category, and color
    recommended_items = items.iloc[[i[0] for i in top_n]][['items', 'category', 'color']]
    return recommended_items.reset_index(drop=True)

# Streamlit app
st.title('Content-based Recommendation System')

# Content-based recommendation
st.header('Content-based Recommendation')

# Dropdown for selecting item
item_name = st.selectbox('Select Item for Content-based Recommendation', items['items'].unique())
num_recommendations_content = st.slider('Number of Recommendations', min_value=1, max_value=10, value=5)

if st.button('Get Content-based Recommendations'):
    # Map selected item to item_id
    item_id = items[items['items'] == item_name]['item_id'].values[0]
    recommendations = get_content_based_recommendations(item_id, num_recommendations_content)
    st.write(recommendations)

if _name_ == '_main_':
    st.run()