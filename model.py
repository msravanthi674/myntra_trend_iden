import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from collections import Counter
import plotly.express as px

# Load the data
@st.cache
def load_data():
    return pd.read_csv('navigator-batch-generate-66911bf981ad034c3ff10509-data.csv')

data = load_data()

# Prepare the features and target variables
features = ['gender', 'pattern', 'color', 'age_group', 'season', 'price', 'sales_count', 'reviews_count', 'average_rating', 'out_of_stock_times', 'discount', 'wish_list_count', 'month_of_sale', 'year_of_sale']
targets = ['category', 'material']

# Filter out rare classes in target variables
for target in targets:
    class_counts = Counter(data[target])
    classes_to_keep = [cls for cls, count in class_counts.items() if count > 1]
    data = data[data[target].isin(classes_to_keep)]

# Encode categorical variables
le_dict = {}
for column in data.columns:
    if data[column].dtype == 'object':
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        le_dict[column] = le

# Prepare features and target data
X = data[features]
y = data[targets]

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the multi-output model
xgb_model = XGBClassifier(random_state=42)
multi_target_xgb = MultiOutputClassifier(xgb_model)
multi_target_xgb.fit(X_train, y_train)

# Function to get top 5 predictions
def get_top_5_predictions(model, X):
    probabilities = model.predict_proba(X)
    top_5_product_indices = np.argsort(probabilities[0][0])[-5:][::-1]
    top_5_material_indices = np.argsort(probabilities[1][0])[-5:][::-1]
    
    top_5_product_names = le_dict['category'].inverse_transform(top_5_product_indices)
    top_5_materials = le_dict['material'].inverse_transform(top_5_material_indices)
    
    return top_5_product_names, top_5_materials

# Define Streamlit app
def main():
    st.title('Fashion Product Prediction App')

    # Sidebar with options
    st.sidebar.title('Options')
    show_data = st.sidebar.checkbox('Show Raw Data', False)

    if show_data:
        st.subheader('Raw Data')
        st.write(data)

    # st.subheader('Model Accuracy')

    # # Calculate accuracy for each target
    # for i, target in enumerate(targets):
    #     accuracy = accuracy_score(y_test.iloc[:, i], multi_target_xgb.predict(X_test)[:, i])
    #     st.write(f"Accuracy for {target}: {accuracy:.2f}")

    # Get top 5 predictions for a sample input
    st.subheader('Top 5 Predicted Product Names and Materials')
    sample_input = X_test.iloc[0].to_frame().T
    top_5_product_names, top_5_materials = get_top_5_predictions(multi_target_xgb, sample_input)

    for i, (product_name, material) in enumerate(zip(top_5_product_names, top_5_materials), 1):
        st.write(f"{i}. Product Name: {product_name}, Material: {material}")

    # Visualize top predicted product categories and materials
    st.subheader('Top Predicted Product Categories and Materials')

    fig = px.bar(x=top_5_product_names, y=np.arange(1, 6), orientation='h', 
                 labels={'x': 'Product Category', 'y': 'Rank'},
                 title='Top Predicted Product Categories')
    st.plotly_chart(fig)

    fig = px.bar(x=top_5_materials, y=np.arange(1, 6), orientation='h', 
                 labels={'x': 'Material', 'y': 'Rank'},
                 title='Top Predicted Materials')
    st.plotly_chart(fig)

if __name__ == '__main__':
    main()
