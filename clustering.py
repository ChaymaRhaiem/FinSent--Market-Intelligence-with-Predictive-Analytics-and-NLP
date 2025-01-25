import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import plotly.graph_objects as go
from scipy.stats import skew, kurtosis
import umap.umap_ as umap
from pymongo import MongoClient
from db_connection.credentials import get_db

# Mapping from collection names to display names
categories_display_names = {
    "Assurance Leasing": "Insurance Leasing",
    "Banques": "Bank",
    "Companies_autre": "Other",
    "banques leasing sicav": "Bank Leasing Sicav"
}


def feature_engineering(selected_category, db):
    # Load dataset for selected category from MongoDB collection
    collection = db[selected_category]
    cursor = collection.find({})
    df = pd.DataFrame(list(cursor))

    # Data Cleaning
    df['Date'] = pd.to_datetime(df['Date'])
    df['Vol.'] = pd.to_numeric(
        df['Vol.'].str.replace('K', '').str.replace('M', ''))
    df['Change %'] = pd.to_numeric(df['Change %'].str.replace('%', ''))

    # Feature Engineering
    df['Day'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    columns_to_drop = ['Open', 'High', 'Low']
    df = df.drop(columns=columns_to_drop)

    # Calculate skewness and kurtosis for 'Price' column
    price_skewness = skew(df['Price'])
    price_kurtosis = kurtosis(df['Price'])
    df['PriceSkewness'] = price_skewness
    df['PriceKurtosis'] = price_kurtosis
    df['Risk'] = df['Price'].rolling(window=30).std()
    df['Return'] = df['Price'].pct_change()
    df['Median'] = df['Price'].rolling(window=30).median()
    df['Q1'] = df['Price'].rolling(window=30).quantile(0.25)
    df['Q3'] = df['Price'].rolling(window=30).quantile(0.75)
    df['AverageQ1'] = df['Q1'].rolling(window=30).mean()
    df['AverageQ3'] = df['Q3'].rolling(window=30).mean()

    # Fill missing values in 'Vol.' column with the mean
    df['Vol.'].fillna(df['Vol.'].mean(), inplace=True)
    numeric_columns = ['Risk', 'Return', 'Median',
                       'Q1', 'Q3', 'AverageQ1', 'AverageQ3']
    df[numeric_columns] = df[numeric_columns].fillna(
        df[numeric_columns].mean())

    return df


# Streamlit App
st.title('Market Segmentation Explorer: Unveiling Investment Opportunities')

# Connect to MongoDB
db = get_db()

# Sidebar - Select Category
selected_category = st.sidebar.selectbox(
    'Select Category', list(categories_display_names.values()))

# Get the collection name corresponding to the selected category
selected_collection = [
    k for k, v in categories_display_names.items() if v == selected_category][0]

# Perform feature engineering based on selected category
selected_category_data = feature_engineering(selected_collection, db)
# Apply UMAP
numeric_df = selected_category_data.select_dtypes(include=['float64', 'int64'])
umap_embedding = umap.UMAP().fit_transform(numeric_df)

# Apply DBSCAN clustering
# eps = st.slider("Epsilon", 0.1, 2.0, 0.5, 0.1)
# min_samples = st.slider("Minimum Samples", 1, 10, 5, 1)
dbscan = DBSCAN(eps=0.5, min_samples=5)
cluster_labels = dbscan.fit_predict(umap_embedding)

# Get unique cluster labels (excluding noise points)
unique_labels = np.unique(cluster_labels[cluster_labels != -1])

# Define cluster interpretations
cluster_interpretations = {
    0: {'risk': 'Low', 'return': 'Low', 'volatility': 'Stable'},
    1: {'risk': 'Moderate', 'return': 'Moderate', 'volatility': 'Fluctuating'},
    2: {'risk': 'High', 'return': 'High', 'volatility': 'Volatile'},
    3: {'risk': 'Very High', 'return': 'Very High', 'volatility': 'Highly Volatile'}
    # Add more interpretations if needed
}

# Create traces for each cluster (excluding noise points)
cluster_traces = []
for label in unique_labels:
    if label in cluster_interpretations:  # Ensure the label exists in cluster_interpretations
        interpretation = cluster_interpretations[label]
        trace = go.Scatter(
            x=umap_embedding[cluster_labels == label, 0],
            y=umap_embedding[cluster_labels == label, 1],
            mode='markers',
            marker=dict(
                size=8,
                opacity=0.7,
                color=label,  # Color by cluster label
                colorscale='Viridis'  # You can choose other color scales
            ),
            name=f'Cluster {label}, {interpretation["risk"]} Risk, {interpretation["return"]} Return, {interpretation["volatility"]} Volatility'
        )
        cluster_traces.append(trace)

# Combine all traces
data = cluster_traces

# Create layout
layout = go.Layout(
    title=f'Clustering for {selected_category}',
    xaxis=dict(title='UMAP Dimension 1'),
    yaxis=dict(title='UMAP Dimension 2'),
    hovermode='closest',
    showlegend=True
)

# Create figure
fig = go.Figure(data=data, layout=layout)

# Show figure
st.plotly_chart(fig, use_container_width=True)
