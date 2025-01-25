import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import umap.umap_ as umap
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Define the image path
image_path = 'value.jpg'

# Function to import and preprocess data


@st.cache_data
def import_data(data):
    df = pd.read_fwf(data)
    df.drop(index=df.index[0:1], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['SEANCE'] = pd.to_datetime(
        df['SEANCE'], format='%d/%m/%Y', errors='coerce')
    df.set_index('SEANCE', inplace=True)
    df['CODE_INDICE'] = pd.to_numeric(df['CODE_INDICE'])
    df['INDICE_JOUR'] = pd.to_numeric(df['INDICE_JOUR'])
    df['INDICE_VEILLE'] = pd.to_numeric(df['INDICE_VEILLE'])
    df['VARIATION_VEILLE'] = pd.to_numeric(df['VARIATION_VEILLE'])
    df['INDICE_PLUS_BAS'] = pd.to_numeric(df['INDICE_PLUS_BAS'])
    df['INDICE_PLUS_HAUT'] = pd.to_numeric(df['INDICE_PLUS_HAUT'])
    df['INDICE_OUV'] = pd.to_numeric(df['INDICE_OUV'])
    return df

# Function to visualize data columns


def visualize(column, data):
    max_value = data[column].max().round(3)
    min_value = data[column].min().round(3)
    avg_value = data[column].mean().round(3)
    col1, col2, col3 = st.columns(3)
    if column == "VARIATION_VEILLE":
        with col1:
            st.write("Highest Value:", max_value, "%")
        with col2:
            st.write("Lowest Value:", min_value, "%")
        with col3:
            st.write("Average Value:", avg_value, "%")
    else:
        with col1:
            st.write("Highest Value:", max_value)
        with col2:
            st.write("Lowest Value:", min_value)
        with col3:
            st.write("Average Value:", avg_value)
    fig = px.line(data, x=data.index, y=column,
                  title="{} over Time".format(column))
    fig.update_layout(height=600, width=1800)
    st.info("This plot is interactive. You can hover over data points for details.")
    st.plotly_chart(fig, use_container_width=True)

# Function to engineer features


def engineer_features(df):
    df['Percentage_Change'] = (
        df['INDICE_JOUR'] - df['INDICE_VEILLE']) / df['INDICE_VEILLE'] * 100
    df['Volatility'] = df['INDICE_PLUS_HAUT'] - df['INDICE_PLUS_BAS']
    df['Gap'] = df['INDICE_OUV'] - df['INDICE_VEILLE']
    df['Gap_Up'] = df['Gap'] > 0
    df['Gap_Down'] = df['Gap'] < 0
    df['MA_5'] = df['INDICE_JOUR'].rolling(window=5).mean()
    df['MA_10'] = df['INDICE_JOUR'].rolling(window=10).mean()

    def calculate_rsi(data, window=14):
        delta = data['INDICE_JOUR'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    df['RSI'] = calculate_rsi(df)

    def calculate_bollinger_bands(data, window=20):
        rolling_mean = data['INDICE_JOUR'].rolling(window=window).mean()
        rolling_std = data['INDICE_JOUR'].rolling(window=window).std()
        upper_band = rolling_mean + 2 * rolling_std
        lower_band = rolling_mean - 2 * rolling_std
        return upper_band, lower_band

    upper_band, lower_band = calculate_bollinger_bands(df)
    df['Upper_BB'] = upper_band
    df['Lower_BB'] = lower_band

    return df

# Function to impute missing values


def impute_missing_values(df):
    for col in df.columns:
        if col in df.select_dtypes('number'):
            df[col] = df[col].fillna(df[col].median())
        elif col in df.select_dtypes('object'):
            df[col] = df[col].fillna(df[col].mode()[0])

# Function to get risk stats


def get_risk_stats(df, risk):
    max_value = df[risk].max().round(3)
    min_value = df[risk].min().round(3)
    avg_value = df[risk].mean().round(3)
    return max_value, min_value, avg_value

# Function to create a risk plot


def create_risk_plot(df, risk):
    fig = go.Figure()
    fig.update_layout(height=600, width=1800, title='{} Over Time'.format(risk),
                      xaxis_title='Date', yaxis_title=risk,
                      legend_title='Index Labels', showlegend=True,
                      template='plotly_white')

    unique_index_codes = df['CODE_INDICE'].unique()
    index_labels = df.groupby('CODE_INDICE')[
        'LIB_INDICE'].first().loc[unique_index_codes]

    for index_code in unique_index_codes:
        index_data = df[df['CODE_INDICE'] == index_code]
        fig.add_trace(go.Scatter(x=index_data.index, y=index_data[risk], mode='lines',
                                 name=index_labels[index_code], visible=True))

    fig.update_layout(legend=dict(itemclick='toggleothers'))
    return fig

# Function to visualize risk


def visualize_risk(df, risk):
    try:
        st.info("This plot is interactive. You can hover over data points for details.")
        max_value, min_value, avg_value = get_risk_stats(df, risk)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("Highest Value:", max_value)
        with col2:
            st.write("Lowest Value:", min_value)
        with col3:
            st.write("Average Value:", avg_value)
        fig = create_risk_plot(df, risk)

        avg_volatility_by_index = df.groupby('LIB_INDICE')['Volatility'].mean()
        high_risk_threshold = avg_volatility_by_index.quantile(0.7)
        low_risk_threshold = avg_volatility_by_index.quantile(0.3)
        high_risk_indexes = avg_volatility_by_index[avg_volatility_by_index >
                                                    high_risk_threshold]
        low_risk_indexes = avg_volatility_by_index[avg_volatility_by_index <
                                                   low_risk_threshold]
        recommended_indexes = avg_volatility_by_index[(avg_volatility_by_index >= low_risk_threshold) &
                                                      (avg_volatility_by_index <= high_risk_threshold)]

        def update_values(chart, df):
            nonlocal max_value, min_value, avg_value
            if chart['relayoutData'] and 'xaxis.range[0]' in chart['relayoutData']:
                selected_indices = df.index.get_loc(
                    chart['relayoutData']['xaxis.range[0]'])
                selected_data = df.iloc[selected_indices]
                max_value, min_value, avg_value = get_risk_stats(
                    selected_data, risk)

        st.plotly_chart(fig, use_container_width=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            st.info("High Risk Indexes")
            if not high_risk_indexes.empty:
                for el in high_risk_indexes.index:
                    col1.code(el)
            else:
                col1.code("No High Risk Indexes")

        with col2:
            st.info("Low Risk Indexes")
            if not low_risk_indexes.empty:
                for el in low_risk_indexes.index:
                    col2.code(el)
            else:
                col2.code("No Low Risk Indexes")

        with col3:
            st.info("Recommended Indexes for Investment")
            if not recommended_indexes.empty:
                for el in recommended_indexes.index:
                    col3.code(el)
            else:
                col3.code("No Recommended Indexes")
        st.warning(
            "Please note that the low, high, and recommended indexes are automatically determined based on volatility measures.")

    except Exception as e:
        st.error(f"An error occurred: {e}")

# Function to get data stats


def get_data_stats(data: pd.DataFrame, column: str) -> tuple:
    max_value = data[column].max().round(3)
    min_value = data[column].min().round(3)
    avg_value = data[column].mean().round(3)
    return max_value, min_value, avg_value

# Function to create a plot


def create_plot(data: pd.DataFrame, column: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(height=600, width=1800, title=f'{column} Over Time',
                      xaxis_title='Date', yaxis_title=column,
                      legend_title='Index Labels', showlegend=True,
                      template='plotly_white')

    unique_index_codes = data['CODE_INDICE'].unique()
    index_labels = data.groupby('CODE_INDICE')[
        'LIB_INDICE'].first().loc[unique_index_codes]

    for i, index_code in enumerate(unique_index_codes):
        index_data = data[data['CODE_INDICE'] == index_code]
        grouped_data = index_data.groupby(index_data.index)[column].mean()
        fig.add_trace(go.Scatter(x=grouped_data.index, y=grouped_data, mode='lines',
                                 name=index_labels[index_code], visible=True))

    fig.update_layout(legend=dict(itemclick='toggleothers'))
    return fig

# Function to plot numeric column over time per index


def plot_numeric_column_over_time_per_index(data: pd.DataFrame, column: str) -> None:
    st.info("Click on a legend item to visualize the corresponding index.")
    max_value, min_value, avg_value = get_data_stats(data, column)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"Highest Value: ", max_value)
    with col2:
        st.write(f"Lowest Value: ", min_value)
    with col3:
        st.write(f"Average Value: ", avg_value)

    try:
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index, errors='coerce')
            data = data[data.index.notnull()]

        fig = create_plot(data, column)

        def update_values(chart, data):
            nonlocal max_value, min_value, avg_value
            if chart['relayoutData'] and 'xaxis.range[0]' in chart['relayoutData']:
                selected_indices = data.index.get_loc(
                    chart['relayoutData']['xaxis.range[0]'])
                selected_data = data.iloc[selected_indices]
                max_value, min_value, avg_value = get_data_stats(
                    selected_data, column)

        st.plotly_chart(fig, use_container_width=True)

    except ValueError as e:
        st.error(f"An error occurred: {e}")

# Function to visualize clusters


def visualize_cluster(df, model):
    cols = ['INDICE_JOUR', 'Percentage_Change', 'Volatility',
            'Gap', 'MA_5', 'MA_10', 'RSI', 'Upper_BB', 'Lower_BB']
    X = df[cols]
    silhouette_scores = []
    for n_clusters in range(2, 6):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X)
        silhouette_scores.append(silhouette_score(X, labels))
    optimal_num_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
    model.n_clusters = optimal_num_clusters
    labels = model.fit_predict(X)
    reducer = umap.UMAP(n_components=2)
    umap_result = reducer.fit_transform(X)
    umap_df = pd.DataFrame(umap_result, columns=['UMAP1', 'UMAP2'])
    umap_df['Cluster'] = labels

    scatter_plot_umap = go.Scatter(
        x=umap_df['UMAP1'],
        y=umap_df['UMAP2'],
        mode='markers+text',
        marker=dict(
            color=umap_df['Cluster'],
            colorscale='plasma',
            size=8,
            line=dict(
                color='rgba(217, 217, 217, 0.14)',
                width=0.5
            ),
            opacity=0.8,
            colorbar=dict(title='Cluster'),
        ),
        text=df['LIB_INDICE'],
        hoverinfo='text',
        textposition='top right',
        textfont=dict(size=10),
        showlegend=False
    )

    layout_umap = go.Layout(
        xaxis=dict(title='UMAP1'),
        yaxis=dict(title='UMAP2'),
        margin=dict(l=50, r=50, b=50, t=50),
        width=1800,
        height=600
    )

    fig_umap = go.Figure(data=[scatter_plot_umap], layout=layout_umap)
    st.plotly_chart(fig_umap, use_container_width=True)


# Define the dataset dictionary
dataset_dict = {}
for year in range(2011, 2022):
    dataset_name = f"Data/histo_indice_{year}.txt"
    dataset_dict[year] = dataset_name

# Sidebar for navigation and information
with st.sidebar:
    st.markdown("<center><h1> <b style='color:orange;'>V</b>ALUE</h1></center>",
                unsafe_allow_html=True)
    choice = st.radio("Navigation", ["Historical Indexes", "Ja"])
    st.info("This application helps you optimize your portfolio!")

# Main content based on user choice
if choice == "Historical Indexes":
    st.markdown("<center><h1 style='color:orange;font-weight:bolder'>Data Analysis</h1></center>",
                unsafe_allow_html=True)
    st.info("A brief description about the data.")
    with st.expander("Variables Description"):
        st.markdown("""
        <hr>
        <li>INDICE_JOUR: Index value at the end of the trading day.</li>
        <hr>
        <li>INDICE_VEILLE: Index value from the previous trading day.</li>
        <hr>
        <li>VARIATION_VEILLE: Change or variation in the index value from the previous trading day.</li>
        <hr>
        <li>INDICE_PLUS_HAUT: Highest value the index reached during the trading session.</li>
        <hr>
        <li>INDICE_PLUS_BAS: Lowest value the index reached during the trading session.</li>
        <hr>
        <li>INDICE_OUV: Opening value of the index at the beginning of the trading session.</li>
        <hr>""", unsafe_allow_html=True)
    st.markdown("### Choose Year")
    chosen_year = st.selectbox("Select Year", list(dataset_dict.keys()))
    st.markdown("### Choose Data Column for Historical Trend Visualization")
    chosen_dataset = dataset_dict[chosen_year]
    df = import_data(chosen_dataset)
    cols = df.select_dtypes('number').columns.drop('CODE_INDICE')
    selected_column = st.selectbox("select column", cols)
    plot_numeric_column_over_time_per_index(df, selected_column)
    st.markdown("<center><h1 style='color:orange;font-weight:bolder;'>Risk Analysis</h1></center>",
                unsafe_allow_html=True)
    risks = ["RSI", "Volatility"]
    chosen_risk = st.selectbox("Select Indicator", risks)
    df = engineer_features(df)
    impute_missing_values(df)
    visualize_risk(df, chosen_risk)
    st.markdown("<center><h1 style='color:orange;font-weight:bolder;'>Explore Indexes Similarities</h1></center>",
                unsafe_allow_html=True)
    model_2011 = joblib.load("Data/kmeans_2011.pkl")
    cols = ['INDICE_JOUR', 'Percentage_Change', 'Volatility',
            'Gap', 'MA_5', 'MA_10', 'RSI', 'Upper_BB', 'Lower_BB']
    df_avg = df.groupby('LIB_INDICE').mean().reset_index()
    X = df_avg[cols]
    visualize_cluster(df_avg, model_2011)
