#######################
# Import libraries
import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go
import warnings
warnings.filterwarnings('ignore')


#######################
# CSS styling
st.markdown("""
<style>
    .css-1d391kg { background-color: #ffffff; color: #000000; }
    .css-1aumxhk { background-color: #ffffff; }
    .st-bx { border-color: #ffffff; }
    .st-bs { background-color: #000000; }
    .stMetric {
        border: 1px solid #FFFFFF !important;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

#######################
# Data loading and processing functions


@st.cache_data
def import_data_1(data):
    df = pd.read_csv(data, sep=',')
    df.drop(index=df.index[0:1], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    if df['Date'].isnull().values.any():
        st.write("Warning: Some dates are invalid.")
    df.set_index('Date', inplace=True)
    return feature_engineering(df)


def feature_engineering(data):
    data['HighLowSpread'] = data['High'] - data['Low']
    data['HighLowRangePercentage'] = (
        data['High'] - data['Low']) / data['Open']
    data['Growth_Rate'] = ((data['Close'] - data['Open']) / data['Open']) * 100
    data['RSI'] = calculate_rsi(data)
    data['Volatility'] = calculate_volatility(data)
    return data


def calculate_rsi(df, window=14):
    delta = df['Close'].diff()
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    avg_gain = gains.ewm(span=window, min_periods=window).mean()
    avg_loss = losses.ewm(span=window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_volatility(df, window=20):
    price_range = df['High'] - df['Low']
    smoothed_range = price_range.rolling(window=window).mean()
    volatility = smoothed_range.rolling(window=window).std()
    return volatility


def visualize(df, column):
    fig = px.line(df, x=df.index, y=column, title=f"{column} over time")
    fig.update_layout(width=None, xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)


#######################
# Sidebar and main page layout
with st.sidebar:
    choice = st.selectbox('Select an option:', options=[
                          "Data Analysis & Visualization", "FORECASTING"])

Indexes_list = ['TUNALIM', 'TUNASS', 'TUNBANK', 'TUNBASE', 'TUNBATIM', 'TUNCONS',
                'TUNDIS', 'TUNFIN', 'TUNINDEX', 'TUNINDEX20', 'TUNINPMP', 'TUNSAC', 'TUNSEFI']
index_dict = {i: f"Data/{i}_pi_ds_esprit.csv" for i in Indexes_list}

chosen_idx = st.selectbox("Choose Index", list(index_dict.keys()))
chosen_dataset = index_dict[chosen_idx]
df = import_data_1(chosen_dataset)

#######################
# Data Analysis & Visualization
if choice == "Data Analysis & Visualization":
    st.title("Data Analysis & Visualization")

    st.header("Index Performance over Time:")
    chosen_col = st.selectbox(
        "Choose a column", df.columns.drop(['RSI', 'Volatility']))
    visualize(df, chosen_col)

    def plot_raw_data(df):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Open'], name='Stock Open', line=dict(color='green')))
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Close'], name='Stock Close', line=dict(color='red')))
        fig.layout.update(
            title_text="Time Series Data of Open and Close", xaxis_rangeslider_visible=True, width=1600)
        st.plotly_chart(fig, use_container_width=True)

    plot_raw_data(df)

    st.header("Risk Visualizations:")
    chosen_risk = st.selectbox("Choose a feature", ['RSI', 'Volatility'])
    visualize(df, chosen_risk)

#######################
# Forecasting Analysis
if choice == "FORECASTING":
    st.title("Forecasting Analysis:")
    n_years = st.slider("Years of Prediction:", 1, 10)
    period = n_years * 365
    df_train = df.reset_index()[['Date', 'Close']].rename(
        columns={"Date": "ds", "Close": "y"})
    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    st.subheader('Forecasting Closing Price:')
    fig_forecast = plot_plotly(m, forecast)
    fig_forecast.update_layout(width=None)
    st.plotly_chart(fig_forecast, use_container_width=True)

    # Additional forecasting blocks can be added similarly

    # Forecasting on Volatility
    st.title("Forecasting on Volatility:")
    df_train_vol = df.reset_index()[['Date', 'Volatility']].rename(
        columns={"Date": "ds", "Volatility": "y"})
    m_vol = Prophet()
    m_vol.fit(df_train_vol)
    future_vol = m_vol.make_future_dataframe(periods=period)
    forecast_vol = m_vol.predict(future_vol)

    st.subheader('Forecasted Volatility:')
    fig_volatility = plot_plotly(m_vol, forecast_vol)
    fig_volatility.update_layout(width=None)
    st.plotly_chart(fig_volatility, use_container_width=True)

    # Plotting Volatility components with Plotly
    components_vol = m_vol.make_future_dataframe(
        periods=period, include_history=True)
    forecast_components_vol = m_vol.predict(components_vol)
    fig_components_vol = go.Figure()
    fig_components_vol.add_trace(go.Scatter(
        x=forecast_components_vol['ds'], y=forecast_components_vol['trend'], name='Trend'))
    fig_components_vol.add_trace(go.Scatter(
        x=forecast_components_vol['ds'], y=forecast_components_vol['yearly'], name='Yearly'))
    fig_components_vol.update_layout(
        title="Forecast on Volatility Components", width=None)
    st.plotly_chart(fig_components_vol, use_container_width=True)

    # Forecasting on RSI
    st.title("Forecasting on RSI:")
    df_train_RSI = df.reset_index()[['Date', 'RSI']].rename(
        columns={"Date": "ds", "RSI": "y"})
    m_RSI = Prophet()
    m_RSI.fit(df_train_RSI)
    future_RSI = m_RSI.make_future_dataframe(periods=period)
    forecast_RSI = m_RSI.predict(future_RSI)

    st.subheader('Forecasted RSI:')
    fig_RSI = plot_plotly(m_RSI, forecast_RSI)
    fig_RSI.update_layout(width=None)
    st.plotly_chart(fig_RSI, use_container_width=True)

    # Plotting RSI components with Plotly
    components_RSI = m_RSI.make_future_dataframe(
        periods=period, include_history=True)
    forecast_components_RSI = m_RSI.predict(components_RSI)
    fig_components_RSI = go.Figure()
    fig_components_RSI.add_trace(go.Scatter(
        x=forecast_components_RSI['ds'], y=forecast_components_RSI['trend'], name='Trend'))
    fig_components_RSI.add_trace(go.Scatter(
        x=forecast_components_RSI['ds'], y=forecast_components_RSI['yearly'], name='Yearly'))
    fig_components_RSI.update_layout(
        title="Forecast on RSI Components", width=None)
    st.plotly_chart(fig_components_RSI, use_container_width=True)
