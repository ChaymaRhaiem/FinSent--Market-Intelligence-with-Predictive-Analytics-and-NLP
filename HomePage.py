import subprocess
import streamlit as st
from streamlit_navigation_bar import st_navbar
from PIL import Image
import pandas as pd
import importlib.util
import requests
import time
import plotly.graph_objects as go
from datetime import datetime

# Initialize session state variables
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = 'Guest'
if 'user_image' not in st.session_state:
    st.session_state['user_image'] = None

# Sidebar for navigation
pages = ["Home Page", "Dividend", "Cotation", "Market Data Index",
         "Risk Management", "Market Sentiment", "Consult Chat"]
page = st.sidebar.selectbox("Navigation", pages)
styles = {
    "nav": {"background-color": "rgb(255, 255, 255)", "color": "rgb(49, 51, 63)"},
    "div": {"max-width": "70rem"},
    "span": {"border-radius": "0.5rem", "color": "rgb(49, 51, 63)", "margin": "0 0.5rem", "padding": "0.4375rem 0.875rem"},
    "active": {"background-color": "rgba(255, 255, 255, 0.25)"},
    "hover": {"background-color": "rgba(255, 255, 255, 0.35)"}
}
page = st_navbar(pages, styles=styles)

# Sidebar welcome and logout button
username = st.session_state.get('username', 'Guest')
user_image_path = st.session_state.get('user_image', None)
with st.sidebar:
    st.sidebar.header("User Options")
    if user_image_path:
        st.sidebar.image(user_image_path, caption='Profile Image', width=100)
    st.write(f"Welcome, {username}")
    if st.button("Logout", key='sidebar_logout'):
        st.session_state['logged_in'] = False
        st.session_state['username'] = None
        st.session_state['user_image'] = None
        st.experimental_rerun()

# Additional sidebar info
st.sidebar.info("Use the sidebar to navigate through different options. You can open and close the sidebar by clicking the > arrow at the top-left corner.")

# Custom CSS to enhance sidebar styling
st.sidebar.markdown("""
<style>
    .css-1d391kg {
        background-color: #262730;  /* Secondary background color */
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .stButton>button {
        width: 100%;
        color: #FAFAFA; /* Text color */
        border: 1px solid #FF4B4B; /* Primary color */
        background-color: #0E1117; /* Background color */
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def get_stock_data(ticker, start_date, end_date, api_key):
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}?apiKey={api_key}"
    response = requests.get(url)
    data = response.json()
    if 'results' in data:
        return pd.DataFrame(data['results'])
    else:
        st.error(f"API Error: {data.get('error', 'Unknown error')}")
        return pd.DataFrame()


df = pd.read_csv('Data\\output.csv')
df.columns = df.columns.str.strip().str.replace(' ', '_')

if page == "Home Page":
    subprocess.Popen(["python", "Scrape.py", "--server.port", "8507"])
    header_cols = st.columns([4, 1])
    with header_cols[1]:
        img = Image.open("Data\\ValueLogo.png")
        st.image(img, width=150)
    max_variation = df.loc[df['Variation'].idxmax()]
    max_haut = df.loc[df['+Haut'].idxmax()]
    max_bas = df.loc[df['+Bas'].idxmax()]
    cols = st.columns(3)
    with cols[0]:
        st.metric(label="Highest Variation", value=max_variation['Nom'], delta=str(
            max_variation['Variation']))
    with cols[1]:
        st.metric(label="Highest +Haut",
                  value=max_haut['Nom'], delta=str(max_haut['+Haut']))
    with cols[2]:
        st.metric(label="Highest +Bas",
                  value=max_bas['Nom'], delta=str(max_bas['+Bas']))

    # Real-time stock data plot
    api_key = "_nOZIY1gWc9IOVVMlq0J0omrXxTl6Uap"
    default_ticker = "GOOGL"
    default_start_date = datetime(2015, 1, 1)
    default_end_date = datetime.today()

    ticker = st.text_input(
        'Enter the stock ticker, e.g., "AAPL":', value=default_ticker)
    start_date = st.date_input(
        'Select the start date:', value=default_start_date)
    end_date = st.date_input('Select the end date:', value=default_end_date)
    if ticker and start_date and end_date:
        df = get_stock_data(ticker, start_date, end_date, api_key)
        if not df.empty:
            fig = go.Figure(data=[go.Candlestick(x=df['t'],
                                                 open=df['o'],
                                                 high=df['h'],
                                                 low=df['l'],
                                                 close=df['c'])])
            fig.update_layout(
                title=f'Real-time Data for {ticker}', xaxis_title='Time', yaxis_title='Price')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Failed to fetch data. Please check the ticker or API key.")
        time.sleep(300)  # Updates every 5 minutes

    # Display the SDG cards at the bottom

elif page == "Dividend":
    # Dynamically load Dividend.py
    spec = importlib.util.spec_from_file_location(
        "Dividend", "./pages_pi/Dividend.py")
    dividend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dividend)

elif page == "Market Sentiment":
    if 'market_sentiment_session_started' not in st.session_state:
        st.session_state['market_sentiment_session_started'] = True
        subprocess.Popen(
            ["streamlit", "run", "./scraping/data/streamlit_app.py", "--server.port", "8520"])
    st.subheader("Market Sentiment Analysis")
    st.markdown("""
        <iframe src="http://localhost:8520" width="110%" height="1500" frameborder="0"></iframe>
    """, unsafe_allow_html=True)

elif page == "Cotation":
    # Dynamically load Cotation.py
    spec = importlib.util.spec_from_file_location(
        "Cotation", "./pages_pi/Cotation.py")
    cotation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cotation)

elif page == "Risk Management":
    risk_options = ["Risk Segmentation", "Price by Date", "Risk Prediction"]
    if 'selected_risk_option' not in st.session_state:
        st.session_state.selected_risk_option = "Risk Segmentation"
    st.session_state.selected_risk_option = st.sidebar.selectbox(
        "Select Risk Management Option", risk_options, index=risk_options.index(st.session_state.selected_risk_option))

    if st.session_state.selected_risk_option == "Risk Segmentation":
        spec = importlib.util.spec_from_file_location(
            "Risk Segmentation", "./clustering.py")
        RiskSegmentation = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(RiskSegmentation)
    elif st.session_state.selected_risk_option == "Price by Date":
        spec = importlib.util.spec_from_file_location(
            "Price by Date", "./app.py")
        Pricebydate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(Pricebydate)
    elif st.session_state.selected_risk_option == "Risk Prediction":
        spec = importlib.util.spec_from_file_location(
            "Risk Prediction", "./risk.py")
        risk = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(risk)

elif page == "Market Data Index":
    if 'show_options' not in st.session_state:
        st.session_state.show_options = False
    if 'selected_option' not in st.session_state:
        st.session_state.selected_option = None

    with st.sidebar:
        st.write("Choose an Option")
        index_options = ["Market Data Index Forecasting",
                         "Market Data Index Analysis"]
        st.session_state.selected_option = st.selectbox(
            "Select Market Data Index Option", index_options)

    if st.session_state.selected_option == "Market Data Index Forecasting":
        spec = importlib.util.spec_from_file_location(
            "Market Data Index Forecasting", "./pages_pi/MarketDataIndex.py")
        MarketDataIndex = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(MarketDataIndex)
    elif st.session_state.selected_option == "Market Data Index Analysis":
        spec = importlib.util.spec_from_file_location(
            "MarketDataIndexAnalysis", "./pages_pi/MarketDataIndexAnalysis.py")
        MarketDataIndexAnalysis = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(MarketDataIndexAnalysis)
    else:
        st.write(
            "Please select an option from the sidebar to view the respective Market Data Index functionality.")

elif page == "Consult Chat":
    # Dynamically load Chat.py
    spec = importlib.util.spec_from_file_location(
        "Consult Chat", "./pages_pi/Chat.py")
    ConsultChat = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ConsultChat)
