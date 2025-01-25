# Intelligent Trading Agent: Predictive Analytics, Risk Management & Sentiment Analysis for Trading


## Overview
This repository contains the source code and documentation for the Intelligent Trading Agent, a capstone project designed to integrate Natural Language Processing (NLP), , and financial PDF summarization,  web scraping, predictive analytics and Risk prediction to revolutionize decision-making in the Tunisian stock market. 

The platform achieves:
- Real-time financial news sentiment analysis.
- Forecasting market indexes and stock prices.
- Risk assessment and analysis for portfolio optimization.
- Summarization of financial PDFs with specialized NLP models.
- Real-time data visualization for market movements and trends.

## Key Features

### 1. Market Sentiment Analysis
- **News Scraping:** Financial articles scraped from sources like BVMT, Tunisienumerique and Agence Tunis Afrique News using Selenium.
- **Sentiment Analysis:** Leveraging fine-tuned models (e.g., FinBERT) for extracting sentiment from financial news articles.
- **Named Entity Recognition (NER):** Extracting key entities like company names using Custom SpaCy NER.

### 2. Financial Document Summarization
- **PDF Processing:** Parsing financial PDFs from sources like cmf.tn and BVMT using tools such as Tabula, PDFPlumber, and OCR techniques like PubLayNet and Pytesseract.
- **Summarization:** Utilizing transformers-based NLP models to provide concise insights from income statements, balance sheets, and cash flow statements.

### 3. Predictive Analytics
- **Market Index and Stock Forecasting:** Utilizing models like Prophet, LSTM, and XGBoost for predicting market index trends and stock prices.
- **Risk Analysis:** Employing clustering and dimensionality reduction techniques to identify risk clusters and assess volatility.
- **Portfolio Optimization:** Analyzing risk-return trade-offs using metrics such as Sharpe and Sortino ratios.

### 4. Real-Time Visualization
- **Dashboards:** Interactive Streamlit dashboards displaying:
  - Market sentiment trends.
  - Forecasted stock prices and indexes.
  - Risk indicators and volatility trends.
- **Custom Visualizations:** Created using Plotly, Matplotlib, and Seaborn.

## Technical Stack

| Component                     | Technology                                      |
|-------------------------------|------------------------------------------------|
| **Web Scraping**              | Selenium, BeautifulSoup                        |
| **Database**                  | MongoDB Atlas                                  |
| **NLP Framework**             | Hugging Face Transformers, SpaCy               |
| **Predictive Models**         | Prophet, LSTM, XGBoost                         |
| **Dashboards**                | Streamlit, Plotly                              |
| **Data Processing**           | Pandas, NumPy                                  |
| **Deployment**                | Docker, Scheduler                              |
| **Financial PDF Chat**        | LangChain, Ollama (Llama2)

## Architecture

1. **Data Collection:**
   - Real-time scraping of financial data, news, and PDFs.
   - Storage in MongoDB Atlas for structured and unstructured data.

2. **Processing:**
   - Preprocessing of financial news and reports.
   - Feature engineering for time series data.

3. **Analysis and Prediction:**
   - Sentiment analysis and named entity extraction.
   - Time series forecasting for stock prices and indexes.
   - Clustering for risk segmentation.

4. **Visualization:**
   - Dashboards for real-time analytics.
   - Interactive user interfaces for decision-making.

## Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/username/intelligent-trading-agent.git](https://github.com/ChaymaRhaiem/FinSent-AI-Driven-Financial-Sentiment-Market-Prediction.git
   cd intelligent-trading-agent
   ```

2. Set up the environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows, use `env\Scripts\activate`
   pip install -r requirements.txt
   ```

3. Configure environment variables in a `.env` file:
   ```env
   MONGODB_URI=your_mongo_uri
   API_KEY=your_api_key
   ```

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Usage

### Financial News Sentiment Analysis
- Navigate to the "Market Sentiment" tab.
- View sentiment scores and trends for specific entities or markets.

### Predictive Analytics
- Access the "Forecasting" tab.
- View time series predictions for stock prices and indexes.

### Risk Analysis
- Explore the "Risk Assessment" section for clustering and volatility trends.

### Financial Document Summarization
- Upload PDFs in the "Document Summarization" tab.
- View extracted summaries and insights.

## Data Sources
- Tunisian Stock Market (BVMT)
- Financial Market Council (cmf.tn)
- Agence Tunis Afrique News
- Tunisienumerique

## Key Metrics
- **Performance:** Sharpe and Sortino ratios, ROI, annualized return.
- **Risk Indicators:** Volatility, skewness, kurtosis.
- **Forecast Accuracy:** Model evaluation metrics like MAE, RMSE.


## Contributors
- **Chayma Rhaiem**
- **Zeineb Benfredj**
- **Aziz Ben Romdhan**
- Yassine Traidi
- Ghofrane Ben Rhaiem
- Mohamed Jasser Chtourou


## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
Special thanks to Value Digital Services for their support and mentorship throughout this project.

---

For more information, please refer to the full project report or feel free to contact me via email.
