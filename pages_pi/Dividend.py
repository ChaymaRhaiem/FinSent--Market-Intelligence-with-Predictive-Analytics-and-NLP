import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st
import pandas as pd
from joblib import load
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import streamlit_shadcn_ui as ui
import plotly.express as px
import plotly.graph_objects as go
from db_connection.credentials import get_db
# Set page configuration
# st.set_page_config(page_title="Innovest Ai Strategist", page_icon="📈", layout="wide")
import warnings
warnings.filterwarnings('ignore')

# Apply custom styles for better UI design
st.markdown("""
<style>
    .css-1d391kg { background-color: #ffffff; color: #000000; }
    .css-1aumxhk { background-color: #ffffff; }
    .st-bx { border-color: #ffffff; }
    .st-bs { background-color: #000000; }
    .stMetric {
        border: 1px solid #FFFFFF !important;  /* Set border color to white */
        border-radius: 0.5rem;  /* Adjust border radius as needed */
    }
</style>
""", unsafe_allow_html=True)

# Load data with caching
# Connect to MongoDB
db = get_db()

# Specify the collection (table) you want to use
collection_name = 'Dividend'

# Get the specified collection from the database
collection = db[collection_name]

# Convert the collection to a DataFrame
df = pd.DataFrame(list(collection.find()))


@st.cache_data  # Use caching to load the data only once
def load_modelgradient_boosting_model():
    model_path = 'Data\\gradient_boosting_model.joblib'  # Adjust the path as necessary
    return load(model_path)


modelgradient_boosting_model = load_modelgradient_boosting_model()


def load_model():
    return load('Data\\KMeans_Pipeline.joblib')  # Adjust the path as necessary


model = load_model()

# Specify the collection (table) you want to use
collection_name = 'dataPrep'

# Get the specified collection from the database
collection = db[collection_name]

# Convert the collection to a DataFrame
dataPrep = pd.DataFrame(list(collection.find()))

# Extract years from 'Montant YYYY' columns
years = sorted(int(col.split()[-1]) for col in df.columns if 'Montant' in col)

# Sidebar setup for structured controls
with st.sidebar:
    st.sidebar.header("Graphical Analysis")
    company = st.selectbox('Select a company:', df['Companies'].unique())
    plot_type = st.selectbox('Select plot type:', [
                             'Line Plot', 'Bar Plot', 'Pie Chart', 'Histogram'])

    st.sidebar.header("Companies Growth")
    start_year = st.selectbox('Select starting year:',
                              years, index=years.index(2017))
    end_year = st.selectbox('Select ending year:',
                            years, index=years.index(2022))

# Display metrics with icons
metrics_col = st.columns(3)
icons = ["📈", "🚀", "💰"]
labels = ["Best Liquidity Company",
          "Best Growth Rate Company", "Highest Dividend in 2022"]
values = [df.loc[df['Average DL'].idxmax()]['Companies'],
          df.loc[df['Growth Rate 2022'].idxmax()]['Companies'],
          df.loc[df['Montant 2022'].idxmax()]['Companies']]
deltas = [f"Average DL: {df['Average DL'].max():.2f}",
          f"Growth Rate 2022: {df['Growth Rate 2022'].max():.2%}",
          f"Montant 2022: ${df['Montant 2022'].max()}"]

for col, icon, label, value, delta in zip(metrics_col, icons, labels, values, deltas):
    with col:
        st.metric(label=f"{icon} {label}", value=value, delta=delta)

# Filtering and plotting data
dividends = [df.loc[df['Companies'] == company,
                    f'Montant {year}'].values[0] for year in years if f'Montant {year}' in df.columns]
plot_data = pd.DataFrame({'Year': years, 'Dividend': dividends}).dropna()

with st.expander("Graphical Analysis"):
    st.write("Dividend Distibution")
    with st.spinner('Generating plot...'):
        plot_function = {'Line Plot': st.line_chart, 'Bar Plot': st.bar_chart,
                         'Pie Chart': st.pyplot, 'Histogram': st.pyplot}
        plot_func = plot_function[plot_type]
        if plot_type in ['Line Plot', 'Bar Plot']:
            plot_func(plot_data.set_index('Year'))
        elif plot_type == 'Pie Chart' and plot_data['Dividend'].sum() > 0:
            fig, ax = plt.subplots()
            ax.pie(plot_data['Dividend'],
                   labels=plot_data['Year'], autopct='%1.1f%%')
            ax.set_title('Dividend Distribution')
            st.pyplot(fig)
        elif plot_type == 'Histogram':
            fig, ax = plt.subplots()
            ax.hist(plot_data['Dividend'], bins=len(years))
            ax.set_title('Dividend Histogram')
            ax.set_xlabel('Dividend')
            ax.set_ylabel('Frequency')
            st.pyplot(fig)
        else:
            st.error("No data to display.")

# Additional analysis with filters
filtered_years = [year for year in years if start_year <= year <= end_year]
valid_columns = ['Companies'] + \
    [f'Growth Rate {year}' for year in filtered_years if f'Growth Rate {year}' in df.columns]
avg_dl_data = df[valid_columns]
if len(avg_dl_data.columns) > 1:
    avg_dl_data = avg_dl_data.set_index('Companies')
    avg_dl_data.columns = [
        f'Growth Rate {year}' for year in filtered_years if f'Growth Rate {year}' in df.columns]
    with st.expander("Companies Growth"):
        growth_plot_type = st.selectbox(
            'Select plot type:', ['Line Plot', 'Bar Chart'], key='growth_plot')
        if growth_plot_type == 'Line Plot':
            st.line_chart(avg_dl_data)
        elif growth_plot_type == 'Bar Chart':
            st.bar_chart(avg_dl_data)
else:
    st.error(
        "Insufficient data to plot. Please check the selected years or data availability.")

if ui.button(text="Companies Status", key="styled_btn_tailwind", className="bg-orange-500 text-white"):
    if 'Average DL' in df.columns:
        # Applying clustering
        df['KMeans_Cluster'] = model.predict(
            df[['Average DL']].values.reshape(-1, 1))
        df['Needs_Improvement'] = df['KMeans_Cluster'].apply(
            lambda x: 'Yes' if x == 0 else 'No')

        # Filter data for companies that need improvement
        improvement_needed_df = df[df['Needs_Improvement'] == 'Yes']

        if not improvement_needed_df.empty:
            with st.expander("Consult Table"):
                st.dataframe(improvement_needed_df[[
                             'Companies', 'Average DL', 'Needs_Improvement']], height=600)
            # Displaying the table with improvements needed
                # You can adjust height as needed

            # Sorting and plotting distribution if desired
            improvement_needed_df = improvement_needed_df.sort_values(
                'Average DL')
            improvement_needed_df['Company Index'] = range(
                1, len(improvement_needed_df) + 1)
            st.line_chart(improvement_needed_df.set_index(
                'Company Index')['Average DL'])
        else:
            st.write("No companies marked for improvements, or data unavailable.")
    else:
        st.error("Column 'Average DL' not found in the dataset.")

company = st.selectbox('Select a company:', dataPrep['Company'].unique())
company_data = dataPrep[dataPrep['Company'] == company]

if not company_data.empty:
    # Assuming 'Nominal' and 'Liquidity' are the features used for training
    if 'Nominal' in company_data.columns and 'Liquidity' in company_data.columns and 'Variation' in company_data.columns:
        features = company_data[['Nominal', 'Liquidity', 'Variation']]
        prediction = modelgradient_boosting_model.predict(features)
        # st.write(f"Predicted Dividend for {company} in 2023: {prediction[0]}")
    else:
        st.error(
            "The necessary features for prediction are not available in the dataset.")
else:
    st.error("No data available for the selected company.")
if not company_data.empty:
    # Check for the necessary features
    if {'Nominal', 'Liquidity', 'Variation'}.issubset(company_data.columns):
        features = company_data[['Nominal', 'Liquidity', 'Variation']]
        # Predict the dividend for 2023
        predicted_dividend = modelgradient_boosting_model.predict(
            features)[-1]  # Assuming the last prediction is for 2023

        # Create a DataFrame for the predicted value
        predicted_data = pd.DataFrame({
            'Year': [2023],
            'Dividend': [predicted_dividend],
            'Type': ['Predicted']  # Marking the predicted row
        })

        # Prepare historical data for plotting
        historical_data = company_data[['Year', 'Dividend']]
        historical_data['Type'] = 'Historical'

        # Combine historical and predicted data
        full_data = pd.concat([historical_data, predicted_data])

        # Plotting
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=full_data['Year'], y=full_data['Dividend'],
                                 mode='lines+markers',
                                 name='Dividend Trend',
                                 marker=dict(color=['blue']*len(historical_data) + ['red'])))

        # Add annotation for the year 2023 under the red dot
        fig.add_annotation(x=2023, y=predicted_dividend,
                           text="2023",
                           showarrow=True,
                           arrowhead=1,
                           ax=0,
                           ay=-40)

        fig.update_layout(title=f'Dividend Trend Predection for {company}',
                          xaxis_title='Year',
                          yaxis_title='Dividend',
                          legend_title='Data Type',
                          width=1500, height=600)
        st.plotly_chart(fig)
    else:
        st.error(
            "The necessary features for prediction are not available in the dataset.")
else:
    st.error("No data available for the selected company.")
# Optional: Additional visualizations or analytics
# For example, a plot of dividends over time
fig = px.line(dataPrep, x='Year', y='Dividend',
              color='Company', title='Dividend Over Time')
fig.update_layout(width=1500, height=600)
st.plotly_chart(fig)


""" filtered_df = dataPrep[(dataPrep['Year'] >= start_year) & (dataPrep['Year'] <= end_year)]
if not filtered_df.empty:
    fig = px.line(filtered_df, x='Year', y='Dividend', color='Company', title='Dividend Over Time')
    st.plotly_chart(fig)
else:
    st.error("No data available for the selected range.") """
