import streamlit as st
import pandas as pd
import plotly.express as px  # Import Plotly Express
from db_connection.credentials import get_db

db = get_db()

# Specify the collection (table) you want to use


def load_data_from_db(year):
    collection_name = f"histo_cotation_{year}"
    collection = db[collection_name]
    # Fetching all documents from the collection
    data_list = list(collection.find())
    # Convert the list of documents (dictionaries) to a DataFrame
    df = pd.DataFrame(data_list)
    # Assuming you want to handle numeric conversion similar to the CSV's thousands separator
    # Here you need to specify which columns require numeric conversion if not automatically inferred
    for col in df.columns:
        # Attempt to convert columns to numeric, errors='ignore' will keep original data if conversion fails
        df[col] = pd.to_numeric(df[col], errors='ignore')
    return df


st.title("Company Capital Distribution")
year = st.selectbox("Select a year", options=[2019, 2020, 2021, 2022, 2023])

# Load the dataset based on selected year
data = load_data_from_db(year)

if 'CAPITAUX' in data.columns and 'VALEUR' in data.columns:
    data['CAPITAUX'] = pd.to_numeric(
        data['CAPITAUX'], errors='coerce').fillna(0)
    company_capital = data.groupby('VALEUR')['CAPITAUX'].sum()

    # Define classification based on capital
    def classify_company(capital):
        if capital > 200000:
            return 'Class A'
        elif capital == 0:
            return 'Class S'
        else:
            return 'Class B'

    # Trigger to execute classification and plotting
    if st.button("Analyze and Classify Companies"):
        with st.expander("Classification Analysis"):
            company_capital_df = company_capital.reset_index()
            company_capital_df['Class'] = company_capital_df['CAPITAUX'].apply(
                classify_company)
            # Display companies with their classification
            st.write("Company Classification:", company_capital_df)
            class_distribution = company_capital_df['Class'].value_counts()
            st.line_chart(class_distribution)
            # Create a Plotly scatter plot for the distribution of classes
            fig = px.scatter(company_capital_df, x='VALEUR', y='CAPITAUX', color='Class',
                             labels={"VALEUR": "Company",
                                     "CAPITAUX": "Capital"},
                             title="Distribution of Companies by Class")
            st.plotly_chart(fig)

else:
    st.error("Required columns 'VALEUR' or 'CAPITAUX' are missing in the dataset.")

# Handle numeric conversion for other columns
for col in ['OUVERTURE', 'CLOTURE', 'PLUS_BAS', 'PLUS_HAUT', 'QUANTITE_NEGOCIEE', 'NB_TRANSACTION']:
    data[col] = pd.to_numeric(data[col], errors='coerce')
data.fillna(0, inplace=True)

# Clean and convert SEANCE to datetime, specifying the format
if 'SEANCE' in data.columns:
    data['SEANCE'] = data['SEANCE'].str.strip()
    data = data[data['SEANCE'].str.match(r'\d{2}/\d{2}/\d{4}') == True]
    try:
        data['SEANCE'] = pd.to_datetime(data['SEANCE'], format='%d/%m/%Y')
        data.set_index('SEANCE', inplace=True)

        # Display line charts for each metric
        metrics = ['OUVERTURE', 'CLOTURE', 'PLUS_BAS',
                   'PLUS_HAUT', 'QUANTITE NEGOCIEE', 'NB_TRANSACTION']
        for metric in metrics:
            st.subheader(metric)
            st.line_chart(data[metric])
    except Exception as e:
        st.error(f"Failed to parse SEANCE column: {e}")
else:
    st.error("Date column 'SEANCE' is missing in the dataset.")
