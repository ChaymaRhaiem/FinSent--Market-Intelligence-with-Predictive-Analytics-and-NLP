from pymongo import MongoClient
import pandas as pd

from credentials import get_db


def insert_data_to_mongodb(db, collection_name, csv_file):
    print(f"Loading CSV file {csv_file} into a pandas DataFrame...")
    df = pd.read_csv(csv_file)

    print("Converting the DataFrame to a list of dictionaries...")
    records = df.to_dict('records')

    print(f"Inserting records into the {collection_name} collection...")
    db[collection_name].insert_many(records)
    print("Data insertion complete.")


def insert_data_to_mongodb_Delimiter(db, collection_name, csv_file):
    print(f"Loading CSV file {csv_file} into a pandas DataFrame...")
    df = pd.read_csv(csv_file, delimiter=';', thousands=',')

    print("Converting the DataFrame to a list of dictionaries...")
    records = df.to_dict('records')

    print(f"Inserting records into the {collection_name} collection...")
    db[collection_name].insert_many(records)
    print("Data insertion complete.")


# Connect to MongoDB
db = get_db()

# Specify CSV file paths and collection names
csv_files = ["C:/Users/21629/Downloads/Deployement (1)/Deployement/Data/dataPrep.csv",
             "C:/Users/21629/Downloads/Deployement (1)/Deployement/Data/Dividend.csv"]
collection_names = ["dataPrep", "Dividend"]

csv_files_Delimiter = ["C:/Users/21629/Downloads/Deployement (1)/Deployement/Data/histo_cotation_2019.csv",
                       "C:/Users/21629/Downloads/Deployement (1)/Deployement/Data/histo_cotation_2020.csv",
                       "C:/Users/21629/Downloads/Deployement (1)/Deployement/Data/histo_cotation_2021.csv",
                       "C:/Users/21629/Downloads/Deployement (1)/Deployement/Data/histo_cotation_2022.csv",
                       "C:/Users/21629/Downloads/Deployement (1)/Deployement/Data/histo_cotation_2023.csv"]
collection_names_Delimiter = ["histo_cotation_2019", "histo_cotation_2020",
                              "histo_cotation_2021", "histo_cotation_2022", "histo_cotation_2023"]
# Insert data from each CSV file into its corresponding collection
for csv_file, collection_name in zip(csv_files, collection_names):
    insert_data_to_mongodb(db, collection_name, csv_file)
for csv_file, collection_name in zip(csv_files_Delimiter, collection_names_Delimiter):
    insert_data_to_mongodb_Delimiter(
        db, collection_name, csv_file)
