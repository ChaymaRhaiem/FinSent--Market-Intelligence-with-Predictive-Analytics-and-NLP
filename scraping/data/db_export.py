from pymongo import MongoClient
import csv

def export_to_csv(db_name, collection_name, csv_file_path):
    client = MongoClient("mongodb+srv://chaymarhaiem:value@trading.mvmqr8m.mongodb.net/?retryWrites=true&w=majority")
    db = client[db_name]
    collection = db[collection_name]

    cursor = collection.find({})
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Date", "Title", "Content"])  # Define the header row directly
        for doc in cursor:
            writer.writerow([doc["Date"], doc["Title"], doc["Content"]])
            

#if __name__ == "__main__":
#    export_to_csv("trading", "bvmt1", "actus_bvmt.csv")