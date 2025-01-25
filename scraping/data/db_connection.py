from pymongo import MongoClient

def get_db():
    # Replace the following with your MongoDB Atlas connection string
    connection_string = "mongodb+srv://..." # replace with ur api acess string or .env variable
     
    client = MongoClient(connection_string)
    db = client.trading  # Replace with your database name

    # Fetch and print the names of all collections in your database
    collection_names = db.list_collection_names()
    print("Collections in the database:", collection_names)
    collection_tn = db.tnumeco  # Update with actual collection name for Tunisie Numerique articles
    collection_bvmt = db.societes  # Update with actual collection name for BVMT articles

    num_tn_documents = collection_tn.count_documents({})

            # Get the count of documents in collection_bvmt
    num_bvmt_documents = collection_bvmt.count_documents({})

            # Print the counts
    print(f"Number of documents in collection_tn: {num_tn_documents}")
    print(f"Number of documents in collection_bvmt: {num_bvmt_documents}")


    return db


