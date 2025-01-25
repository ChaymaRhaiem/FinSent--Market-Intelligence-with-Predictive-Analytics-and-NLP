from pymongo import MongoClient


def get_db():
    # Replace the following with your MongoDB Atlas connection string
    connection_string = "mongodb+srv://chaymarhaiem:value@.."

    client = MongoClient(connection_string)
    db = client.trading  # Replace with your database name

    collection_names = db.list_collection_names()
    print("Collections in the database:", collection_names)

    return db
