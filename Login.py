import subprocess
import streamlit as st
from pymongo import MongoClient
import bcrypt
from streamlit_lottie import st_lottie
import requests
import os
import pandas as pd
import plotly.express as px
from PIL import Image
from bson import ObjectId

# Load Lottie animations


def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


login_animation = load_lottieurl(
    'https://lottie.host/cef92f19-46da-4168-8557-c3cc288d33ac/4iTyHRCUkD.json')
signup_animation = load_lottieurl(
    'https://lottie.host/9d73d129-a797-4fc8-aa91-3a1201d91af4/8sO2yGdInS.json')

# MongoDB connection setup
client = MongoClient(
    "mongodb+srv://chaymarhaiem:value@trading.mvmqr8m.mongodb.net/?retryWrites=true&w=majority")
db = client["trading"]
users_collection = db["user"]
logs_collection = db["logs"]

# Save image function


def save_image(image_file, username):
    directory = "uploaded_images"
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, f"{username}.png")
    image = Image.open(image_file)
    image.save(filepath, format='PNG')
    return filepath

# Register user function


def register_user(username, password, image_file):
    if users_collection.find_one({"username": username}):
        return False
    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    image_path = save_image(image_file, username)
    users_collection.insert_one(
        {"username": username, "password": hashed, "image_path": image_path})
    return True

# Verify user function


def verify_user(username, password):
    if username == "admin" and password == "admin":
        return {"username": "admin", "is_admin": True}
    user = users_collection.find_one({"username": username})
    if user and bcrypt.checkpw(password.encode('utf-8'), user['password']):
        return user
    return None

# Password validation function


def valid_password(password):
    missing_requirements = []
    if not any(c.isupper() for c in password):
        missing_requirements.append("at least one uppercase letter")
    if not any(c.islower() for c in password):
        missing_requirements.append("at least one lowercase letter")
    if not any(c.isdigit() for c in password):
        missing_requirements.append("at least one digit")
    if not any(c in "!@#$%^&*()-_=+[{]}\\|;:'\",<.>/?`~" for c in password):
        missing_requirements.append("at least one special character")
    if len(password) < 8:
        missing_requirements.append("at least 8 characters")
    return missing_requirements


# Set page config
st.set_page_config(page_title="Innovest Ai Strategist",
                   page_icon="📈", layout="wide")

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'show_login' not in st.session_state:
    st.session_state.show_login = True

# Function to logout


def logout():
    st.session_state.logged_in = False
    st.session_state.username = None

# Render backend function


def render_backend():
    st.markdown("""
    <style>
        .title {
            font-size: 3em;
            color: #333;
            text-align: center;
            margin-top: 20px;
        }
        .stButton>button {
            width: auto;
            color: #FAFAFA;
            border: 1px solid #FF4B4B;
            background-color: #0E1117;
            margin: 0.5em;
        }
        .stTextInput>div>input {
            color: #FAFAFA;
            background-color: #262730;
        }
        .stDataFrame {
            width: 100%;
        }
        .button-container {
            display: flex;
            justify-content: center;
            align-items: center;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="title">Admin Dashboard</div>',
                unsafe_allow_html=True)

    # User Management Section
    st.subheader("User Management")
    users = list(users_collection.find(
        {}, {"_id": 0, "username": 1, "password": 1, "image_path": 1}))
    df_users = pd.DataFrame(users)
    df_users['username'] = df_users['username'].astype(str)
    df_users['password'] = df_users['password'].apply(
        lambda x: x.decode('utf-8') if isinstance(x, bytes) else str(x))
    df_users['image_path'] = df_users['image_path'].astype(str)
    df_users['image_provided'] = df_users['image_path'].apply(
        lambda x: 1 if x else 0)

    st.write("All Users")
    st.dataframe(df_users, use_container_width=True)

    selected_user = None
    username_to_update = st.selectbox(
        "Select User to Update/Delete", df_users['username'])

    if username_to_update:
        selected_user = df_users[df_users['username']
                                 == username_to_update].iloc[0]
        st.write(f"Selected User: {selected_user['username']}")

        st.markdown('<div class="button-container">', unsafe_allow_html=True)
        if st.button("Delete User", key="delete_user"):
            users_collection.delete_one({"username": username_to_update})
            st.success(f"User {username_to_update} deleted successfully.")
            st.experimental_rerun()

        new_password = st.text_input(
            "New Password for selected user", type="password")
        if st.button("Update Password", key="update_password"):
            missing_reqs = valid_password(new_password)
            if not missing_reqs:
                hashed = bcrypt.hashpw(
                    new_password.encode('utf-8'), bcrypt.gensalt())
                users_collection.update_one({"username": username_to_update}, {
                                            "$set": {"password": hashed}})
                st.success(
                    f"Password for user {username_to_update} updated successfully.")
            else:
                st.error(
                    "Password must meet the following requirements: " + ", ".join(missing_reqs))
        st.markdown('</div>', unsafe_allow_html=True)

    # Logs Section
    st.subheader("Logs")
    logs = list(logs_collection.find().limit(100))
    for log in logs:
        st.markdown(f"**{log['timestamp']}:** {log['message']}")

    # Collection Management Section
    st.subheader("Collection Management")
    collections = db.list_collection_names()
    selected_collection = st.selectbox("Select Collection", collections)

    if selected_collection:
        collection = db[selected_collection]
        documents = list(collection.find())
        if documents:
            df_documents = pd.DataFrame(documents)
            st.write(f"All Documents in {selected_collection}")
            st.dataframe(df_documents, use_container_width=True)

            # Select document to delete by name
            field_name = st.selectbox(
                "Select Field to Delete By", df_documents.columns)
            if field_name:
                document_names = df_documents[field_name].astype(str)
                document_to_delete = st.selectbox(
                    "Select Document to Delete", document_names)
                if document_to_delete:
                    if st.button(f"Delete Document {document_to_delete}", key="delete_document"):
                        collection.delete_one({field_name: document_to_delete})
                        st.success(
                            f"Document {document_to_delete} deleted successfully.")
                        st.experimental_rerun()

        st.write(f"Add New Document to {selected_collection}")
        new_document = {}
        for field in collection.find_one().keys():
            if field != '_id':
                new_document[field] = st.text_input(f"{field}")

        if st.button(f"Add Document to {selected_collection}", key="add_document"):
            collection.insert_one(new_document)
            st.success(
                f"Document added to {selected_collection} successfully.")
            st.experimental_rerun()

    # Plots Section
    st.subheader("Plots")
    st.write("Example plots using user data")

    if not df_users.empty:
        df_users['password_length'] = df_users['password'].apply(len)
        fig1 = px.histogram(df_users, x='password_length',
                            nbins=10, title='Password Length Distribution')
        fig1.update_layout(width=1000)
        st.plotly_chart(fig1, use_container_width=True)

        df_image_provided = df_users['image_provided'].value_counts(
        ).reset_index()
        df_image_provided.columns = ['Image Provided', 'Count']
        fig2 = px.bar(df_image_provided, x='Image Provided',
                      y='Count', title='Image Provided Distribution')
        fig2.update_layout(width=1000)
        st.plotly_chart(fig2, use_container_width=True)


if st.session_state.logged_in:
    if st.session_state.username == "admin":
        # Sidebar for logout option in admin view
        st.sidebar.title("Options")
        if st.sidebar.button("Logout"):
            logout()
            st.experimental_rerun()

        render_backend()
    else:
        exec(open("Homepage.py").read())
        subprocess.Popen(["python", "Scrape.py", "--server.port", "8590"])
else:
    st.markdown("""
    <style>
                .main-title {
            font-size: 3em;
            color: #333;
            text-align: center;
            margin-top: 20px;
        }
        .sub-title {
            font-size: 1.5em;
            color: #666;
            text-align: center;
            margin-bottom: 20px;
        }
        .css-1d391kg {
            background-color: #262730;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .stButton>button {
            width: 100;
            color: #FAFAFA;
            border: 1px solid #FF4B4B;
            background-color: #0E1117;
            margin: 0.5em;
        }
        .stTextInput>div>input {
            color: #FAFAFA;
            background-color: #262730;
        }
        .button-container {
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .button-container-small {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-top: 10px;
        }
        .button-container-small button {
            margin-right: 10px;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-title">Innovest Ai Strategist</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Your AI-powered financial advisor</div>',
                unsafe_allow_html=True)

    st_lottie(
        login_animation if st.session_state.show_login else signup_animation, height=300)

    if st.session_state.show_login:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_submitted = st.form_submit_button("Login")
            if login_submitted:
                user_data = verify_user(username, password)
                if user_data:
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = username
                    st.session_state['user_image'] = user_data.get(
                        'image_path')
                    st.experimental_rerun()
                else:
                    st.error("Invalid username or password")

        st.markdown('<div class="button-container-small">',
                    unsafe_allow_html=True)
        st.button("Sign Up Instead",
                  on_click=lambda: st.session_state.update(show_login=False))
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        with st.form("signup_form"):
            new_username = st.text_input("Choose a username")
            new_password = st.text_input("Choose a password", type="password")
            confirm_password = st.text_input(
                "Confirm password", type="password")
            image_file = st.file_uploader(
                "Upload your profile image", type=['png', 'jpg', 'jpeg'])
            register_submitted = st.form_submit_button("Register")
            if register_submitted:
                if new_password != confirm_password:
                    st.error("Passwords do not match!")
                else:
                    missing_reqs = valid_password(new_password)
                    if missing_reqs:
                        st.error(
                            "Password must meet the following requirements: " + ", ".join(missing_reqs))
                    elif image_file:
                        if register_user(new_username, new_password, image_file):
                            st.success("You have successfully registered.")
                            st.session_state.show_login = True
                    else:
                        st.error("Please upload an image.")

        st.markdown('<div class="button-container-small">',
                    unsafe_allow_html=True)
        st.button("Back to Login",
                  on_click=lambda: st.session_state.update(show_login=True))
        st.markdown('</div>', unsafe_allow_html=True)
