from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import threading
import time
from pymongo import DESCENDING
from selenium import webdriver
import pandas as pd
import spacy
from wordcloud import WordCloud
import os
import time
import logging
import re
import string
import nltk
from nltk.corpus import stopwords
import schedule
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from db_connection import get_db
from tunisienumerique import scrape_articles as scrape_tn_articles
from bvmt_info import scrape_articles as scrape_bvmt_articles
from config import custom_entities_dict, finance_keywords, abbreviation_fullname_pairs
from datetime import datetime

# Set up logging

st.set_page_config(page_title="Innovest Ai Strategist",
                   page_icon="📈", layout="wide")
logger = logging.getLogger(__name__)


# Load FinBERT model and tokenizer
logger.info("Loading FinBERT model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
model.eval()
logger.info("FinBERT model and tokenizer loaded.")

# Load French spaCy model
logger.info("Loading French spaCy model...")
nlp = spacy.load("fr_core_news_lg")
nltk.download('stopwords')
nltk.download('wordnet')
logger.info("French spaCy model loaded.")

# Define directory to save the model
MODEL_DIR = "saved_model"
os.makedirs(MODEL_DIR, exist_ok=True)

# Define stopwords to exclude from removal
custom_stopwords = set(stopwords.words('french')) - {'de', 'et'}

# MongoDB connection
logger.info("Connecting to MongoDB...")
db = get_db()
# Update with actual collection name for Tunisie Numerique articles
collection_tn = db.tnumeco
collection_bvmt = db.societes  # Update with actual collection name for BVMT articles
logger.info("Connected to MongoDB.")

# Function to initialize the Selenium webdriver


def initialize_driver():
    logger.info("Initializing the Selenium webdriver...")
    options = webdriver.ChromeOptions()
    prefs = {
        "profile.managed_default_content_settings.images": 2,
        "profile.managed_default_content_settings.stylesheets": 2,
        "profile.managed_default_content_settings.fonts": 2
    }
    options.add_argument('headless')
    options.add_experimental_option("prefs", prefs)

    driver = webdriver.Chrome(options=options)
    logger.info("Selenium webdriver initialized.")
    return driver

# Function to preprocess text


def preprocess_text(text):
    logger.info("Preprocessing text...")
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    # Remove stopwords, excluding custom stopwords
    words = text.split()
    words = [word for word in words if word not in custom_stopwords]
    # Join the words back into a single string
    preprocessed_text = ' '.join(words)
    logger.info("Text preprocessed.")
    return preprocessed_text


# Function to extract entities and sentiment
def extract_entities_and_sentiment(text):
    try:
        if not text or pd.isnull(text):
            return None, None

        # Preprocess text
        preprocessed_text = preprocess_text(text)

        # Perform Named Entity Recognition (NER)
        doc = nlp(preprocessed_text)
        entities = []

        # Extract entities from custom_entities_dict
        custom_entities = [(entity_text, entity_type) for entity_text, entity_type in custom_entities_dict.items(
        ) if entity_text.lower() in preprocessed_text]

        # If there are entities in custom_entities, add them to the list
        if custom_entities:
            entities.extend(custom_entities)

        # Iterate over all entities found by spaCy
        for ent in doc.ents:
            # Check if the entity is not already in the entities list
            if ent.text.lower() not in [entity[0].lower() for entity in entities]:
                entity_type = custom_entities_dict.get(ent.text.lower())
                if entity_type:
                    entities.append((ent.text, entity_type))

        # If there are no valid entities, return None
        if not entities:
            return None, None

        # Perform sentiment analysis for each entity
        sentiments = []
        for entity_text, entity_type in entities:
            sentiment_score = None
            context = None

            # Find context for the entity
            sentences = re.split(
                r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
            for sentence in sentences:
                if entity_text.lower() in sentence.lower():
                    context = sentence
                    break

            # Check for predefined finance-related keywords in the context
            if context:
                # Initialize sentiment score
                sentiment_score = 0

                # Check for predefined finance keywords and adjust sentiment score accordingly
                for keyword, score in finance_keywords.items():
                    if keyword in context.lower():
                        sentiment_score += score

                # Perform sentiment analysis using FinBERT
                preprocessed_context = preprocess_text(context)
                inputs = tokenizer(
                    preprocessed_context, return_tensors="pt", truncation=True, padding=True)
                with torch.no_grad():
                    outputs = model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
                finbert_sentiment_score = probabilities[:, 2].item(
                ) - probabilities[:, 0].item()

                # Calculate final sentiment score
                sentiment_score += finbert_sentiment_score

                # Determine sentiment label based on sentiment score
                if sentiment_score > 0:
                    sentiment = "Positive"
                elif sentiment_score < 0:
                    sentiment = "Negative"
                else:
                    sentiment = "Neutral"

                # Append sentiment and sentiment score to list
                sentiments.append(
                    (entity_text, entity_type, context, sentiment_score, sentiment))

        return entities, sentiments

    except Exception as e:
        logger.error(f"Error processing text: {text}. Error: {e}")
        return None, None

# Function to update MongoDB document with sentiment, sentiment score, and entities


def update_article_with_sentiment(collection, article_id, sentiment, sentiment_score, entities, context):
    collection.update_one(
        {"_id": article_id},
        {"$set": {
            "sentiment": sentiment,
            "sentiment_score": sentiment_score,
            "entities": entities,
            "context": context
        }}
    )


def create_sentiment_gauge(sentiment_score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=sentiment_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Sentiment"},
        gauge={
            # Adjusted range
            'axis': {'range': [None, 2], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#f08b35"},
            'bgcolor': "white",
            'borderwidth': 1,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 0.9], 'color': 'white'},
                {'range': [0.9, 1.5], 'color': 'grey'},
                {'range': [1.5, 2], 'color': 'grey'}],  # Adjusted steps
            'threshold': {
                'line': {'color': "dark blue", 'width': 4},
                'thickness': 0.5,
                'value': sentiment_score}}))
    fig.update_layout(height=100, margin=dict(l=10, r=10, t=10, b=10))
    return fig


# Modify the display_latest_articles function to ensure sentiment gauges are displayed for all articles
def display_latest_articles(collection_tn, collection_bvmt, num_articles=6, display_key=0):
    # Retrieve the latest articles from the collections
    latest_articles_tn = collection_tn.find().sort(
        '_id', DESCENDING).limit(num_articles)
    latest_articles_bvmt = collection_bvmt.find().sort(
        '_id', DESCENDING).limit(num_articles)

    # Set up the layout
    cols_tn = st.columns(3)
    cols_bvmt = st.columns(3)

    # Display latest articles from Tunisie Numerique
    for i, article in enumerate(latest_articles_tn):
        entities, sentiment = extract_entities_and_sentiment(
            article['Content'])

        with cols_tn[i % 3]:
            with st.expander(article['Title']):
                lines = article['Content'].split('\n')[3:10]
                truncated_content = '\n'.join(lines)
                st.write(truncated_content)

                if 'URL' in article:
                    st.markdown(f"[Read More]({article['URL']})")

                if entities is not None:
                    st.subheader('Entities Detected:')
                    for entity, sentiment_info in zip(entities, sentiment):
                        st.write(f"- {entity[0]} ({entity[1]})")
                        sentiment_score = sentiment_info[3]
                        # Call create_sentiment_gauge for all articles
                        fig = create_sentiment_gauge(sentiment_score)
                        st.plotly_chart(fig)

    # Display latest articles from BVMT
    for i, article in enumerate(latest_articles_bvmt):
        entities, sentiment = extract_entities_and_sentiment(
            article['Content'])

        with cols_bvmt[i % 3]:
            with st.expander(article['Title']):
                lines = article['Content'].split('\n')[:10]
                truncated_content = '\n'.join(lines)
                st.write(truncated_content)

                if entities is not None:
                    st.subheader('Entities Detected:')
                    for entity, sentiment_info in zip(entities, sentiment):
                        st.write(f"- {entity[0]} ({entity[1]})")
                        sentiment_score = sentiment_info[3]
                        # Call create_sentiment_gauge for all articles
                        fig = create_sentiment_gauge(sentiment_score)
                        st.plotly_chart(fig)

    # Add a button to load more articles
    # Unique key for the button
    if st.button("Load More Articles", key=f"load_more_button_{display_key}"):
        display_latest_articles(collection_tn, collection_bvmt,
                                num_articles=num_articles+6, display_key=display_key+1)


def scrape_and_store_articles():
    last_scraped_date = datetime.now()  # Define last_scraped_date here
    num_failed_scrapes = 0  # Initialize num_failed_scrapes outside the try block
    num_tn_articles = 0
    num_bvmt_articles = 0
    start_time = datetime.now()  # Record the start time
    try:
        # Scrape Tunisie Numerique articles
        logger.info("Scraping Tunisie Numerique articles...")
        driver = initialize_driver()
        tn_csv_filename = "tunisienumerique_articles.csv"
        # Pass collection and csv_file_path
        tn_articles = scrape_tn_articles(
            driver, collection_tn, tn_csv_filename)

        # Check if any articles were scraped
        if tn_articles:
            for article in tn_articles:
                # Perform sentiment analysis and entity recognition
                entities, sentiment = extract_entities_and_sentiment(
                    article['Content'])

                # Add sentiment-related information to the article
                article['Entities'] = ', '.join(
                    [entity[0] for entity in entities]) if entities else ''
                article['Sentiment'] = ', '.join(
                    [sent[1] for sent in sentiment]) if sentiment else ''
                article['Sentiment Score'] = ', '.join(
                    [str(sent[3]) for sent in sentiment]) if sentiment else ''
                article['Context'] = ', '.join(
                    [sent[2] for sent in sentiment]) if sentiment else ''

            # Write articles to CSV
            tn_df = pd.DataFrame(tn_articles)
            tn_df.to_csv(tn_csv_filename, index=False)
            logger.info("Scraped and stored Tunisie Numerique articles.")
            num_tn_articles = collection_tn.count_documents({})

            # Verify data before writing to MongoDB
            logger.info(
                "Verifying Tunisie Numerique data before writing to MongoDB...")
            tn_data = pd.read_csv(tn_csv_filename) if os.path.exists(
                tn_csv_filename) else None
            logger.info(f"Tunisie Numerique data: {tn_data}")

            # If data exists, write it to MongoDB
            if tn_data is not None:
                logger.info("Writing Tunisie Numerique data to MongoDB...")
                # Write tn_data to MongoDB collection_tn
                collection_tn.insert_many(tn_data.to_dict(orient='records'))
                logger.info("Tunisie Numerique data written to MongoDB.")

        else:
            logger.warning("No Tunisie Numerique articles were scraped.")

        # Scrape BVMT articles
        logger.info("Scraping BVMT articles...")
        bvmt_csv_filename = "sociétés_bvmt.csv"
        # Pass driver and last_scraped_date
        bvmt_articles = scrape_bvmt_articles(driver, None, last_scraped_date)

        # Check if any articles were scraped
        if bvmt_articles:
            for article in bvmt_articles:
                # Perform sentiment analysis and entity recognition
                entities, sentiment = extract_entities_and_sentiment(
                    article['Content'])

                # Add sentiment-related information to the article
                article['Entities'] = ', '.join(
                    [entity[0] for entity in entities]) if entities else ''
                article['Sentiment'] = ', '.join(
                    [sent[0] for sent in sentiment]) if sentiment else ''
                article['Sentiment Score'] = ', '.join(
                    [str(sent[3]) for sent in sentiment]) if sentiment else ''
                article['Context'] = ', '.join(
                    [sent[2] for sent in sentiment]) if sentiment else ''

            # Write articles to CSV
            bvmt_df = pd.DataFrame(bvmt_articles)
            bvmt_df.to_csv(bvmt_csv_filename, index=False)
            logger.info("Scraped and stored BVMT articles.")
            num_bvmt_articles = collection_bvmt.count_documents({})
            print('bvmt articles found : ', num_bvmt_articles)
            # Verify data before writing to MongoDB
            logger.info("Verifying BVMT data before writing to MongoDB...")
            bvmt_data = pd.read_csv(bvmt_csv_filename) if os.path.exists(
                bvmt_csv_filename) else None
            logger.info(f"BVMT data: {bvmt_data}")

            # If data exists, write it to MongoDB
            if bvmt_data is not None:
                logger.info("Writing BVMT data to MongoDB...")
                # Write bvmt_data to MongoDB collection_bvmt
                collection_bvmt.insert_many(
                    bvmt_data.to_dict(orient='records'))
                logger.info("BVMT data written to MongoDB.")

        logger.info("Data verification complete.")

    except Exception as e:
        logger.error(f"Error in scrape_and_store_articles: {e}")
        num_failed_scrapes += 1  # Increment the number of failed scrapes

    # Calculate the total time taken for scraping
    end_time = datetime.now()
    scraping_duration = end_time - start_time

    # Log the scraping duration and summary
    logger.info(f"Scraping finished at: {end_time}")
    logger.info(f"Scraping duration: {scraping_duration} seconds")
    logger.info(f"Total failed scrapes: {num_failed_scrapes}")

    return num_tn_articles, num_bvmt_articles

# Function to analyze articles and add sentiment-related columns


def analyze_articles(collection):
    articles = collection.find()  # Retrieve articles from MongoDB collection
    for article in articles:
        # Perform sentiment analysis and entity recognition
        entities, sentiment = extract_entities_and_sentiment(
            article['Content'])

        # Prepare data to be added to the article
        entity_names = [entity[0] for entity in entities] if entities else []
        sentiment_scores = [sent[3] for sent in sentiment] if sentiment else []
        context = [sent[2] for sent in sentiment] if sentiment else []

        # Add new columns to the article
        collection.update_one(
            {"_id": article['_id']},
            {"$set": {
                "entities": entity_names,
                "sentiment": sentiment,
                "sentiment_scores": sentiment_scores,
                "context": context
            }}
        )


def check_schedule():
    while True:
        # Get current time
        now = datetime.now().time()
        logger.info(f"Current time: {now}")
        # Check if it's time to scrape and store articles (every 5 minutes)
        if now.minute == 0:  # Check if the current minute is a multiple of 5
            logger.info("Scraping and storing articles...")
            scrape_and_store_articles()
            logger.info("Articles scraped and stored.")
            logger.info("Analyzing articles...")
            analyze_articles(collection_tn)
            analyze_articles(collection_bvmt)
            logger.info("Articles analyzed.")
            # Get the count of documents in collection_tn

        # Sleep for 1 minute
        time.sleep(60)


def generate_word_cloud(text):
    # Preprocess text to remove unwanted words
    stopwords_fr = set(stopwords.words('french'))
    custom_stopwords = {'linkedin', 'facebook', 'twitter', 'dinar', 'whatsapp', 'tunisien', 'tunisienne', 'abonnez',
                        'commentaires', 'tunisie', 'tunis', 'lâ', 'tâ', 'dâ', 'il', 'abonnez', 'dã', 'lã'}  # Add other unwanted words here
    stopwords_fr.update(custom_stopwords)

    # Tokenize the text and filter out stopwords and punctuation
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)

    # Filter out stopwords, punctuation, and certain POS tags
    filtered_tokens = [word for word, pos in pos_tags if word.lower(
    ) not in stopwords_fr and word not in string.punctuation and pos not in ('CC', 'IN')]

    # Lemmatize the words to reduce them to their root form
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(
        token) for token in filtered_tokens]

    # Join the filtered tokens back into a single string
    filtered_text = ' '.join(lemmatized_tokens)

    # Calculate TF-IDF scores for words
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([filtered_text])
    feature_names = vectorizer.get_feature_names_out()

    # Create a dictionary to store TF-IDF scores for words
    word_scores = {}
    for word, score in zip(feature_names, tfidf_matrix.toarray()[0]):
        word_scores[word] = score

    # Sort words by their TF-IDF scores in descending order
    sorted_words = sorted(word_scores.items(),
                          key=lambda x: x[1], reverse=True)

    # Get top 100 important words based on TF-IDF scores
    important_words = [word for word, score in sorted_words[:80]]

    # Generate the word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='#0E1117').generate(
        ' '.join(important_words))

    # Display the word cloud
    st.image(wordcloud.to_array())

    # Return the word cloud object for interaction
    return wordcloud

# Function to display latest articles and word cloud


def display_latest_articles_with_word_cloud(collection_tn, collection_bvmt, selected_word=None):
    st.title("Trading Trends: Tunisian Stock Market Buzz")

    # Retrieve the text data from the latest articles
    tn_articles = list(collection_tn.find().sort('_id', DESCENDING).limit(10))
    bvmt_articles = list(collection_bvmt.find().sort(
        '_id', DESCENDING).limit(10))

    # Combine the text data from both collections
    all_articles = tn_articles + bvmt_articles
    combined_text = ' '.join([article['Content'] for article in all_articles])

    # Generate and display the word cloud
    wordcloud = generate_word_cloud(combined_text)

    # Get user-selected word
    clicked_word = st.selectbox("Select a word from the word cloud", [
                                word for word in wordcloud.words_], index=0)

    # Filter articles based on clicked word
    filtered_articles = [article for article in all_articles if clicked_word and clicked_word.lower(
    ) in article['Content'].lower()]
    # Display filtered articles
    st.subheader('Articles containing the selected word:')
    for article in filtered_articles:
        with st.expander(article['Title']):
            # Display the first 10 lines of the article
            lines = article['Content'].split('\n')[:10]
            truncated_content = '\n'.join(lines)
            st.write(truncated_content)

            # Perform sentiment analysis and entity extraction
            entities, sentiment = extract_entities_and_sentiment(
                article['Content'])
            if sentiment:
                st.subheader('Sentiment Analysis:')
                for sent in sentiment:
                    st.write(f"Context: {sent[2]}")
                    st.write(f"Sentiment Score: {sent[3]}")
                    st.write(f"Sentiment: {sent[4]}")
                    # Display sentiment gauge
                    sentiment_score = sent[3]
                    fig = create_sentiment_gauge(sentiment_score)
                    st.plotly_chart(fig)


# Update the function to generate entity sentiment gauges
def generate_entity_sentiment_gauges(all_articles):
    entity_sentiment_scores = {}
    for article in all_articles:
        entities = article.get('entities', [])
        sentiment_scores = article.get('sentiment_scores', [])

        for entity, score in zip(entities, sentiment_scores):
            if entity in entity_sentiment_scores:
                entity_sentiment_scores[entity].append(score)
            else:
                entity_sentiment_scores[entity] = [score]
        # Map abbreviations to full names
    abbreviation_mapping = create_abbreviation_mapping()
    entity_sentiment_scores_full = {}
    for entity, scores in entity_sentiment_scores.items():
        full_name = abbreviation_mapping.get(entity.lower(), entity)
        entity_sentiment_scores_full[full_name] = scores

    average_scores = {entity: np.mean(
        scores) for entity, scores in entity_sentiment_scores.items()}

    # Sort companies based on average sentiment score
    top_companies = dict(sorted(average_scores.items(),
                         key=lambda x: x[1], reverse=True)[:5])

    return average_scores, top_companies


def create_abbreviation_mapping():

    # Create the abbreviation mapping dictionary
    abbreviation_mapping = {abbreviation.lower(
    ): fullname for abbreviation, fullname in abbreviation_fullname_pairs}

    return abbreviation_mapping


def main():
    tn_articles = list(collection_tn.find().sort('_id', DESCENDING).limit(10))
    bvmt_articles = list(collection_bvmt.find().sort(
        '_id', DESCENDING).limit(10))
    all_articles = tn_articles + bvmt_articles

    # Add home page content here
    # Display latest articles and word cloud
    display_latest_articles_with_word_cloud(collection_tn, collection_bvmt)

    # Display latest articles from the database
    st.subheader('Latest Market Headlines ')
    # Include both collections here
    display_latest_articles(collection_tn, collection_bvmt)

    # Add a refresh button
    if st.button("Refresh"):
        logger.info("Manual refresh triggered.")
        # Scrape and store articles
        # Update variables with returned values
        num_tn_articles, num_bvmt_articles = scrape_and_store_articles()
        # Analyze articles and add sentiment-related columns
        analyze_articles(collection_tn)
        analyze_articles(collection_bvmt)
        num_tn_documents = collection_tn.count_documents({})
        # Get the count of documents in collection_bvmt
        num_bvmt_documents = collection_bvmt.count_documents({})
        # Print the counts
        logger.info(
            f"Total Tunisie Numerique articles scraped: {num_tn_documents}")
        logger.info(f"Total BVMT articles scraped: {num_bvmt_documents}")
        logger.info(
            f"Total articles scraped: {num_tn_articles + num_bvmt_articles}")

    # Generate entity sentiment gauges
    average_scores, top_companies = generate_entity_sentiment_gauges(
        all_articles)

    # Create a dropdown with all companies
    selected_company = st.selectbox(
        "Select a company", list(average_scores.keys()))

    # Display sentiment info for the selected company
    if selected_company:
        average_score = average_scores[selected_company]
        st.write(
            f"Avg. Sentiment Score for {selected_company}: {average_score}")
        sentiment_type = "Positive" if average_score > 0.5 else "Negative"
        st.write(f"Sentiment Type: {sentiment_type}")
        # Create sentiment gauge for the selected company
        fig = create_sentiment_gauge(average_score)
        st.plotly_chart(fig)

    worst_companies = dict(
        sorted(average_scores.items(), key=lambda item: item[1])[:5])

# Get the top 5 best companies (highest scores)
    # Sort companies based on average sentiment score
    top_companies = dict(sorted(average_scores.items(),
                         key=lambda x: x[1], reverse=True)[0:5])

# Create two columns for positive and negative sentiment
    # Create two columns for positive and negative sentiment
    col1, col2 = st.columns(2)

    # Display the title of each column with colored headers
    with col1:
        st.markdown(
            "<h2 style='color: #336699;'>Positivity Corner: Top Players in Financial Sentiment</h2>", unsafe_allow_html=True)

    with col2:
        st.markdown(
            "<h2 style='color: #CC0000;'>Red Flags: Financial Entities with Negative Sentiment</h2>", unsafe_allow_html=True)

    # Display the average sentiment gauges for the top 5 companies with the best sentiment scores
    with col1:
        for company, sentiment_score in top_companies.items():
            st.subheader(company)
            fig = create_sentiment_gauge(sentiment_score)
            # Adjust size to fit container
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("---")  # Add separation line

    # Display the sentiment info for the worst 5 companies
    with col2:
        for company, sentiment_score in worst_companies.items():
            st.subheader(company)
            fig = create_sentiment_gauge(sentiment_score)
            # Adjust size to fit container
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("---")  # Add separation line


if __name__ == '__main__':
    # Start the check_schedule function in a separate thread
    schedule_thread = threading.Thread(target=check_schedule, daemon=True)
    schedule_thread.start()

    # Run the Streamlit application
    main()
