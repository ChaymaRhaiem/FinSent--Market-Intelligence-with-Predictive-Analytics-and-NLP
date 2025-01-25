from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from db_connection import get_db
import csv

# Function to initialize the Selenium webdriver


def initialize_driver():
    options = webdriver.ChromeOptions()
    # options.add_argument('--headless')  # Use headless mode to not open a browser window
    driver = webdriver.Chrome(options=options)
    return driver


def scrape_articles(driver, writer):
    # Navigate to the BVMT website
    driver.get(
        "https://www.mosaiquefm.net/fr/actualites/actualite-economie-tunisie/5/1")

    while True:
        try:
            # Wait for the articles to be loaded
            WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located(
                (By.CSS_SELECTOR, ".col-xl-6.col-md-8")))

            # Find all articles
            articles = driver.find_elements(
                By.CSS_SELECTOR, ".col-xl-6.col-md-8")

            # Loop through each article and scrape data
            for article in articles:
                try:
                    # Extract the URL of the article
                    article_url = article.find_element(
                        By.TAG_NAME, "a").get_attribute("href")

                    # Navigate to the URL of the article
                    driver.get(article_url)

                    # Wait for the article content to be visible
                    WebDriverWait(driver, 10).until(EC.visibility_of_element_located(
                        (By.CSS_SELECTOR, ".article-content")))

                    # Extract the title of the article
                    title_element = WebDriverWait(driver, 10).until(
                        EC.visibility_of_element_located((By.TAG_NAME, "h1")))
                    title = title_element.text

                    # Extract the content of the article
                    content_element = driver.find_element(
                        By.CSS_SELECTOR, ".article-content")
                    content = content_element.text

                    # Write the scraped data to the CSV file
                    writer.writerow([article_url, title, content])

                except NoSuchElementException as e:
                    print("Element not found. Skipping article...")
                    print(e)  # Print the exception for further investigation
                    continue

                finally:
                    # Navigate back to the list of articles
                    driver.back()

        except TimeoutException:
            print("TimeoutException: No more articles found.")
            break

        try:
            # Find and click the 'next' button to go to the next page
            next_button = driver.find_element(By.LINK_TEXT, ">")
            next_button.click()

        except NoSuchElementException:
            print("No more pages left.")
            break


def main():
    # Initialize the webdriver
    driver = initialize_driver()

    # Scrape articles and write them into a CSV file
    with open('mozaique.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Define the header row directly
        writer.writerow(["Date", "Title", "Content"])
        scrape_articles(driver, writer)  # Pass the CSV writer to the function

    # Close the webdriver
    driver.quit()

    # Now, you can read from the CSV file
    with open('mozaique.csv', 'r') as file:
        reader = csv.reader(file)
        # Define the header as the first row of the CSV file
        header = next(reader)
        # Get a reference to the database
        db = get_db()
        # Get a reference to your collection
        collection = db.mozaique  # Replace with your collection name
        # Insert each row of the CSV file into the collection
        for row in reader:
            doc = {field: value for field, value in zip(header, row)}
            collection.insert_one(doc)


if __name__ == "__main__":
    main()
