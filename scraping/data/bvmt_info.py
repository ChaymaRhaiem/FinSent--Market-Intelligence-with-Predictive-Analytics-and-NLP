import logging
import csv
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

from db_connection import get_db

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Function to initialize the Selenium webdriver


def initialize_driver():
    options = webdriver.ChromeOptions()
    prefs = {
        "profile.managed_default_content_settings.images": 2,
        "profile.managed_default_content_settings.stylesheets": 2,
        "profile.managed_default_content_settings.fonts": 2
    }
    options.add_argument('headless')
    options.add_experimental_option("prefs", prefs)

    driver = webdriver.Chrome(options=options)
    return driver


def scrape_articles(driver, writer, last_scraped_date):
    logger.info("Scraping articles...")
    driver.get("https://www.bvmt.com.tn/fr/actualites-emetteurs")

    # Define the range of pages to scrape
    for page_number in range(1, 10):  # Scraping last 10 pages
        logger.info(f"Scraping page {page_number}...")

        articles = driver.find_elements(
            By.CSS_SELECTOR, "#actus-list > div[class='midl']")
        for i in range(len(articles)):
            try:
                articles = driver.find_elements(
                    By.CSS_SELECTOR, "#actus-list > div")
                article_url = articles[i].find_element(
                    By.XPATH, "./a").get_attribute("href")
                date = articles[i].find_element(
                    By.CSS_SELECTOR, "#list-actu01 > p > span.orange").text
                if date <= last_scraped_date:  # If the article's date is older or equal to the last scraped date, exit
                    logger.info("Reached last scraped article. Exiting...")
                    return
                for attempt in range(3):  # Retry 3 times
                    try:
                        driver.get(article_url)
                        break
                    except Exception as e:
                        logger.error(
                            f"Attempt {attempt+1} failed with error: {str(e)}. Retrying...")
                        time.sleep(5)  # Wait for 5 seconds before retrying
                else:  # If all attempts fail, skip this article
                    logger.warning(
                        "All attempts failed. Skipping this article.")
                    continue
                WebDriverWait(driver, 10).until(
                    EC.visibility_of_element_located((By.CSS_SELECTOR, "#bl-01")))
                title = driver.find_element(
                    By.CSS_SELECTOR, "#bl-01 > h3").text
                content = driver.find_element(By.CSS_SELECTOR, "#bl-01").text
                writer.writerow([date, title, content])
                logger.info(f"Scraped article: {title} - Date: {date}")
                print(f"Scraped article: {title} - Date: {date}")
            except NoSuchElementException:
                logger.warning("Element not found. Skipping...")
                continue
            finally:
                driver.back()
                WebDriverWait(driver, 10).until(EC.visibility_of_element_located(
                    (By.CSS_SELECTOR, "#actus-list > div")))

        # Navigate to the next page
        try:
            next_button = driver.find_element(By.LINK_TEXT, ">")
            next_button.click()
        except NoSuchElementException:
            logger.info("No more pages left.")
            break


def main():
    logger.info("Starting the main function...")
    driver = initialize_driver()

    # Read the last scraped date from MongoDB
    db = get_db()
    collection = db.societes
    last_article = collection.find_one(
        sort=[("Date", -1)])  # Get the latest article
    # Get the date of the latest article or empty string if no articles exist
    last_scraped_date = last_article["Date"] if last_article else ""

    with open('sociétés_bvmt.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Define the header row directly
        writer.writerow(["Date", "Title", "Content"])
        scrape_articles(driver, writer, last_scraped_date)

    driver.quit()

    with open('sociétés_bvmt.csv', 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        for row in reader:
            doc = {field: value for field, value in zip(header, row)}
            # Check if the article already exists in the collection
            if collection.count_documents(doc) > 0:
                logger.warning(
                    "Article already exists in the collection. Exiting...")
                print("Article already exists in the collection. Exiting...")
                return
            collection.insert_one(doc)
            logger.info("Inserted article into MongoDB.")
            print("Inserted article into MongoDB.")
        # Update the last scraped article's date
        last_scraped_date = row[0]

    logger.info("Main function completed successfully.")


if __name__ == "__main__":
    main()
