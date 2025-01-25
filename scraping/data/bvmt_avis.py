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


def scrape_articles(driver, writer):  # Add the 'writer' argument
    # Navigate to the BVMT website
    driver.get("https://www.bvmt.com.tn/fr/avis-decisions")

    while True:
        # Find all articles
        articles = driver.find_elements(By.CSS_SELECTOR, "#actus-list > div")

        for i in range(len(articles)):
            try:
                articles = driver.find_elements(
                    By.CSS_SELECTOR, "#actus-list > div")

                article_url = articles[i].find_element(
                    By.XPATH, "./a").get_attribute("href")
                date = articles[i].find_element(
                    By.CSS_SELECTOR, "#list-actu01 > p > span.orange").text

                # Navigate to the URL of the article
                driver.get(article_url)

                # Wait for the article content to be visible
                WebDriverWait(driver, 10).until(
                    EC.visibility_of_element_located((By.CSS_SELECTOR, "#bl-01")))

                # Re-locate the elements before interacting with them
                # date = driver.find_element(By.CSS_SELECTOR, "#bl-01 > p:nth-child(3)").text
                title = driver.find_element(
                    By.CSS_SELECTOR, "#bl-01 > h3").text
                content = driver.find_element(By.CSS_SELECTOR, "#bl-01").text

                # Write the scraped data to the CSV file
                writer.writerow([date, title, content])

            except NoSuchElementException:
                print("Element not found. Skipping...")
                continue

            finally:
                # Navigate back to the list of articles
                driver.back()
                WebDriverWait(driver, 10).until(EC.visibility_of_element_located(
                    (By.CSS_SELECTOR, "#actus-list > div")))

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
    with open('avis_bvmt.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Define the header row directly
        writer.writerow(["Date", "Title", "Content"])
        scrape_articles(driver, writer)  # Pass the CSV writer to the function

    # Close the webdriver
    driver.quit()

    # Now, you can read from the CSV file
    with open('avis_bvmt.csv', 'r') as file:
        reader = csv.reader(file)
        # Define the header as the first row of the CSV file
        header = next(reader)
        # Get a reference to the database
        db = get_db()
        # Get a reference to your collection
        collection = db.avis  # Replace 'trading' with 'bvmt1'
        # Insert each row of the CSV file into the collection
        for row in reader:
            doc = {field: value for field, value in zip(header, row)}
            collection.insert_one(doc)


if __name__ == "__main__":
    main()
