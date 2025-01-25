import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import csv
import time
from db_connection import get_db


def initialize_driver():
    options = webdriver.EdgeOptions()
    # Use headless mode to not open a browser window
    options.add_argument('--headless')
    prefs = {"profile.managed_default_content_settings.images": 2}
    options.add_experimental_option("prefs", prefs)
    driver = webdriver.Edge(options=options)
    return driver


def scrape_articles(driver, writer):
    page_number = 1

    while True:
        try:
            url = f'https://www.tap.info.tn/en/portal%20-%20economy?pg={page_number}'
            # Print the URL of the page being accessed
            print(f"Accessing page: {url}")
            driver.get(url)
            WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located(
                (By.CSS_SELECTOR, "#ctl09_ctl00_MAIN_TABLE .NewsItemHeadline a")))

            article_links = driver.find_elements(
                By.CSS_SELECTOR, "#ctl09_ctl00_MAIN_TABLE .NewsItemHeadline a")

            if not article_links:
                print("No more articles found. Exiting...")
                break

            for i, article_link in enumerate(article_links):
                try:
                    article_url = article_link.get_attribute("href")
                    # Print the URL of the article being scraped
                    print(
                        f"Scraping article {i+1} of {len(article_links)} on page {page_number}: {article_url}")
                    driver.execute_script(
                        "window.open(arguments[0]);", article_url)
                    # Switch to the new tab
                    driver.switch_to.window(driver.window_handles[-1])

                    # Handle pop-up advertisement
                    try:
                        WebDriverWait(driver, 20).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, ".close-button")))
                        driver.find_element(
                            By.CSS_SELECTOR, ".close-button").click()
                    except TimeoutException:
                        pass

                    date = driver.find_element(
                        By.CSS_SELECTOR, ".NewsItemText").text.split(',')[0]
                    title = driver.find_element(
                        By.CSS_SELECTOR, ".NewsItemHeadline").text
                    content = driver.find_element(
                        By.CSS_SELECTOR, ".NewsItemTextForArticle").text

                    writer.writerow([date, title, content])
                    # Print a success message after each article is scraped
                    print(
                        f"Successfully scraped article {i+1} of {len(article_links)} on page {page_number}")

                except Exception as e:
                    print(f"Error: {e}. Skipping article...")
                    continue

                driver.close()  # Close the current tab
                # Switch back to the main tab
                driver.switch_to.window(driver.window_handles[0])

            page_number += 1

        except TimeoutException:
            print("Timeout. No more pages left.")
            break


def main():
    driver = initialize_driver()

    # Scrape articles and write them into a CSV file
    with open('tap.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Define the header row directly
        writer.writerow(["Date", "Title", "Content"])
        scrape_articles(driver, writer)  # Pass the CSV writer to the function

    # Close the webdriver
    driver.quit()

    # Now, you can read from the CSV file
    with open('tap.csv', 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        # Define the header as the first row of the CSV file
        header = next(reader)
        # Get a reference to the database
        db = get_db()
        # Get a reference to your collection
        collection = db.tap  # Replace with your collection name
        # Insert each row of the CSV file into the collection
        for row in reader:
            doc = {field: value for field, value in zip(header, row)}
            collection.insert_one(doc)

    driver.quit()


if __name__ == "__main__":
    main()
