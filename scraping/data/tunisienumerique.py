import re
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException, NoSuchElementException
import csv
import time
from pymongo import DESCENDING


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


def handle_popup(driver):
    try:
        popup_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CLASS_NAME, 'onesignal-popover-cancel-button')))
        popup_button.click()
    except TimeoutException:
        pass


def clean_date(date_str):
    if date_str.strip().lower() == 'date':
        return None

    # Extracting only the time information from the date string
    time_info = re.search(r'Il y a (\d+) heure', date_str, re.IGNORECASE)
    if time_info:  # If the pattern is matched
        hours_ago = int(time_info.group(1))  # Extract the number of hours ago
        # Subtract the hours from the current datetime
        date = datetime.now() - timedelta(hours=hours_ago)
        return date.strftime('%d %B %Y %H:%M')
    else:
        # Try to match the pattern like "| 19 Avril 2024 à 20:19"
        custom_date_info = re.search(
            r'\| (\d+) (\w+) (\d+) à (\d+):(\d+)', date_str)
        if custom_date_info:
            day = int(custom_date_info.group(1))
            month_str = custom_date_info.group(2)
            year = int(custom_date_info.group(3))
            hour = int(custom_date_info.group(4))
            minute = int(custom_date_info.group(5))

            # Convert month name to its corresponding number
            month_number = {
                "janvier": 1, "février": 2, "mars": 3, "avril": 4,
                "mai": 5, "juin": 6, "juillet": 7, "août": 8,
                "septembre": 9, "octobre": 10, "novembre": 11, "décembre": 12
            }[month_str.lower()]

            date = datetime(year, month_number, day, hour, minute)
            return date.strftime('%d %B %Y %H:%M')
        else:
            return date_str  # Return the original date string as it is


def get_last_date_from_collection(collection):
    last_date_document = collection.find_one({}, sort=[('Date', DESCENDING)])
    if last_date_document:
        last_date_str = last_date_document['Date']
        return clean_date(last_date_str)
    else:
        return None


def scrape_articles(driver, collection, csv_file_path):
    scraped_urls = set()
    page_number = 1
    last_date_str = get_last_date_from_collection(collection)
    last_date = None

    if last_date_str:
        last_date = datetime.strptime(
            last_date_str, '%d %B %Y %H:%M')  # Update format string

    while True:
        print(f"Scraping page {page_number}")
        driver.get(
            f"https://www.tunisienumerique.com/actualite-tunisie/economie/page/{page_number}/")
        time.sleep(10)
        handle_popup(driver)

        try:
            WebDriverWait(driver, 30).until(EC.visibility_of_all_elements_located(
                (By.CSS_SELECTOR, 'ul.archive-col-list li.infinite-post a')))

            article_links = driver.find_elements(
                By.CSS_SELECTOR, 'ul.archive-col-list li.infinite-post a')

            print(f"Found {len(article_links)} article(s) on this page.")

            for link in article_links:
                try:
                    article_url = link.get_attribute("href")

                    if collection.find_one({"URL": article_url}):
                        print(
                            f"Article already exists in the collection: {article_url}")
                        return  # Stop scraping process if article already exists

                    print("Article URL:", article_url)

                    driver.execute_script("window.open('');")
                    driver.switch_to.window(driver.window_handles[1])
                    driver.get(article_url)

                    WebDriverWait(driver, 30).until(
                        EC.visibility_of_element_located((By.CSS_SELECTOR, 'h1.entry-title')))

                    title = driver.find_element(
                        By.CSS_SELECTOR, 'h1.entry-title').text
                    content = driver.find_element(
                        By.CSS_SELECTOR, 'div.post-cont-out').text
                    date_str = driver.find_element(
                        By.CSS_SELECTOR, 'time.post-date').text
                    # Print original date string
                    print(f"Original date string from article: {date_str}")

                    date = clean_date(date_str)

                    if date is None:
                        continue

                    date_obj = datetime.strptime(date, '%d %B %Y %H:%M')

                    if last_date is None or date_obj > last_date:
                        doc = {"Title": title, "Content": content,
                               "Date": date, "URL": article_url}
                        collection.insert_one(doc)

                        with open(csv_file_path, 'a', newline='', encoding='utf-8') as file:
                            writer = csv.writer(file)
                            writer.writerow(
                                [date, title, content, article_url])

                    scraped_urls.add(article_url)

                    driver.close()
                    driver.switch_to.window(driver.window_handles[0])
                    time.sleep(2)
                except (StaleElementReferenceException, TimeoutException) as e:
                    print(f"Error scraping article: {str(e)}")
                    driver.refresh()
                    handle_popup(driver)
                    continue
                except NoSuchElementException as e:
                    print(f"Element not found: {str(e)}")
                    continue

        except TimeoutException as e:
            print(f"Timeout waiting for elements: {str(e)}")
            break
        except Exception as e:
            print("Error during scraping:", str(e))
            break

        page_number += 1


"""def main():
    csv_file_path = 'tunisienumerique_articles.csv'

    driver = initialize_driver()
    db = get_db()
    collection = db.tnumeco

    last_date_str = get_last_date_from_collection(collection)
    if last_date_str:
        print("Last date from collection:", last_date_str)
    else:
        print("No previous date found in collection.")

    with open(csv_file_path, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Date", "Title", "Content", "URL"])

    scrape_articles(driver, collection, csv_file_path)

    driver.quit()


if __name__ == "__main__":
    main()

"""
