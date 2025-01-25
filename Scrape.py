from selenium import webdriver
import time
import pandas as pd
from selenium.webdriver.common.by import By
from io import StringIO
import schedule


def initialize_driver():
    options = webdriver.ChromeOptions()
    # Use headless mode to not open a browser window
    options.add_argument('--headless')
    driver = webdriver.Chrome(options=options)
    return driver


def scrape_dynamic_content():
    # Initialize the WebDriver
    driver = initialize_driver()

    try:
        url = "https://www.ilboursa.com/marches/aaz"

        # Navigate to the URL
        driver.get(url)

        # Wait for the dynamic content to load
        time.sleep(3)  # Adjust the sleep time as necessary

        # Now you can locate the elements within the iframe as usual
        # For example, to scrape a table:
        table_html = driver.find_element(
            By.CSS_SELECTOR, '#tabQuotes').get_attribute('outerHTML')

        # Convert the table HTML to a pandas DataFrame
        # Use StringIO to wrap the HTML content
        df_list = pd.read_html(StringIO(table_html))

        # Assuming there's only one DataFrame in the list
        if df_list:
            df = df_list[0]
            # Do something with the data, e.g., save to Excel
            # Use to_csv() method to save as CSV
            df.to_csv("Data/output.csv", index=False)
        else:
            print("No table found on the page.")

    finally:
        # Clean up
        driver.quit()


# Schedule the scraping task to run every hour
schedule.every().minutes.do(scrape_dynamic_content)

# Run the scheduler indefinitely
while True:
    schedule.run_pending()
    time.sleep(1)  # Sleep for 1 second to avoid high CPU usage
