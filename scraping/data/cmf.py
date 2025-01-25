import os
import time
import urllib.request
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from bs4 import BeautifulSoup

def initialize_driver():
    options = webdriver.EdgeOptions()
    prefs = {
        "profile.managed_default_content_settings.images": 2,
        "profile.managed_default_content_settings.stylesheets": 2,
        "profile.managed_default_content_settings.fonts": 2
    }
    # options.add_argument('headless')
    options.add_experimental_option("prefs", prefs)

    driver = webdriver.Edge(options=options)
    return driver

def search_enterprise(driver, enterprise):
    # Select enterprise in dropdown
    dropdown = driver.find_element(By.CSS_SELECTOR, 'div.chosen-container-single')
    dropdown.click()
    
    # Wait until the dropdown options are visible
    WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.XPATH, '//div[@class="chosen-drop"]//li')))
    
    try:
        option = driver.find_element(By.XPATH, f'//div[@class="chosen-drop"]//li[text()="{enterprise}"]')
        option.click()
    except NoSuchElementException:
        print(f"No such element: {enterprise}")
        return

    # Click the search button
    search_button = driver.find_element(By.CSS_SELECTOR, 'input#edit-submit-consultation-des-tats-financier-des-soci-t-s-faisant-ape')
    search_button.click()
    time.sleep(3)  # Wait for the results to load

    # Get the page source after selecting the enterprise
    page_source = driver.page_source
    soup = BeautifulSoup(page_source, 'html.parser')

    # Extract PDF links and download
    pdf_rows = soup.select('.views-row')  # Selecting rows with PDFs
    for index, row in enumerate(pdf_rows, start=1):
        group_first_elem = row.find(class_='group-first')
        group_second_elem = row.find(class_='group-second')
        group_third_elem = row.find(class_='group-third')
        group_fourth_elem = row.find(class_='group-fourth')

        # Check if any of the elements are None
        if None in [group_first_elem, group_second_elem, group_third_elem, group_fourth_elem]:
            print(f"Some required elements are missing in row {index}. Skipping this PDF.")
            continue

        group_first = group_first_elem.text.strip()
        group_second = group_second_elem.text.strip()
        group_third = group_third_elem.text.strip()
        group_fourth = group_fourth_elem.text.strip()

        # Check if group_second is not in the specified years
        if group_second not in ['2019', '2020', '2021', '2022', '2023', '2024']:
            print(f"Skipping PDF in row {index} because group_second value ({group_second}) is not within the specified years.")
            continue

        pdf_link = row.select_one('.field-item.even a[href$=".pdf"]')
        if pdf_link:
            pdf_url = pdf_link['href']
            download_pdf(pdf_url, enterprise, group_first, group_second, group_third, group_fourth, index)




def download_pdf(url, enterprise, group_first, group_second, group_third, group_fourth, index):
    directory = f'pdfs/{enterprise}'
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Modify the filename to replace invalid characters
    sanitized_enterprise = ''.join(c if c.isalnum() or c in [' ', '_'] else '_' for c in enterprise)
    sanitized_group_first = ''.join(c if c.isalnum() or c in [' ', '_'] else '_' for c in group_first)
    sanitized_group_second = ''.join(c if c.isalnum() or c in [' ', '_'] else '_' for c in group_second)
    sanitized_group_third = ''.join(c if c.isalnum() or c in [' ', '_'] else '_' for c in group_third)
    sanitized_group_fourth = ''.join(c if c.isalnum() or c in [' ', '_'] else '_' for c in group_fourth)
    sanitized_file_name = f'{sanitized_enterprise}_{sanitized_group_first}_{sanitized_group_second}_{sanitized_group_third}_{sanitized_group_fourth}_pdf_{index}.pdf'
    file_path = os.path.join(directory, sanitized_file_name)

    try:
        # Download PDF
        urllib.request.urlretrieve(url, file_path)
        print(f'Downloaded {file_path}')
    except Exception as e:
        print(f'Error downloading {url}: {e}')

def main():
    # Initialize the Edge WebDriver
    driver = initialize_driver()

    start_url = 'https://www.cmf.tn/?q=consultation-des-tats-financier-des-soci-t-s-faisant-ape'

    # List of enterprises
    enterprises = ['BH BANK', 'BH ASSURANCE (EX ASSURNCES SALIM)', 'BH LEASING']

    # Loop through each enterprise
    for enterprise in enterprises:
        driver.get(start_url)
        search_enterprise(driver, enterprise)

    # Quit the driver after finishing scraping
    driver.quit()

if __name__ == "__main__":
    main()
