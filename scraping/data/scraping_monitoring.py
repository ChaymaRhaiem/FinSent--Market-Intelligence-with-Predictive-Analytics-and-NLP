import streamlit as st
import logging
import os
from datetime import datetime



def monitoring_page():
    st.title('Scraping Monitoring Dashboard')

    # Display the status of scraping jobs
    st.subheader('Scraping Jobs Status')
    log_file_path = os.path.join(os.path.dirname(__file__), '..', 'logs', 'scraping.log')
    if os.path.exists(log_file_path):
        # Read log file
        with open(log_file_path, 'r') as log_file:
            logs = log_file.readlines()

            # Initialize a dictionary to store log entries by date
            log_entries_by_date = {}

        for log in logs:
            log_entry = log.strip().split(' - ')
            
            # Check if the log entry has at least three elements
            if len(log_entry) >= 3:
                # Extract log level, timestamp, and message
                log_level = log_entry[1].strip()
                timestamp = log_entry[0].strip()
                message = log_entry[2].strip()

                # Parse timestamp to datetime object
                timestamp = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')

                # Check if the log message contains the specific keywords
                keywords = ["Scraping finished at:", "Scraping duration", "Total Tunisie Numerique articles scraped", "Total BVMT articles scraped", "Total failed scrapes"]
                if any(keyword in message for keyword in keywords):
                    # Get the date from the timestamp
                    date = timestamp.date()

                    # Add log entry to the dictionary
                    if date not in log_entries_by_date:
                        log_entries_by_date[date] = []
                    log_entries_by_date[date].append((timestamp, log_level, message))

        # Sort log entries by date and display them
        for date, log_entries in sorted(log_entries_by_date.items()):
            st.subheader(f"Date: {date}")
            for log_entry in log_entries:
                timestamp_str = log_entry[0].strftime('%Y-%m-%d %H:%M:%S')
                log_message = f"{timestamp_str} - [{log_entry[1]}] - {log_entry[2]}"
                st.text(log_message)

    else:
        st.write("No log file found.")

    # Display health tracking information
    st.subheader('Health Tracking of Scraping')
    st.write("Add code to display the health tracking information of scraping here.")
