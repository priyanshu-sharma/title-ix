import logging
import re
import requests
import PyPDF2
import docx2txt
from io import BytesIO
import pandas as pd
from bs4 import BeautifulSoup
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Crawl:
    def __init__(self, csv_path):
        self.headers = {}
        self.csv_data = pd.read_csv(csv_path)
        logger.info("columns - {}".format(self.csv_data.columns))
        logger.info(self.csv_data.head())
        self.url_map = {
            'https://www.twc.texas.gov/sites/default/files/vr/docs/title-ix-procedure-manual-twc.docx': 'docs/texas_6.docx'
        }
        self.starting_state = self.csv_data['state'][0]
        self.formatted_data = self.title_and_info(self.csv_data['state'][0], self.csv_data['color'][0])
        print(self.formatted_data)
        self.initial_patterns = ['Title IX', 'Commission', 'regulations', 'Sex', 'Rights', 'Discrimination', 'Law', 'Harassment', 'Policy']
        self.match_patterns = self.initialize_patterns()
        self.crawl()

    def initialize_patterns(self):
        final_patterns = self.initial_patterns
        for i in range(0, len(self.initial_patterns)):
            final_patterns.append(self.initial_patterns[i].lower())
        return final_patterns

    def ismatch_title_link_map(self, s):
        for i in self.match_patterns:
            if re.search(i, s):
                return True
        return False

    def clean_and_add_data(self, text_data):
        text_list = text_data.split("\n")
        pre_formatted_list = []
        total_len = 0
        for text in text_list:
            if len(text) > 1 and not text.isspace():
                preformatted_text = text.strip()
                total_len = total_len + len(preformatted_text)
                pre_formatted_list.append(preformatted_text)
        data = ''
        average = total_len/len(pre_formatted_list)
        logger.info("Total len - {}, Total words - {}, Average - {}".format(total_len, len(text_list), average))
        # logger.info("Pre formatted - {}".format(pre_formatted_list))
        icount = 0
        for preformatted_data in pre_formatted_list:
            if len(preformatted_data) > average:
                icount = icount + 1
                data = data + preformatted_data + '\n'
        logger.info("Rejecting around due to average criteria - {}".format(len(pre_formatted_list) - icount))
        return data

    def get_child_crawled_data(self, child_url, child_data):
        child_data = child_data + '\n'
        if child_url[-4:] == '.pdf' or '.pdf' in child_url:
            raw_child_data = requests.get(child_url).content
            with BytesIO(raw_child_data) as data:
                read_pdf = PyPDF2.PdfFileReader(data)
                for page in range(read_pdf.getNumPages()):
                    child_data = child_data + read_pdf.getPage(page).extractText()
            return child_data, {child_url: 'empty'}
        else:    
            child_html_page = requests.get(child_url).text
            soup = BeautifulSoup(child_html_page, 'html.parser')
            child_header = soup.find_all(["title", "h1", "h2", "h3"])
            child_text_data = soup.text.strip()
            formatted_child_data = self.clean_and_add_data(child_text_data)
            child_data = child_data + formatted_child_data
            # logger.info("Formatted Child Data - {}".format(child_data))
            return child_data, {child_url : str(child_header)}

    def reject_unwanted_links(self):
        filtered_links = []
        for links in self.child_links:
            if self.ismatch_title_link_map(links.text.strip()):
                filtered_links.append(links)
        self.child_links = filtered_links

    def get_child_link_data(self, link_list):
        children_data = ''
        self.child_links = []
        child_headers = []
        for link in link_list:
            if link.get('href') is not None and link.get('class') is None and link['href'][:4] == 'http':
                self.child_links.append(link)
        logger.info("Rejecting around links - {}".format(len(link_list) - len(self.child_links)))
        logger.info("Remaining Child links - {}".format(len(self.child_links)))
        if len(self.child_links) > 20:
            self.reject_unwanted_links()
            logger.info("Rejecting Unwanted Child links - {}".format(len(self.child_links)))
        skip_links = ['https://disclosures.utah.gov/Search/PublicSearch?type=PCC', 'https://image.le.utah.gov/imaging/History.asp', 'https://image.le.utah.gov/imaging/bill.asp', 'https://lag.utleg.gov/audits_current.jsp', 'https://lag.utleg.gov/', 'https://lag.utleg.gov/annual_report.jsp', 'https://lag.utleg.gov/best_practices.jsp']
        for child_link in self.child_links:
            child_url = child_link.get('href')
            if child_url not in skip_links:
                child_topic = child_link.text.strip()
                logger.info("{} --------------------------------- {}".format(child_url, child_topic))
                try:
                    child_data, child_header = self.get_child_crawled_data(child_url, child_topic)
                    children_data = children_data + child_data
                    child_headers.append(child_header)
                except Exception as e:
                    print(e)
        return children_data, child_headers

    def title_and_info(self, state, color):
        if state == 'federal':
            format_header = "TITLE - {} TITLE IX DOCUMENTATION\n\n (This is crawled data)\n\n".format(state.upper())
        else:
            party = 'Democratic' if color == 'Blue' else 'Republic'
            format_header = "TITLE - {} TITLE IX DOCUMENTATION\n\n{} State ({} Region) (This is crawled data)\n\n".format(state.upper(), party, color)
        return format_header


    def crawl(self):
        for index, row in self.csv_data.iterrows():
            print(row['type'])
            if row['state'] != self.starting_state:
                with open('../output_domain/federal/{}.txt'.format(self.starting_state), "w", encoding="utf-8") as f:
                    f.write(self.formatted_data)
                self.starting_state = row['state']
                self.formatted_data = self.title_and_info(row['state'], row['color'])
            if row['type'] == 'pdf':
                raw_child_data = requests.get(row['url']).content
                with BytesIO(raw_child_data) as data:
                    read_pdf = PyPDF2.PdfFileReader(data)
                    for page in range(read_pdf.getNumPages()):
                        self.formatted_data = self.formatted_data + read_pdf.getPage(page).extractText()
                main_headers = [{row['url'] : 'empty'}]
                child_headers = []
            elif row['type'] == 'docs':
                file = self.url_map[row['url']]
                self.formatted_data = self.formatted_data + docx2txt.process(file)
                main_headers = [{row['url'] : 'empty'}]
                child_headers = []
            else:
                html_page = requests.get(row['url']).text
                soup = BeautifulSoup(html_page, 'html.parser')
                temp_headers = soup.find_all(["title", "h1", "h2", "h3"])
                main_headers = [{row['url'] : str(temp_headers)}]
                text_data = soup.text.strip()
                format_data = self.clean_and_add_data(text_data)
                link_list = soup.find_all("a")
                child_data, child_headers = self.get_child_link_data(link_list)
                self.formatted_data = self.formatted_data + format_data + child_data
            print(main_headers, child_headers)
            if self.starting_state not in self.headers.keys():
                self.headers[self.starting_state] = {
                        'headers': main_headers,
                        'child_headers': child_headers
                }
            else:
                self.headers[self.starting_state]['headers'] = self.headers[self.starting_state]['headers'] + main_headers
                self.headers[self.starting_state]['child_headers'] = self.headers[self.starting_state]['child_headers'] + child_headers
        # logger.info("Formatted Child Data - {}".format(self.formatted_data))
        with open('../output_domain/federal/{}.txt'.format(self.starting_state), "w", encoding="utf-8") as f:
            f.write(self.formatted_data)
        with open("headers.json", "w") as f:
            json.dump(self.headers, f)

csv_path = 'test_data.csv'
crawl = Crawl(csv_path)