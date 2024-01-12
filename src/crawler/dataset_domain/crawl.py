import logging
import requests
import PyPDF2
from io import BytesIO
import pandas as pd
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Crawl:
    def __init__(self, csv_path):
        self.csv_data = pd.read_csv(csv_path)
        logger.info("columns - {}".format(self.csv_data.columns))
        logger.info(self.csv_data.head())
        self.crawl()

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
        else:    
            child_html_page = requests.get(child_url).text
            soup = BeautifulSoup(child_html_page, 'html.parser')
            child_text_data = soup.text.strip()
            formatted_child_data = self.clean_and_add_data(child_text_data)
            child_data = child_data + formatted_child_data
        # logger.info("Formatted Child Data - {}".format(child_data))
        return child_data

    def get_child_link_data(self, link_list):
        children_data = ''
        self.child_links = []
        for link in link_list:
            if link.get('href') is not None and link.get('class') is None and link['href'][:4] == 'http':
                self.child_links.append(link)
        logger.info("Rejecting around links - {}".format(len(link_list) - len(self.child_links)))
        logger.info("Remaining Child links - {}".format(len(self.child_links)))
        skip_links = ['https://disclosures.utah.gov/Search/PublicSearch?type=PCC', 'https://image.le.utah.gov/imaging/History.asp', 'https://image.le.utah.gov/imaging/bill.asp', 'https://lag.utleg.gov/audits_current.jsp', 'https://lag.utleg.gov/', 'https://lag.utleg.gov/annual_report.jsp', 'https://lag.utleg.gov/best_practices.jsp']
        for child_link in self.child_links:
            child_url = child_link.get('href')
            if child_url not in skip_links:
                child_topic = child_link.text.strip()
                logger.info("{} --------------------------------- {}".format(child_url, child_topic))
                try:
                    child_data = self.get_child_crawled_data(child_url, child_topic)
                    children_data = children_data + child_data
                except Exception as e:
                    print(e)
        return children_data

    def crawl(self):
        for index, row in self.csv_data.iterrows():
            if row['type'] == 'pdf':
                formatted_data = ''
                raw_child_data = requests.get(row['url']).content
                with BytesIO(raw_child_data) as data:
                    read_pdf = PyPDF2.PdfFileReader(data)
                    for page in range(read_pdf.getNumPages()):
                        formatted_data = formatted_data + read_pdf.getPage(page).extractText()
                logger.info("Formatted Child Data - {}".format(formatted_data))
            elif row['type'] == 'docs':
                continue
            else:
                html_page = requests.get(row['url']).text
                soup = BeautifulSoup(html_page, 'html.parser')
                text_data = soup.text.strip()
                formatted_data = self.clean_and_add_data(text_data)
                link_list = soup.find_all("a")
                child_data = self.get_child_link_data(link_list)
                formatted_data = formatted_data + child_data
            with open('output/{}_{}.txt'.format(row['state'], row['id']), "w", encoding="utf-8") as f:
                f.write(formatted_data)

csv_path = 'test_data.csv'
crawl = Crawl(csv_path)