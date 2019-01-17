from models import Author, Paper
import arxiv
import tabula
import urllib
import urllib.request
import os, glob, sys, re
from tqdm import tqdm
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter, PDFPageAggregator
from pdfminer.layout import LAParams, LTFigure, LTImage
from pdfminer.pdfpage import PDFPage
from io import StringIO
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from tabula import wrapper
from pdf_processor import extract_text, extract_text2
import logging
import time
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# the path where you want to store the downloaded pdf file
store_path = './pdf_files/'

def find_images_in_thing(outer_layout):
    image_count = 0
    for thing in outer_layout:
        if isinstance(thing, LTImage):
            image_count += 1
    return image_count

def get_figure_count(path):
    fp = open(path, 'rb')
    parser = PDFParser(fp)
    document = PDFDocument(parser)
    rsrcmgr = PDFResourceManager()
    laparams = LAParams()
    device = PDFPageAggregator(rsrcmgr, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)

    figures = 0
    pages = 0
    for page in PDFPage.create_pages(document):
        interpreter.process_page(page)
        pages += 1
        pdf_item = device.get_result()
        for thing in pdf_item:
            if isinstance(thing, LTImage):
                figures += 1
            if isinstance(thing, LTFigure):
                figures += find_images_in_thing(thing)
    return figures, pages


def download_extract(paper, extract_figure=False, extract_table=False):
    if paper.pages >= 0 and paper.table >= 0:
        return False
    paper_info = {
        'pdf_url': paper.url,
        'title': paper.title,
    }
    api_paper = arxiv.query(id_list=[paper.arvixID])[0]
    if 'pdf_url' not in api_paper:
        return False
    pdf_url = api_paper['pdf_url']
    # pdf_url = 'https://arxiv.org/pdf/' + paper.url.split('/')[-1] +'.pdf'
    file_path = os.path.join(store_path, paper.paperId+'.pdf')
    # if not os.path.isfile(file_path):
    urllib.request.urlretrieve(pdf_url, file_path)

    if extract_table:
        df = wrapper.read_pdf(file_path, multiple_tables=True, pages='all')    
        table_count = len(df)
        del df

    if extract_figure:
        figure_count, page_count = get_figure_count(file_path)
        modified = False
        if paper.pages == -1:
            modified = True
            paper.pages = page_count
        else:
            page_count = paper.pages
        if paper.table == -1:
            modified = True
            paper.table = table_count
        if os.path.exists(file_path):
            os.remove(file_path)
        if modified:
            Paper.update(table=table_count, pages=page_count).where(Paper.arvixID == paper.arvixID).execute()
            # paper.save()
            return modified
    # api_paper = arxiv.query(id_list=[paper.arvixID])[0]
    # if 'pdf_url' not in api_paper:
    #     return False
    # pdf_url = api_paper['pdf_url']
    texts = extract_text(file_path, pdf_url)
    if texts is None:
        print("PDF either do not exists or failed : ", paper.url)
        return False
    affiliation = []
    for text in texts.split():
        if re.match("[^@]+@[^@]+\.[^@]+", text):
            domain_name = text.split('@')[-1]
            affiliation.append(domain_name)
    if len(affiliation) > 0:
        Paper.update(affiliation=affiliation).where(Paper.arvixID == paper.arvixID).execute()

    return False



def main():
    papers = Paper.select().where((Paper.year == 2015))
    for paper in tqdm(papers):
        try:
            download_extract(paper)
        except KeyboardInterrupt:
            sys.exit()
        # except:
        #     logging.warning("\Failed in : %s" % (str(paper.paperId)))
    # papers = Paper.select().where(Paper.year == 2017)
    # for paper in tqdm(papers):
    #     try:
    #         download_extract(paper)
    #     except KeyboardInterrupt:
    #         sys.exit()
    #     except:
    #         logging.warning("\Failed in : %s" % (str(paper.paperId)))

if __name__ == "__main__":
    # papers = Paper.select().where(Paper.arvixID == '1705.09871')
    # for paper in tqdm(papers):
    #     try:
    #         changed = download_extract(paper)
    #         print(changed)
    #     except KeyboardInterrupt:
    #         sys.exit()
    main()
