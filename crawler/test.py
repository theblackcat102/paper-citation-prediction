from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter, PDFPageAggregator
from pdfminer.layout import LAParams, LTFigure, LTImage
from pdfminer.pdfpage import PDFPage
from io import StringIO
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
import logging
import time
import arxiv
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_pdf_to_txt(path):
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
    fp = open(path, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    password = ""
    maxpages = 0
    caching = True
    pagenos=set()

    for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password,caching=caching, check_extractable=True):
        interpreter.process_page(page)

    text = retstr.getvalue()

    fp.close()
    device.close()
    retstr.close()
    return text


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
        print(pages)
        for thing in pdf_item:
            if isinstance(thing, LTFigure):
                print(thing.matrix)
                figures += find_images_in_thing(thing)
    return figures, pages


def find_images_in_thing(outer_layout):
    image_count = 0
    for thing in outer_layout:
        if isinstance(thing, LTImage):
            image_count += 1
    return image_count


def test_arvix():
    articles = arxiv.query(id_list=["1303.5778", "1503.02531"])
    for article in articles:
        print(article)

if __name__ == "__main__":
    # text = convert_pdf_to_txt('35eb0d2022282d44ffb30fde1f2b3dc481964b6a.pdf')
    # print(len(text))
    # print(get_figure_count('1.pdf'))
    test_arvix()