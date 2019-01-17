import os
import sys
import re
import urllib
import urllib.request
import time
import PyPDF2
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter  # process_pdf
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from io import StringIO


def pdf_to_text(pdfname):

	# PDFMiner boilerplate
	rsrcmgr = PDFResourceManager()
	sio = StringIO()
	codec = 'utf-8'
	laparams = LAParams()
	device = TextConverter(rsrcmgr, sio, codec=codec, laparams=laparams)
	interpreter = PDFPageInterpreter(rsrcmgr, device)

	# Extract text
	fp = open(pdfname, 'rb')
	for idx, page in enumerate(PDFPage.get_pages(fp)):
		if idx > 2:
			break
		interpreter.process_page(page)
	fp.close()

	# Get text from StringIO
	text = sio.getvalue()

	# Cleanup
	device.close()
	sio.close()

	return text


def getPageCount(pdf_file):

	pdfFileObj = open(pdf_file, 'rb')
	pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
	pages = pdfReader.numPages
	return pages

def extractData(pdf_file, page):
	pdfFileObj = open(pdf_file, 'rb')
	pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
	pageObj = pdfReader.getPage(page)
	data = pageObj.extractText()
	return data

def getWordCount(data):
	data=data.split()
	return len(data)

def fixPdf(pdfFile):
	try:
		fileOpen = open(pdfFile, "a")
		fileOpen.write("%%EOF")
		fileOpen.close()
		return "Fixed"
	except Exception as e:
		return "Unable to open file: %s with error: %s" % (pdfFile, str(e))

def extract_text(pdf_path, pdf_url, page_num=0):
	read_pdf = None
	try:
		pdf_file = open(pdf_path, 'rb')
		read_pdf = PyPDF2.PdfFileReader(pdf_file)
	except:
		try:
			urllib.request.urlretrieve(pdf_url, file_path)
			fixPdf(pdf_path)
			pdf_file = open(pdf_path, 'rb')
			read_pdf = PyPDF2.PdfFileReader(pdf_file)
		except:
			return None

	if read_pdf is not None:	
		try:
			number_of_pages = read_pdf.getNumPages()
			page = read_pdf.getPage(page_num)
			page_content = page.extractText()
			return page_content.encode('utf-8').decode("utf-8") 
		except:
			print("Parsing failed")
	return None

def extract_text2(pdf_path, pdf_url, page_num=0):
	read_pdf = None
	try:
		texts = pdf_to_text(pdf_path)
		return texts
	except:
		fixPdf(pdf_path)
		texts = pdf_to_text(pdf_path)
		return texts

if __name__ == "__main__":
	from email.utils import parseaddr
	import re, os, glob
	for filename in glob.glob("./*.pdf"):
		texts = extract_text(filename, 0)
		for text in texts.split():
			if re.match("[^@]+@[^@]+\.[^@]+", text):
				print(text)
		# result = parseaddr(str(text))
		# if '@' in result[1]:
		# 	print(result)
