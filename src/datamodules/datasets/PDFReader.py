# First install PyPDF2 and pdfminer:
# pip install PyPDF2
# pip install pdfminer.six

import os, re, io
import PyPDF2
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser

class Page:
    def __init__(self, page_nr, title, destObj, level):
        self.page_nr = page_nr
        self.title = title
        self.destObj = destObj
        self.level = level
        self.next = None
    
    def get_values(self):
        return self.page_nr, self.title, self.destObj, self.level
    
    def __str__(self):
        return self.title

    def __repr__(self):
        return f"\n  Class: {self.__class__.__name__},\n  Title: {self.title},\n  Page:  {self.page_nr}\n  Level: {self.level}\n"

class Title(Page):
    def __init__(self, page_nr, title, destObj, level):
        super().__init__(page_nr, title, destObj, level)
        self.next = None
    
class Read_PDF:
    def __init__(self, file_name, path):
        self.file_path = os.path.join(path, file_name)
        self.reader = PyPDF2.PdfFileReader(self.file_path)
        self.pageObjects = []
        self.titleObject = []

    def _add_title(self, line, level=0):
        if type(line) == PyPDF2.generic.Destination:
            new_page = Page(self.reader.getDestinationPageNumber(line), line['/Title'], line, level)
            self.pageObjects.append(new_page)
        if type(line) == list:
            for l in line:
                self._add_title(l, level+1)
    
    def create_PageObjects(self):
        for line in self.reader.getOutlines():
            self._add_title(line, 0)

        for page, page_next in zip(self.pageObjects[:-1], self.pageObjects[1:]):
            page.next = page_next

        return self.pageObjects
    
    def create_TitleObjects(self):
        if self.pageObjects == []:
            self.create_PageObjects()
        
        self.titleObject = [Title(*page.get_values()) for page in self.pageObjects if page.level == 0]
        for title, title_next in zip(self.titleObject[:-1], self.titleObject[1:]):
            title.next = title_next

        return self.titleObject
    
    def get_header(self):
        pageObj = self.reader.getPage(0)
        first = pageObj.extractText().strip()

        first_cleaned = re.sub('\n', '', first)
        if len(re.split('[0-9]+ \\([0-9]+\\)', first_cleaned)) == 2:
            first_page = re.split('[0-9]+ \\([0-9]+\\)', first_cleaned)
        elif first.count('\n') == 1:
            first_page = first.split('\n')
        elif first.count(' \n ') == 1:
            first_page = first.split(' \n ')
        #elif re.sub('( ) +', '  ', first).count('  ') == 1:
        #    first = re.sub('( ) +', '  ', first)
        #    first_page = first.split('  ')
        else:
            first = re.sub('\n', '', first)
            first = re.sub(' +', ' ', first)
            break_point = re.search('V??gledning [0-9]+:[0-9]+ Version [0-9]+', first)
            if break_point == None:
                first_page = []
            else:
                break_point = break_point.end()
                first_page = [first[:break_point], first[break_point:]]

        assert len(first_page) >= 2, "Can't divide the first page into title and headnote"

        head = first_page[0]
        head = re.sub('\n', '', head)
        head = head.strip()

        title = first_page[1]
        title = re.sub('\n', '', title)
        title = title.strip()

        return head, title

class Clean_PDF(Read_PDF):
    def __init__(self, file_name, path):
        super().__init__(file_name, path)
        self.pageObjects = self.create_PageObjects()
        self.head, self.title = self.get_header()
        self.remove_pages = [self.title, "Inneh??ll", "F??rkortningar", "K??llf??rteckning"]    # this list can be changed
        self.remove_text = [self.head, '(\n+| +)?[0-9]+(\n+| +)?\\( ?[0-9]+ ?\\)(\n+| +)?', '\x0c']
        self.pagenumbers = []   #saved
        self.remove_before = 'Sammanfattning'
        self.remove_after = 'K??llf??rteckning'

    def _add_pagenumbers(self):
        titles = self.create_TitleObjects()

        remove_pagenumbers = set()
        if titles == []:
            A = 0
            B = self.reader.getNumPages()
            for i in range(B):
                text = self.reader.getPage(i).extractText()
                if self.remove_before in text[:100]:
                    A = i
                if self.remove_after in text[:100]:
                    B = i
            for i in range(A):
                remove_pagenumbers.add(i)
            for i in range(B, self.reader.getNumPages()):
                remove_pagenumbers.add(i)            
        else:
            for title in titles:
                if title.title in self.remove_pages:
                    start = title.page_nr
                    stop = self.reader.getNumPages() if title.next == None else title.next.page_nr
                    for i in range(start, stop):
                        remove_pagenumbers.add(i)
                if title.title == self.remove_before:
                    for i in range(title.page_nr):
                        remove_pagenumbers.add(i)
                if title.title == self.remove_after:
                    for i in range(title.page_nr, self.reader.getNumPages()):
                        remove_pagenumbers.add(i)

        for page_nr in range(self.reader.getNumPages()):
            if page_nr not in remove_pagenumbers:
                self.pagenumbers.append(page_nr)

    def _clean_sentence(self, sentence):
        for bad_text in self.remove_text:
            sentence = re.sub(bad_text, '', sentence)
        sentence = re.sub('( \n\n)( \n\n)+', '\n\n\n', sentence)
        while sentence != sentence.strip() or sentence != sentence.strip('\n'):
            sentence = sentence.strip()
            sentence = sentence.strip('\n')
        return sentence

    def _create_file(self):
        # a PdfFileWriter object to keep track of the pdf file we want to create:
        writer = PyPDF2.PdfFileWriter()
        for page_nr in self.pagenumbers:
            writer.addPage(self.reader.getPage(page_nr))     # add a page to the file to be created
        # save the contents in a file
        output_filename = 'pages_we_want_to_save.pdf'
        with open(output_filename, 'wb') as output:
            writer.write(output)

    def convert_pdf_to_string(self, file_path):
        """generic text extractor"""
        output_string = io.StringIO()
        with open(file_path, 'rb') as in_file:
            parser = PDFParser(in_file)
            doc = PDFDocument(parser)
            rsrcmgr = PDFResourceManager()
            device = TextConverter(rsrcmgr, output_string, laparams=LAParams())
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            for page in PDFPage.create_pages(doc):
                interpreter.process_page(page)
        return(output_string.getvalue())

    def clean_text(self):
        self._add_pagenumbers()
        self._create_file()
        new_txt = self.convert_pdf_to_string('pages_we_want_to_save.pdf')
        new_txt = self._clean_sentence(new_txt)
        return new_txt

    def run(self):
        return f"{self.title}\n\n\n{self.clean_text()}"


if __name__ == '__main__':
    #file_name = "vagledning-2004-05.pdf"
    #file_name = "vagledning-2021-01.pdf"
    #file_name = "vagledning-2018-04.pdf"
    #file_name = "vagledning-2018-03.pdf"
    #file_name = "vagledning-2018-02.pdf"
    #file_name = "vagledning-2018-01.pdf"
    #file_name = "vagledning-2017-02.pdf"
    #file_name = "vagledning-2017-01.pdf"
    #file_name = "Vagledning-2016-2.pdf"
    #file_name = "vagledning-2016-01.pdf"
    #file_name = "vagledning-2015-01.pdf"
    #file_name = "vagledning-2013-03.pdf"
    #file_name = "vagledning-2013-02.pdf"
    #file_name = "vagledning-2013-01.pdf"
    #file_name = "vagledning-2012-03.pdf"
    #file_name = "vagledning-2012-02.pdf"
    #file_name = "vagledning-2012-01.pdf"
    #file_name = "vagledning-2011-01.pdf"
    #file_name = "vagledning-2010-03.pdf"
    #file_name = "vagledning-2010-02.pdf"
    #file_name = "vagledning_2010_01.pdf"
    file_name = "vagledning-2009-02.pdf"
    file_name = "vagledning-2009-01.pdf"
    file_name = "vagledning-2008-02.pdf"
    file_name = "vagledning-2006-01.pdf"
    file_name = "vagledning-2005-03.pdf"
    file_name = "vagledning-2005-02.pdf"
    file_name = "vagledning-2005-01.pdf"
    file_name = "vagledning-2004-10.pdf"
    file_name = "vagledning-2004-08.pdf"
    file_name = "vagledning-2004-07.pdf"
    file_name = "vagledning-2004-06.pdf"
    file_name = "vagledning-2004-03.pdf"
    file_name = "vagledning-2004-01.pdf"
    file_name = "vagledning-2003-06.pdf"
    file_name = "vagledning-2003-04.pdf"
    file_name = "vagledning-2003-03.pdf"
    file_name = "vagledning-2003-01.pdf"
    file_name = "vagledning-2002-14.pdf"
    file_name = "vagledning-2002-12.pdf"
    file_name = "vagledning-2002-10.pdf"
    file_name = "vagledning-2002-06.pdf"
    file_name = "vagledning-2002-03.pdf"
    file_name = "vagledning-2002-01.pdf"
    file_name = "vagledning-2001-09.pdf"
    file_name = "vagledning-2001-07.pdf"
    file_name = "vagledning-2001-03.pdf"
    file_name = "vagledning-2001-02.pdf"
    path = os.path.dirname(os.path.abspath(__file__))

    cleaned_file = Clean_PDF(file_name, path)
    text = cleaned_file.run()
    
    # print some part of the text:
    print(text[:1000])

    # print the whole text:
    #print(text)

