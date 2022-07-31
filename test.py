# import PyPDF2 
# import re
    
# # creating a pdf file object 
# pdfFileObj = open('/Users/cornelstefanache/Downloads/childrens-illustrated-dictionary.pdf', 'rb') 
    
# # creating a pdf reader object 
# pdfReader = PyPDF2.PdfFileReader(pdfFileObj) 
    
# # printing number of pages in pdf file 
# print(pdfReader.numPages) 
    
# # creating a page object 
# pageObj = pdfReader.getPage(13) 
# print('-----------------------------------------------------')
# # extracting text from page 

# start = False
# last_word = False

# word = None
# for line in pageObj.extractText().split('\n'):
#     if line == 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
#         start = True
#     else:
#         print(line)
#         # if re.search('^(\S+)$', line):
#         #     print('---------------------------')
#         #     last_word = True
#         #     word = line
#         #     print(word)
#         # elif last_word == True:
#         #     last_word = False
#         # else:
#         #     print(line)    
    



# # for content in pageObj.get_contents():
# #     print(content)
    
# # # closing the pdf file object 
# # pdfFileObj.close() 
file = '/Users/cornelstefanache/Downloads/childrens-illustrated-dictionary.pdf'
from io import StringIO

from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser

output_string = StringIO()
with open(file, 'rb') as in_file:
    parser = PDFParser(in_file)
    doc = PDFDocument(parser)
    rsrcmgr = PDFResourceManager()
    device = TextConverter(rsrcmgr, output_string, laparams=LAParams())
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    for page in PDFPage.create_pages(doc):
        interpreter.process_page(page)

with open('readme.txt', 'w') as f:
    f.write(output_string.getvalue())
