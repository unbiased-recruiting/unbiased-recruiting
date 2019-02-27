import PyPDF2

def convert_pdf_to_str(path):
    pdfFileObj = open(path,'rb')
    pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
    converted=""
    for i in range(pdfReader.numPages):
        pageObj=pdfReader.getPage(i)
        converted=converted+pageObj.extractText()
    return converted

print(convert_pdf_to_str("../cv_100.pdf"))