from pypdf import PdfReader 
from docx import Document

class Reader:
    def __init__(self):
        pass 

    def read_pdf(self,dir):
        text = "unclassified"
        try:
            reader = PdfReader(dir)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
        except:
            print(f"Warning: can't open file {dir}")
            
        return text
    
    def read_docx(self, dir):
        text = "unclassified"
        try:
            doc = Document(dir)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text
        except:
            print(f"Warning: can't open file {dir}")
        return text
    
    def read(self, dir):
        text = "unclassified"
        ending = dir[-4:]
        if ending == ".pdf":
            text = self.read_pdf(dir)
        elif ending == "docx":
            text = self.read_docx(dir)
        else:
            print(f"Warning: unsupported file type {dir}")

        if text == "":
            text = "unclassified"
            print(f"Warning: error reading file {dir}")

        return text

