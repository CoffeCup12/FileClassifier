from pathlib import Path
import docx2txt
import re
import fitz
import os

class Extractor:
    def __init__(self, path):
        self.path = path

    def showPath(self):
        print(f"Path: {self.path}")

    def pdfsInFolder(self):
        folder = Path(self.path)
        return [str(file.resolve()) for file in folder.rglob("*.pdf")]
    
    def docxsInFolder(self):
        folder = Path(self.path)
        return [str(file.resolve()) for file in folder.rglob("*.docx")]
    
    def docxscontents(self, path):
        return docx2txt.process(path)
    
    def extractTextFromPdf(self, path):
        doc = fitz.open(path)
        for page in doc:
            text += re.sub("\s+", " ", page.get_text()).strip()
        return text
    
    def extractText(self):
        listOfPDF = self.pdfsInFolder()
        listOfDocx = self.docxsInFolder()
        listOfText = []
        
        for pdf in listOfPDF:
            filePath = self.path + "/" + pdf
            listOfText.append(self.extractTextFromPdf(filePath), filePath)
        for docx in listOfDocx:
            filePath = self.path + "/" + docx
            listOfText.append(self.docxscontents(filePath), filePath)
        return listOfText
            







if __name__ == "__main__":
    target_folder = input("Enter the folder: ").strip()
    extractor1 = Extractor(target_folder)
    print("PDF:")
    print(extractor1.pdfsInFolder())
    print("\nDOCX:")
    print(extractor1.docxsInFolder())
    print("\nDOCX_content:")
    print(extractor1.docxscontents())