from pathlib import Path
import docx2txt
import re
import fitz
import torch
import os
import shutil

class Classifier:
    def __init__(self, path, targetPath):
        self.folder = Path(path)
        self.path = path
        self.model = torch.load("model.pth", weights_only=False)
        self.model.eval()
        self.targets = [(targetPath + "/" + dir) for dir in os.listdir(targetPath)]
    
    def docxscontents(self, path):
        return docx2txt.process(path)
    
    def extractTextFromPdf(self, path):
        doc = fitz.open(path)
        for page in doc:
            text += re.sub("\s+", " ", page.get_text()).strip()
        return text
    
    def extractText(self):
        listOfPDF = [str(file.resolve()) for file in self.folder.rglob("*.pdf")]
        listOfDocx = [str(file.resolve()) for file in self.folder.rglob("*.docx")]
        listOfText = []

        for pdf in listOfPDF:
            filePath = self.path + "/" + pdf
            listOfText.append((self.extractTextFromPdf(filePath), pdf))
        for docx in listOfDocx:
            filePath = self.path + "/" + docx
            listOfText.append((self.docxscontents(filePath), pdf))
        return listOfText
    
    def classify(self):
        try:
            documents = self.extractText()
            for document, path in documents:
                result = torch.argmax(self.model.forward(document))
                shutil.move(self.path + "/" + path, self.targets[result] + "/" + path)
            return True
        except:
            return False
        
        


