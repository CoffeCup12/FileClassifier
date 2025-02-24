import docx2txt
import re
import fitz
import torch
import os
import shutil

class Classifier:
    def __init__(self, path, targetPath):
        self.path = path
        self.model = torch.load("model.pth", weights_only=False)
        self.model.eval()

        self.targets = [(targetPath + "/" + dir) for dir in os.listdir(targetPath) if dir != ".DS_Store"]
    
    def docxscontents(self, path):
        return docx2txt.process(path)
    
    def extractTextFromPdf(self, path):
        doc = fitz.open(path)
        text = ""
        for page in doc:
            text += re.sub(r"([. ])\1+", r"\1", page.get_text()).strip()
        return text
    
    def extractText(self):
        listOfPDF = [pdf for pdf in os.listdir(self.path) if pdf.endswith(".pdf") and pdf != ".DS_Store"]
        listOfDocx = [pdf for pdf in os.listdir(self.path) if pdf.endswith(".docx") and pdf != ".DS_Store"]
        listOfText = []

        for pdf in listOfPDF:
            listOfText.append((self.extractTextFromPdf(self.path + "/" + pdf), pdf))
        for docx in listOfDocx:
            try:
                listOfText.append((self.docxscontents(self.path + "/" + docx), docx))
            except:
                pass

        return listOfText
    
    def classify(self):
        # try:
            documents = self.extractText()
            for document, path in documents:
                # print(path)
                result = torch.argmax(self.model.forward(document))
                shutil.copy(self.path + "/" + path, self.targets[result] + "/" + path)
                print(result)
            return True
        # except:
            return False

        
        


