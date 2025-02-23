import model
import torch.optim as optim
import torch
import re
import os
import fitz
import docx2txt
import random

class trainer():

    def __init__(self, path):
        self.path = path
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        self.punctuation = ['.', ',', '!', '?', ':', ';', '(', ')', '[', ']', '{', '}', '<', '>', '"', "'"]
        self.HAN

    def extractTextFromPdf(self, path):
        doc = fitz.open(path)
        text = ""
        for page in doc:
            text += re.sub(r"([. ])\1+", r"\1", page.get_text()).strip()
        return text

    def generateVocab(self, text):
        words = text.lower().split()
        for word in words:
            if word not in self.vocab and word != '' and len(word) < 10 and word not in self.punctuation:
                self.vocab.update({word: len(self.vocab)})
    
    # Function to traverse the folder and process all PDFs
    def processFilesInDirectory(self):
        pdfData = []
        label = 0

        listOfFolders = os.listdir(self.path)
        for folders in listOfFolders:

            folderPath = self.path + "/" + folders
            listOfFiles = os.listdir(folderPath)

            for file in listOfFiles:
                filePath = folderPath + "/" + file
                text = ''

                if file.endswith(".pdf"):
                    text = self.extractTextFromPdf(filePath)
                elif file.endswith(".docx"):
                    text = docx2txt.process(filePath)
                
                if text != '':
                    pdfData.append((text, label))
                    self.generateVocab(pdfData[-1][0])   
            label += 1

        return pdfData, label
    
    def train(self):
        documents, numCatgory = self.processFilesInDirectory()

        random.shuffle(documents)

        train = documents[0:int(0.8*len(documents))]
        test = documents[int(0.8*len(documents)):]

        self.HAN = model.HANModel(wordHiddenSize=64, sentenceHiddenSize=128, numLayers=2, embeddingDim=300, vocab=self.vocab, numCategories= numCatgory)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.HAN.parameters(), lr=0.001)

        epochs = 10
        for i in range(epochs):
            total_loss = 0  # Initialize total loss for the epoch
            for text, label in train:
                
                optimizer.zero_grad()
                label = torch.tensor([label])

                output = self.HAN.forward(text)
                loss = criterion(output, label)
                
                # Add the loss of this batch to the total loss
                total_loss += loss.item()

                loss.backward()
                optimizer.step()
            
            # Print the average loss for this epoch
            avg_loss = total_loss / len(documents)
            #print(f"Epoch {i+1}/{epochs}, Average Loss: {avg_loss:.4f}")

            return test, self.HAN

    def saveModel(self):    
        torch.save(self.HAN, "model.pth")  


if __name__ == "__main__":

    trainer = trainer(input("Enter testing Path"))
    test, HAN = trainer.train()
    
    numCorrect = 0
    for doc, label in test:
        output = HAN.forward(doc)
        if torch.argmax(output).item() == label:
            numCorrect += 1
    print(f"Accuracy: {numCorrect/len(test)*100:.2f}%")  
        



