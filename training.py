import model
import torch.optim as optim
import torch
import re
import os
import fitz
import docx2txt
import random

#the class models the trainning loop of the HAN
class trainer():

    def __init__(self, path):
        #path to the folder that contains subfolders with organized files in each subfolder 
        self.path = path
        #instance varibles 
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        self.punctuation = ['.', ',', '!', '?', ':', ';', '(', ')', '[', ']', '{', '}', '<', '>', '"', "'"]

    def extractTextFromPdf(self, path):
        #read text from a pdf 
        doc = fitz.open(path)
        text = ""
        for page in doc:
            #get rid of repeating space and .s
            text += re.sub(r"([. ])\1+", r"\1", page.get_text()).strip()
        return text

    def generateVocab(self, text):
        #split the text of the entire pdf to words 
        words = text.lower().split()
        #create a mapping:str -> int to that word 
        for word in words:
            if word not in self.vocab and word != '' and len(word) < 10 and word not in self.punctuation:
                self.vocab.update({word: len(self.vocab)})
    
    def processFilesInDirectory(self):
        #create list of text, label pairs 
        pdfData = []
        label = 0

        #looks in every subfolder of the folder 
        listOfFolders = os.listdir(self.path)
        for folders in listOfFolders:

            folderPath = self.path + "/" + folders
            listOfFiles = os.listdir(folderPath)
            #looks in every file in each subfolder 
            for file in listOfFiles:
                filePath = folderPath + "/" + file
                text = ''
                #generate text
                if file.endswith(".pdf"):
                    text = self.extractTextFromPdf(filePath)
                elif file.endswith(".docx"):
                    text = docx2txt.process(filePath)
                #append text, label pair 
                if text != '':
                    pdfData.append((text, label))
                    self.generateVocab(pdfData[-1][0])   
            #update lable 
            label += 1

        return pdfData, label
    
    def train(self):
        #get the text label pair and the number of category to sotry 
        documents, numCategory = self.processFilesInDirectory()

        #shuffle the list
        random.shuffle(documents)

        #separate train and tes dataset 
        train = documents[0:int(0.8*len(documents))]
        test = documents[int(0.8*len(documents)):]

        #create model 
        HAN = model.HANModel(wordHiddenSize=64, sentenceHiddenSize=128, numLayers=2, embeddingDim=300, vocab=self.vocab, numCategories= numCategory)

        #loss function and optimizer 
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(HAN.parameters(), lr=0.001)
        
        #trainnning loop 
        epochs = 10
        for i in range(epochs):
            # Initialize total loss for the epoch
            total_loss = 0  
            for text, label in train:
                
                optimizer.zero_grad()
                label = torch.tensor([label])
                #caclulate loss 
                output = HAN.forward(text)
                loss = criterion(output, label)
                
                #add the loss of this batch to the total loss
                total_loss += loss.item()
                #backpropgation 
                loss.backward()
                optimizer.step()
            
            #print the average loss for this epoch
            avg_loss = total_loss / len(documents)
            print(f"Epoch {i+1}/{epochs}, Average Loss: {avg_loss:.4f}")

        #save model
        torch.save(HAN, "model.pth") 

        #return test set and network
        return test, HAN
             


if __name__ == "__main__":
    #for testing purpose 
    trainer = trainer(input("Enter testing Path"))
    test, HAN = trainer.train()
    
    #output accurary from testing data set 
    numCorrect = 0
    for doc, label in test:
        output = HAN.forward(doc)
        if torch.argmax(output).item() == label:
            numCorrect += 1
    print(f"Accuracy: {numCorrect/len(test)*100:.2f}%")  
        



