import model
import torch.optim as optim
import torch
import re
import os
import fitz
import docx2txt
import random

def extract_text_from_pdf(path):
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += re.sub("\s+", " ", page.get_text()).strip()
    return text

vocab = {"<PAD>": 0, "<UNK>": 1}

def generateVocab(text):
    words = text.lower().split()
    for word in words:
        if word not in vocab and word != '':
            vocab.update({word: len(vocab)})
    
# Function to traverse the folder and process all PDFs
def process_pdfs_in_directory(directory):
    pdfData = []
    label = 0

    listOfFolders = os.listdir(directory)
    for folders in listOfFolders:

        folderPath = directory + "/" + folders
        listOfFiles = os.listdir(folderPath)

        for file in listOfFiles:
            filePath = folderPath + "/" + file
            text = ''

            if file.endswith(".pdf"):
                text = extract_text_from_pdf(filePath)
            elif file.endswith(".docx"):
                text = docx2txt.process(filePath)
            
            if text != '':
                pdfData.append((text, label))
                generateVocab(pdfData[-1][0])
                
        label += 1

    return pdfData, label

main_folder = input("Enter the path of the folder: ")  
documents, numCatgory = process_pdfs_in_directory(main_folder)

random.shuffle(documents)
HAN = model.HANModel(wordHiddenSize=32, sentenceHiddenSize=64, numLayers=1, embeddingDim=20, vocab=vocab, numCategories= numCatgory)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(HAN.parameters(), lr=0.001)

# # Example usage
epochs = 10
for i in range(epochs):
    total_loss = 0  # Initialize total loss for the epoch
    for text, label in documents:
        
        label = torch.tensor([label])

        output = HAN.forward(text)

        loss = criterion(output, label)
        
        # Add the loss of this batch to the total loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
    
    # Print the average loss for this epoch
    avg_loss = total_loss / len(documents)
    print(f"Epoch {i+1}/{epochs}, Average Loss: {avg_loss:.4f}")
        
        



