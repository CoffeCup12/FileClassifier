import warnings
import model
from pathlib import Path
import torch.optim as optim
import torch
import re
import os
import fitz

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += re.sub(r"([. ])\1+", r"\1", page.get_text()).strip()
    return text

vocab = {"<PAD>": 0, "<UNK>": 1}
punctuation = ['.', ',', '!', '?', ':', ';', '(', ')', '[', ']', '{', '}', '<', '>', '"', "'"]

def generateVocab(text):
    words = text.lower().split()
    for word in words:
        if word not in vocab and word != '' and len(word) < 10 and word not in punctuation:
            vocab.update({word: len(vocab)})
    
# Function to traverse the folder and process all PDFs
def processFilesInDirectory(directory):
    pdfData = []
    label = 0

    listOfFolders = os.listdir(directory)
    for folders in listOfFolders:

        folderPath = directory + "/" + folders
        listOfFiles = os.listdir(folderPath)

        for file in listOfFiles:
            if file.endswith(".pdf"):
                text = extract_text_from_pdf(filePath)
            elif file.endswith(".docx"):
                text = docx2txt.process(filePath)
            
            if text != '':
                pdfData.append((text, label))
                generateVocab(pdfData[-1][0])   
        label += 1

    return pdfData, label

main_folder = input("Enter the folder: ")
documents, numCatgory = processFilesInDirectory(main_folder)

random.shuffle(documents)

train = documents[0:int(0.8*len(documents))]
test = documents[int(0.8*len(documents)):]

HAN = model.HANModel(wordHiddenSize=64, sentenceHiddenSize=128, numLayers=2, embeddingDim=300, vocab=vocab, numCategories= numCatgory)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(HAN.parameters(), lr=0.01)

# # Example usage
epochs = 50
for i in range(epochs):
    total_loss = 0  # Initialize total loss for the epoch
    for text, label in train:
        
        optimizer.zero_grad()
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

for doc, label in test:
    output = HAN.forward(doc)
    print(f"Predicted: {torch.argmax(output).item()}, Actual: {label}")    
        



