import warnings
from pypdf import PdfReader
import model
import torch.optim as optim
import torch
import re

def extract_text_from_pdf(pdf_path):
    # Suppress specific warnings related to PyPDF
    warnings.filterwarnings("ignore", message="Ignoring wrong pointing object")

    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text
    
vocab = {"<PAD>": 0, "<UNK>": 1}
def generateVocab(text):
    words = text.lower().split()
    i = 2
    for word in words:
        if word not in vocab:
            vocab[word] = i
            i +=1 
    
path = input("Enter the path to the PDF file: ")
text = extract_text_from_pdf(path)
generateVocab(text)



HAN = model.HANModel(wordHiddenSize=32, sentenceHiddenSize=64, numLayers=1, embeddingDim=20, vocab=vocab, numCategories=2)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(HAN.parameters(), lr=0.001)

documents = [("This is a document", 0), ("This is another document", 1)]

for document, label in documents:
    optimizer.zero_grad()
    label = torch.tensor([label])

    output = HAN.forward(document)

    loss = criterion(output, label)
    loss.backward()
    optimizer.step()
