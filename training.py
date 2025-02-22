import warnings
from pypdf import PdfReader
import model
import torch.optim as optim
import torch
import re
import os

def extract_text_from_pdf(pdf_path):
    # Suppress specific warnings related to PyPDF
    warnings.filterwarnings("ignore", message="Ignoring wrong pointing object")

    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            
            # Split the page text into lines
            lines = page_text.split('\n')
            
            # Process each line
            for line in lines:
                # Skip empty lines
                if not line.strip():
                    continue
                
                # Split the line into words and filter out words with length 10 or more
                words = line.split()
                filtered_words = [word for word in words if len(word) < 10]
                
                # Reconstruct the line with filtered words and add it to the text
                if filtered_words:
                    text += " ".join(filtered_words) + " "
    
    return text
    
vocab = {"<PAD>": 0, "<UNK>": 1}
def generateVocab(text):
    words = text.lower().split()
    i = 2
    for word in words:
        if word not in vocab:
            vocab[word] = i
            i +=1 
    
# Function to traverse the folder and process all PDFs
def process_pdfs_in_directory(directory):
    pdf_data = []

    for root, dirs, files in os.walk(directory):
        # Skip non-PDF files
        pdf_files = [f for f in files if f.endswith('.pdf')]

        # Determine the label based on the folder name (assuming two categories)
        if 'Labs' in root:
            label = 0
        elif 'APPs' in root:
            label = 1
        else:
            continue  # Skip if the folder doesn't match either category

        for file in pdf_files:
            pdf_path = os.path.join(root, file)
            text = extract_text_from_pdf(pdf_path)
            pdf_data.append((text, label))

    return pdf_data

main_folder = '/Users/khushpatel/FEH/Testing'  # Replace with your main folder path
documents = process_pdfs_in_directory(main_folder)
# Print the output to check
for text, label in documents:
    print(f"Label: {label}, Text Length: {len(text)}")

HAN = model.HANModel(wordHiddenSize=32, sentenceHiddenSize=64, numLayers=1, embeddingDim=20, vocab=vocab, numCategories=2)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(HAN.parameters(), lr=0.001)

# Example usage
epochs = 5
for i in range(epochs):
    total_loss = 0  # Initialize total loss for the epoch
    for text, label in documents:
        # Tokenize the text into word indices using the vocab
        tokenized_text = [vocab[word] for word in text.split() if word in vocab]  # Replace with actual tokenization logic
        
        # Ensure the tokenized text is not empty
        if len(tokenized_text) == 0:
            continue  # Skip empty texts

        # Convert tokenized text to tensor
        text_tensor = torch.tensor(tokenized_text).unsqueeze(0)  # Unsqueeze to add batch dimension
        
        label = torch.tensor([label])

        output = HAN.forward(text_tensor)

        loss = criterion(output, label)
        
        # Add the loss of this batch to the total loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
    
    # Print the average loss for this epoch
    avg_loss = total_loss / len(documents)
    print(f"Epoch {i+1}/{epochs}, Average Loss: {avg_loss:.4f}")
        
        



