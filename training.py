import pdfplumber
import os

pdf_path = ""
if os.path.exists(pdf_path):
    print("File exists!")
else:
    print("File not found!")
    
with pdfplumber.open(pdf_path) as pdf:
    text = ""
    for page in pdf.pages:
        text += page.extract_text()  # Extract only the text, ignoring other content

print(text)  # Print or process the text