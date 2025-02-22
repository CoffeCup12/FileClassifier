import warnings
from pypdf import PdfReader

# Suppress specific warnings related to PyPDF
warnings.filterwarnings("ignore", message="Ignoring wrong pointing object")

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

path = input("Enter your path: ")
text = extract_text_from_pdf(path)
print(text)  # You can also return or save the text as needed
