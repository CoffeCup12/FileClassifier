import warnings
from pypdf import PdfReader
import model

def extract_text_from_pdf(pdf_path):
    # Suppress specific warnings related to PyPDF
    warnings.filterwarnings("ignore", message="Ignoring wrong pointing object")

    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text


path = input("Enter your path: ")
text = extract_text_from_pdf(path)

HAN = model.HANModel(wordHiddenSize=32, sentenceHiddenSize=64, numLayers=1, embiddingDim=20, numCategories=5)

output = HAN.forward(text)

print(output)
