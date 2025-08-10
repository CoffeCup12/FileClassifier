import torch 
import torch.nn as nn
import torch.nn.functional as F 
import nltk
from pathlib import Path
import os
import spacy
import numpy as np

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    VENV_DIR = Path(".venv")
    NLTK_DATA_DIR = VENV_DIR / "nltk_data"
    os.environ["NLTK_DATA"] = str(NLTK_DATA_DIR)
    nltk.data.path.append(str(NLTK_DATA_DIR))
    nltk.download("punkt_tab", download_dir = str(NLTK_DATA_DIR))

from nltk.tokenize import sent_tokenize

class Processing_network(nn.Module):
    """The Processing network for word and sentence level"""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.encoder = nn.GRU(input_size=input_size, hidden_size=hidden_size, bidirectional=True)
        self.mlp = nn.Sequential(
            nn.Linear(2*hidden_size, 2*hidden_size),
            nn.Tanh(),
        )
        self.context_vector = torch.randn(1, 2*hidden_size)
    
    def forward(self, x):
        """returns a vector of dim (1,hidden_size) that represents the word/sentence"""
        x = self.encoder(x)[0]
        h = x.clone().detach()
        x = self.mlp(x)
        
        #softmax with context vector
        x = x * self.context_vector.repeat(x.size()[0], 1)
        x = torch.sum(x, dim=1)                                          
        x = F.softmax(x, dim=0)
        
        #weighted sum
        x = x.repeat(h.size()[1],1).t() * h
        x = torch.sum(x, dim=0)
        return x     


class Doc_network(nn.Module):
    """The HAN that does the classification """
    def __init__(self, num_classes):
        super().__init__()
        self.word_network = Processing_network(300,50)
        self.sentence_network = Processing_network(100,50)
        self.classifcation = nn.Sequential(
            nn.Linear(100,num_classes),
            nn.Softmax(dim=0)
        )
        self.tokenizer = spacy.load("en_core_web_md")

    def forward(self,x):
        """retrun a (1,num_classes) vector that represents the probability of the document's belonging """
        sentences = sent_tokenize(x) #sentence from doc
        
        #get sentences representations for all sentences in docs
        sentences_rep = []
        for sentence in sentences:
            tokens = self.tokenizer(sentence)
            embeddings = torch.tensor(np.array([token.vector for token in tokens]))
            sentences_rep.append(self.word_network(embeddings))
        
        sentences_rep = torch.stack(sentences_rep)
        
        #doc representations from sentence representations
        doc_rep = self.sentence_network(sentences_rep)

        #classification
        final = self.classifcation(doc_rep)

        return final

if __name__ == "__main__":
    model = Doc_network(2)
    text = "unclassified"
    print(model(text))
    

