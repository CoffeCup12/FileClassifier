import torch 
import torch.nn as nn
import torch.nn.functional as F 
import nltk
from pathlib import Path
import os
import spacy
import numpy as np
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence, PackedSequence

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
        self.encoder = nn.GRU(input_size=input_size, hidden_size=hidden_size, bidirectional=True, batch_first = True)
        self.mlp = nn.Sequential(
            nn.Linear(2*hidden_size, 2*hidden_size),
            nn.Tanh(),
        )
        self.context_vector = torch.randn(1, 2*hidden_size)
    
    def forward(self, x):
        """returns a vector of dim (1,hidden_size) that represents the word/sentence"""
        x,_ = self.encoder(x)
        
        if(isinstance(x, PackedSequence)):
            x,_ = pad_packed_sequence(x, batch_first = True)

        h = x.clone().detach()
        x = self.mlp(x)
        
        #softmax with context vector
        x = x * self.context_vector.repeat(x.size()[0], x.size()[1], 1)
        
        x = torch.sum(x, dim=2, keepdim = True)
        x = F.softmax(x, dim=1)
        #weighted sum
        x = x.repeat(1,1,h.size()[2]) * h
        x = torch.sum(x, dim=1)
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
        """retrun a (batch_size,num_classes) vector that represents the probability of the document's belonging """
        batched_input = [sent_tokenize(doc) for doc in x]
        
        batched_sentences_rep = [self.process_sentence_rep(sentences) for sentences in batched_input]
        length = [rep.size(0) for rep in batched_sentences_rep]

        #add paddig
        padded_sentences = pad_sequence(batched_sentences_rep, batch_first=True)

        packed_input = pack_padded_sequence(padded_sentences, length, batch_first=True, enforce_sorted=False)
        batched_doc_rep = self.sentence_network(packed_input)

        #classification
        final = self.classifcation(batched_doc_rep)

        return final
    
    def process_sentence_rep(self, sentences):
        """returns a (sequence length, 100) vector representations for all sentences in a document """
        sentences_rep = []
        for sentence in sentences:
            tokens = self.tokenizer(sentence)
            embeddings = torch.tensor(np.array([token.vector for token in tokens]))
            #generate (1, 100) vectors for each sentence 
            output = self.word_network(embeddings.unsqueeze(0))
            sentences_rep.append(output.squeeze(0))

        #print(torch.stack(sentences_rep).size())
        return torch.stack(sentences_rep)


if __name__ == "__main__":
    model = Doc_network(2)
    text = ("unclassified sentence1. Sentence2. This is a sample sentence.", "hello world.", "the fox jumps over the lazy dog")
    print(model(text))
    

