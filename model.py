import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

class HANModel(nn.Module):
    
    def __init__(self, wordLevelVariables, sentenceLevelVariables):
        super(HANModel, self).__init__()
        self.embedding = nn.Embedding(wordLevelVariables['vocabSize'], wordLevelVariables['input'])
        self.wordEncoder = nn.GRU(wordLevelVariables['inputSize'], wordLevelVariables['hiddenSize'], wordLevelVariables['numLayers'], batch_first=True, bidirectional=True)  
        self.wordAttention = nn.Sequential(
            nn.Linear(wordLevelVariables['hiddenSize']*2, 1),
            nn.Tanh()
        )

        self.sentenceEncoder = nn.GRU(sentenceLevelVariables['inputSize'], sentenceLevelVariables['hiddenSize'], sentenceLevelVariables['numLayers'], batch_first=True, bidirectional=True)
        self.sentenceAttention = nn.Sequential(
            nn.Linear(sentenceLevelVariables['hiddenSize']*2, 1),
            nn.Tanh()
        )



    