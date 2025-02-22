import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

class HANModel(nn.Module):
    
    def __init__(self, wordEcoderVariables):
        super(HANModel, self).__init__()
        self.wordEncoder = nn.GRU(wordEcoderVariables['inputSize'], wordEcoderVariables['hiddenSize'], wordEcoderVariables['numLayers'], batch_first=True, bidirectional=True)  
        self.wordAttention = nn.Sequential(
            nn.Linear(wordEcoderVariables['hiddenSize']*2, 1),
            nn.Tanh()
        )



    