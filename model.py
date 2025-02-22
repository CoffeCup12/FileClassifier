import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
    
class ProcessinglNetwork(nn.Module):
    def __init__(self, inputSize, hiddenSize, numLayers):
        super(ProcessinglNetwork, self).__init__()
        self.gru = nn.GRU(inputSize, hiddenSize, numLayers, batch_first=True, bidirectional=True)
        self.attention = nn.Sequential(
            nn.Linear(hiddenSize*2, 1),
            nn.Tanh()
        )
    
    def forward(self, inputSentence):
        #input sentence is the processed sentence from the word level network
        output, _ = self.gru(inputSentence)
        attentionWeights = F.softmax(self.attention(output), dim=1)
        contextVector = torch.sum(attentionWeights * output, dim=1)
        return contextVector

class HANModel(nn.Module):
    
    def __init__(self, wordLevel, sentenceLevel, numCategories):
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