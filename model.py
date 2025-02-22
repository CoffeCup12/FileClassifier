import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim


class WordLevelNetwork(nn.Module):
    def __init__(self, inputSize, hiddenSize, numLayers, vocabSize):
        super(WordLevelNetwork, self).__init__()
        self.embedding = nn.Embedding(vocabSize, inputSize)
        self.gru = nn.GRU(inputSize, hiddenSize, numLayers, batch_first=True, bidirectional=True)
        self.attention = nn.Sequential(
            nn.Linear(hiddenSize*2, 1),
            nn.Tanh()
        )
    
    def forward(self, inputText):
        embedded = self.embedding(inputText)
        output, _ = self.gru(embedded)
        attentionWeights = F.softmax(self.attention(output), dim=1)
        contextVector = torch.sum(attentionWeights * output, dim=1)
        return contextVector
    
class SentenceLevelNetwork(nn.Module):
    def __init__(self, inputSize, hiddenSize, numLayers):
        super(SentenceLevelNetwork, self).__init__()
        self.gru = nn.GRU(inputSize, hiddenSize, numLayers, batch_first=True, bidirectional=True)
        self.attention = nn.Sequential(
            nn.Linear(hiddenSize*2, 1),
            nn.Tanh()
        )
    
    def forward(self, inputText):
        output, _ = self.gru(inputText)
        attentionWeights = F.softmax(self.attention(output), dim=1)
        contextVector = torch.sum(attentionWeights * output, dim=1)
        return contextVector

class HANModel(nn.Module):
    
    def __init__(self, wordLevel, sentenceLevel, numCategories):
        super(HANModel, self).__init__()
        self.wordLevel = wordLevel
        self.sentenceLevel = sentenceLevel

        self.documentClassifcation = nn.Sequential(
            nn.Linear(2 * sentenceLevel.hiddenSize, 45),
            nn.ReLU(),
            nn.Linear(45, 45),
            nn.ReLU(),
            nn.Linear(45, 30),
            nn.ReLU(),
            nn.Linear(30, numCategories),
            nn.Softmax()
        )


    def forward(self, inputText):
        #Change this later
        embedding = nn.Embedding(vocab_size = 0, embedding_dim=0)
        





    