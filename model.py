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
        self.wordLevel.forward(embedding)
        





    