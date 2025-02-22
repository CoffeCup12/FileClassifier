import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import re
    
class ProcessinglNetwork(nn.Module):
    def __init__(self, inputSize, hiddenSize, numLayers):
        super(ProcessinglNetwork, self).__init__()

        self.hiddenSize = hiddenSize
        self.numLayers = numLayers

        self.gru = nn.GRU(inputSize, hiddenSize, numLayers, batch_first=True, bidirectional=True)
        self.attention = nn.Sequential(
            nn.Linear(hiddenSize*2, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        output, _ = self.gru(x)
        attentionWeights = F.softmax(self.attention(output), dim=1)
        contextVector = torch.sum(attentionWeights * output, dim=1)
        return contextVector

class HANModel(nn.Module):
    
    def __init__(self, wordLevel, sentenceLevel, numCategories):
        super(HANModel, self).__init__()
        self.wordLevel = wordLevel
        self.sentenceLevel = sentenceLevel

        self.documentClassifcation = nn.Sequential(
            nn.Linear(2 * sentenceLevel.hiddenSize, numCategories),
            nn.Softmax()
        )


    def forward(self, inputDocument):

        sentences = self.separateSentences(inputDocument)
        processedSentences = []

        for sentence in sentences:

            words = sentence.split()
            wordsToIndex = {}

            for i in range(len(words)):
                wordsToIndex.update({words[i]: i})

            embedding = nn.Embedding(vocab_size = len(words), embedding_dim = 10)
            looksUpTensor = torch.tensor([wordsToIndex["geeks"]], dtype=torch.long)
            embeds = embedding(looksUpTensor)

            processedSentence = self.sentenceLevel.foward(self.wordLevel.forward(embeds))
            processedSentences.append(processedSentence)

        self.documentClassifcation(torch.tensor(processedSentences))

        
    def separateSentences(self, document):
        sentenceSeparators = "(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s"
        return re.split(sentenceSeparators, document)
    



    