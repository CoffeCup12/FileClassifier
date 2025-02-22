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

            wordsToIndex = self.separateWords(sentence)

            embedding = nn.Embedding(vocab_size = len(wordsToIndex), embedding_dim = 10)
            embeds = embedding(torch.LongTensor(wordsToIndex))

            processedSentence = self.sentenceLevel.foward(self.wordLevel.forward(embeds))
            processedSentences.append(processedSentence)

        return self.documentClassifcation(torch.tensor(processedSentences))

        
    def separateSentences(self, document):
        sentenceSeparators = "(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s"
        return re.split(sentenceSeparators, document)
    
    def separateWords(self, sentence):
        words = sentence.split()
        vocab = {}

        for i in words:
            if i in vocab:
                vocab[i] += 1
            else:
                vocab[i] = 1

        vocab = sorted(vocab, key=vocab.get, reverse=True)
        
        # create a word to index dictionary from our Vocab dictionary
        word2idx = {word: ind for ind, word in enumerate(vocab)} 
        
        return [word2idx[word] for word in words]
    

    



    