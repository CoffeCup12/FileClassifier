import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import re
    
class ProcessingNetwork(nn.Module):
    def __init__(self, inputSize, hiddenSize, numLayers):
        super(ProcessingNetwork, self).__init__()

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
    
    def __init__(self, wordHiddenSize, sentenceHiddenSize, numLayers, embiddingDim, numCategories):
        super(HANModel, self).__init__()
        self.embeddingDim = embiddingDim

        self.wordLevel = ProcessingNetwork(embiddingDim, wordHiddenSize, numLayers)
        self.sentenceLevel = ProcessingNetwork(2 * wordHiddenSize, sentenceHiddenSize, numLayers)

        self.documentClassifcation = nn.Sequential(
            nn.Linear(2 * self.sentenceLevel.hiddenSize, numCategories),
            nn.Softmax(dim=1)
        )


    def forward(self, inputDocument):

        sentences = self.separateSentences(inputDocument)
        processedSentences = []

        for sentence in sentences:

            wordsToIndex = self.separateWords(sentence)
            embedding = nn.Embedding(num_embeddings = len(wordsToIndex), embedding_dim = self.embeddingDim)
            embeds = embedding(torch.LongTensor(wordsToIndex)).unsqueeze(0)

            processedSentence = self.wordLevel.forward(embeds)
            processedSentences.append(processedSentence)

        sentenceTensor = torch.cat(processedSentences, dim=0).unsqueeze(0)
        documentVector = self.sentenceLevel.forward(sentenceTensor)

        return self.documentClassifcation(documentVector)

        
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
        wordToindex = {word: ind for ind, word in enumerate(vocab)} 
        
        return [wordToindex[word] for word in words]
    

    



    