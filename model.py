import torch 
import torch.nn as nn
import torch.nn.functional as F 
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
    
    def __init__(self, wordHiddenSize, sentenceHiddenSize, numLayers, vocab, embeddingDim, numCategories):
        super(HANModel, self).__init__()
        self.embeddingDim = embeddingDim
        self.vocab = vocab

        self.wordLevel = ProcessingNetwork(embeddingDim, wordHiddenSize, numLayers)
        self.sentenceLevel = ProcessingNetwork(2 * wordHiddenSize, sentenceHiddenSize, numLayers)

        self.embedding = nn.Embedding(len(vocab), embeddingDim)

        self.documentClassifcation = nn.Sequential(
            nn.Linear(2 * self.sentenceLevel.hiddenSize, numCategories),
            nn.Softmax(dim=1)
        )


    def forward(self, inputDocument):

        sentences = self.separateSentences(inputDocument)
        processedSentences = []

        for sentence in sentences:

            wordsToIndex = self.separateWords(sentence)
            embeds = self.embedding(torch.LongTensor(wordsToIndex)).unsqueeze(0)

            processedSentence = self.wordLevel.forward(embeds)
            processedSentences.append(processedSentence)

        sentenceTensor = torch.cat(processedSentences, dim=0).unsqueeze(0)
        documentVector = self.sentenceLevel.forward(sentenceTensor)

        return self.documentClassifcation(documentVector)

        
    def separateSentences(self, document):
        sentenceSeparators = "(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s"
        return re.split(sentenceSeparators, document)
    
    def separateWords(self, sentence):
        words = sentence.lower().split()
        return [self.vocab.get(word, self.vocab['<UNK>']) for word in words]  # Handle unknown words

    



    