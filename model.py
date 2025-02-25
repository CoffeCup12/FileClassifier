import torch 
import torch.nn as nn
import torch.nn.functional as F 
import re
#This definces the word level and sentence level nerual netword of an HAN 
#This part of the network includes an encoder(GRU) and an attension layer     
class ProcessingNetwork(nn.Module):
    def __init__(self, inputSize, hiddenSize, numLayers):
        super(ProcessingNetwork, self).__init__()

        #instance vairable 
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers

        #encoder 
        self.gru = nn.GRU(inputSize, hiddenSize, numLayers, batch_first=True, bidirectional=True)

        #attension layer 
        self.attention = nn.Sequential(
            nn.Linear(hiddenSize*2, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        #forward encoder pass 
        output, _ = self.gru(x)
        #attension to get importance 
        attentionWeights = F.softmax(self.attention(output), dim=1)
        #output
        contextVector = torch.sum(attentionWeights * output, dim=1)
        return contextVector

#This class models the HAN modoled based on a word level network, a sentence level network and a final
#evaluation network 
class HANModel(nn.Module):
    
    def __init__(self, wordHiddenSize, sentenceHiddenSize, numLayers, vocab, embeddingDim, numCategories):
        super(HANModel, self).__init__()
        #instance variabels, embedder and vocab dictionary 
        self.embeddingDim = embeddingDim
        self.vocab = vocab

        #word and sentence level new work 
        self.wordLevel = ProcessingNetwork(embeddingDim, wordHiddenSize, numLayers)
        self.sentenceLevel = ProcessingNetwork(2 * wordHiddenSize, sentenceHiddenSize, numLayers)

        #embedding level for the work level network that translate word to vectors 
        self.embedding = nn.Embedding(len(vocab), embeddingDim)

        #final evaluation network 
        self.documentClassifcation = nn.Sequential(
            nn.Linear(2 * self.sentenceLevel.hiddenSize, numCategories),
            nn.Softmax(dim=1)
        )


    def forward(self, inputDocument):
        
        #separate document into a list of sentencs 
        sentences = self.separateSentences(inputDocument)
        processedSentences = []

        for sentence in sentences:
            #map each word in the sentence to a unique integer representation 
            wordsToIndex = self.separateWords(sentence)
            #for edge cases, if sentence is empty make it <PAD>
            if wordsToIndex == []:
                wordsToIndex = [0]
            #embedding 
            embeds = self.embedding(torch.LongTensor(wordsToIndex)).unsqueeze(0)
            #process the word rep vector to the word level network to get the processed sentence 
            processedSentence = self.wordLevel.forward(embeds)
            processedSentences.append(processedSentence)

        #pass the sentences to the sentence level network 
        sentenceTensor = torch.cat(processedSentences, dim=0).unsqueeze(0)
        documentVector = self.sentenceLevel.forward(sentenceTensor)

        #pass the document vector the classifcation layer
        return self.documentClassifcation(documentVector)

        
    def separateSentences(self, document):
        #separate document into sentences 
        sentenceSeparators = "(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s"
        return re.split(sentenceSeparators, document)
    
    def separateWords(self, sentence):
        #get int rep of words in a sentence 
        words = sentence.lower().split()
        return [self.vocab.get(word, self.vocab['<UNK>']) for word in words]  # Handle unknown words
    

    



    