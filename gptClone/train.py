import torch
import torch.nn as nn
from torch.nn import functional as F

with open('input.txt', 'r', encoding='utf-8') as inputFile:
  text = inputFile.read()
  
# seed for torch
torch.manual_seed(1337)

# Get all characters present in textFile and put in sorted list
characters = sorted(list(set(text)))

# Create dictionaries for all present characters
stringToIndex = { char:index for index, char in enumerate(characters) }
indexToString = { index:char for index, char in enumerate(characters) }

# Functions that convert characters into intergers, and vice versa
encode = lambda string: [ stringToIndex[char] for char in string ]
decode = lambda list: ''.join([ indexToString[index] for index in list ])

# Create tensor with encoded text
data = torch.tensor(encode(text), dtype = torch.long)

# Seperate data to create a training data chunk and a validation data chunk
n = int(0.9 * len(data))
trainingData = data[:n] # 90%
validationData = data[n:] # 10%

# Max length of chunk of training data to train using transformer
blockSize = 8
# Number of chunks processing
batchSize = 4

def getBatch(split):
  # Generate batch of blocks with random offsets in the data
  data = trainingData if split == 'train' else validationData
  
  # Generate random indexes in data then use to create stack (2 dimensional tensor, 4 x 8)
  randomIndexesInData = torch.randint(len(data) - blockSize, (batchSize,))
  
  # Create stack using random indexes to get blockSize chunks of data for each random index
  inputs = torch.stack([data[i:i + blockSize] for i in randomIndexesInData])
  targets = torch.stack([data[i + 1:i + blockSize + 1] for i in randomIndexesInData])
  
  return inputs, targets
  
inputBatch, targetBatch = getBatch('train')

print('inputs:')
print(inputBatch.shape)
print(inputBatch)

print('targets:')
print(targetBatch.shape)
print(targetBatch)
  
print('-----')

for batchIndex in range(batchSize):
  for blockIndex in range(blockSize):
    context = inputBatch[batchIndex, :blockIndex + 1]
    target = targetBatch[batchIndex, blockIndex]
    print(f"when input is {context.tolist()} the target: {target}")
    
# Time to feed this input data + target data into neural networks
class BigramLanguageModel(nn.Module):
  
  def __init__(self, vocabSize):
    super().__init__()
    self.tokenEmbeddingTable = nn.Embedding(vocabSize, vocabSize)
    
  def forward(self, inputIndex, targets):
    logits = self.tokenEmbeddingTable(inputIndex)
    return logits
    

  
  