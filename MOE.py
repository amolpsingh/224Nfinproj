
import pandas as pd
import numpy as np
from datasets import Dataset, load_metric
from sklearn.model_selection import train_test_split
from transformers import TrainingArguments, Trainer
from transformers import BertForSequenceClassification, BertTokenizer
import json
from transformers import TrainerCallback, EarlyStoppingCallback

#get test split from aman
test = pd.read_csv('test')
custom_dataset_train = Dataset.from_pandas(test)

expert1 = BertForSequenceClassification.from_pretrained('', num_labels=2)
expert2 = BertForSequenceClassification.from_pretrained('', num_labels=2)
expert3 = BertForSequenceClassification.from_pretrained('', num_labels=2)

predictions = expert1(**test)
preds = np.argmax(predictions.predictions, axis=-1)

#MLP
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

class BinaryClassification(nn.Module):
    def __init__(self):
        super(BinaryClassification, self).__init__()
        self.layer_1 = nn.Linear(3, 16) 
        self.layer_2 = nn.Linear(16, 16)
        self.layer_out = nn.Linear(16, 1) 
        
        # self.dropout = nn.Dropout(p=0.1)
        # self.batchnorm1 = nn.BatchNorm1d(64)
        # self.batchnorm2 = nn.BatchNorm1d(64)
        
    def forward(self, inputs):
        x = F.relu(self.layer_1(inputs))
        x = F.relu(self.layer_2(x))
        # x = self.dropout(x)
        x = self.layer_out(x)
        
        return x

#each entry shld be three cached predictions and target