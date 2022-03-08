import pandas as pd
import numpy as np
import torch

from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from transformers import BertTokenizer, BertModel
from transformers import get_linear_schedule_with_warmup
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


class MixUpTransformer(torch.nn.Module):     
    def __init__(self, drop_rate=0.1, freeze_bert=False): 
        super(MixUpTransformer, self).__init__() 
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.drop_rate=drop_rate
        self.freeze_bert=freeze_bert
    
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

        self.feedforward = torch.nn.Sequential(
            torch.nn.Dropout(drop_rate),
            torch.nn.Linear(768, 1)
        )
        
    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        output = self.feedforward(bert_output[1])
        return output


def tokenize_data(df, tokenizer):
    tokenized_df = tokenizer(df['text'].tolist(), 
                                add_special_tokens=True,
                                truncation=True, 
                                padding="max_length",
                                return_attention_mask=True)

    input_ids = np.array(tokenized_df['input_ids'])
    attention_mask = np.array(tokenized_df['attention_mask'])
    return input_ids, attention_mask, np.array(df['label'])


def compute_mixup(a, b, l):
    return l * np.array(a) + (1 - l) * np.array(b)


def mixup_aug(input_ids, att_masks, labels, lam, n=50000):
    mixed_input_ids = []
    mixed_att_masks = []
    mixed_labels = []
    
    for i in range(n):
        a = np.random.randint(len(input_ids))
        b = np.random.randint(len(input_ids))
        
        new_input = list(map(int, compute_mixup(input_ids[a], input_ids[b], lam)))
        new_att = att_masks[a] if sum(att_masks[a]) > sum(att_masks[b]) else att_masks[b]
        new_label = float(compute_mixup(labels[a], labels[b], lam))

        mixed_input_ids.append(new_input)
        mixed_att_masks.append(new_att)
        mixed_labels.append(new_label)
        
    return np.array(mixed_input_ids), np.array(mixed_att_masks), np.array(mixed_labels)


def create_dataloaders(inputs, masks, labels, batch_size=16):
    input_tensor = torch.tensor(inputs)
    mask_tensor = torch.tensor(masks)
    labels_tensor = torch.tensor(labels)
    dataset = TensorDataset(input_tensor, mask_tensor, 
                            labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                            shuffle=True)
    return dataloader


def train_model(model, optimizer, scheduler, loss_function, epochs,       
          train_dataloader, device, clip_value=2):
    for epoch in range(epochs):
        print("Epoch", epoch)
        print("-----")
        model.train()      
        for step, batch in enumerate(train_dataloader): 
            batch_inputs, batch_masks, batch_labels = \
                               tuple(b.to(device) for b in batch)
            model.zero_grad()
            outputs = model(batch_inputs, batch_masks)    
            loss = loss_function(outputs.squeeze().float(), 
                             batch_labels.squeeze().float())
            loss.backward()
            clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
            scheduler.step()
    return model


def predict(model, dataloader, device):
    model.eval()
    output = []
    for batch in dataloader:
        batch_inputs, batch_masks, batch_labels = \
                                  tuple(b.to(device) for b in batch)
        with torch.no_grad():
            output += model(batch_inputs, 
                            batch_masks).view(1,-1).tolist()[0]
    return output


def evaluate(test_pred, test_labels):
    predictions = np.array([round(x) for x in test_pred])
    return {
        "accuracy": accuracy_score(test_labels, predictions),
        "precision": precision_score(test_labels, predictions),
        'recall': recall_score(test_labels, predictions),
        'f1': f1_score(test_labels, predictions)
    }


def run_mixup_experiment(lam=0.5):
    # read in data
    labeled = pd.read_csv('data/labeled_scouting.csv')
    test = pd.read_csv('data/augment/test.csv')
    train = pd.concat([labeled, test]).drop_duplicates(keep=False)

    # tokenize all text data using BERT tokenizer 
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_input_ids, train_attention_mask, train_labels = tokenize_data(train, tokenizer)
    test_input_ids, test_attention_mask, test_labels = tokenize_data(test, tokenizer)

    # mix-up algorithm for training data
    mixed_input_ids, mixed_att_masks, mixed_labels = mixup_aug(train_input_ids, train_attention_mask, train_labels, lam)

    # load data into batches (size 16)
    train_dataloader = create_dataloaders(mixed_input_ids, mixed_att_masks, mixed_labels)
    test_dataloader = create_dataloaders(test_input_ids, test_attention_mask, test_labels)

    # train model
    model = MixUpTransformer()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    epochs = 3
    scheduler = get_linear_schedule_with_warmup(optimizer,       
                    num_warmup_steps=0, num_training_steps=len(train_dataloader) * epochs)
    loss_function = torch.nn.MSELoss()
        
    trained_model = train_model(model, optimizer, scheduler, loss_function, epochs, 
              train_dataloader, device)

    # evaluate
    test_pred = predict(trained_model, test_dataloader, device)
    print(evaluate(test_pred, test_labels))

    
for lam in [0.1, 0.2, 0.3, 0.4, 0.5]:
    print(f"Starting experiment: MixUp w/ Lambda = {lam}")
    run_mixup_experiment(lam)
    print(f"Experiment finished")
