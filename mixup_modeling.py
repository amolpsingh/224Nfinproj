import pandas as pd
import numpy as np
from datasets import Dataset, load_metric
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from transformers import TrainingArguments, Trainer
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import TrainerCallback, EarlyStoppingCallback
import json
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


class LoggingCallback(TrainerCallback):
    def __init__(self, log_path):
        self.log_path = log_path
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(logs) + "\n")


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def mixup(row1, row2, l, label=1):
    mixed = list(map(int, l * np.array(row1['input_ids'].iloc[0]) + (1 - l) * np.array(row2['input_ids'].iloc[0])))
    if sum(row1['attention_mask'].iloc[0]) > sum(row2['attention_mask'].iloc[0]):
        new_attention = row1['attention_mask'].iloc[0]
    else:
        new_attention = row2['attention_mask'].iloc[0]

    return pd.DataFrame([{
        'label': label,
        'input_ids': mixed,
        'token_type_ids': row1['token_type_ids'].iloc[0],
        'attention_mask': new_attention
    }])


def mixup_aug(train, lam):
    tokenized_train = pd.DataFrame(train)
    pos = tokenized_train[tokenized_train["label"] == 1].copy(deep=True)
    neg = tokenized_train[tokenized_train["label"] == 0].copy(deep=True)
    for i in range(len(neg) - len(pos)):
        row1 = pos.sample()
        row2 = pos.sample()
        new_embed = mixup(row1, row2, lam)
        tokenized_train = tokenized_train.append(new_embed, ignore_index=True)
    shuffle(tokenized_train)
    return Dataset.from_pandas(tokenized_train)


def run_mixup_experiments(lam=0.5):
    # read in data
    labeled = pd.read_csv('data/labeled_scouting.csv')
    test = pd.read_csv('data/augment/test.csv')
    train = pd.concat([labeled, test]).drop_duplicates(keep=False)

    #convert to huggingface dataset type
    custom_dataset_train = Dataset.from_pandas(train)
    custom_dataset_test = Dataset.from_pandas(test)

    # tokenize all text data using BERT tokenizer 
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train = custom_dataset_train.map(lambda e: tokenizer(e['text'], truncation=True, padding="max_length"), batched=True)
    train = train.remove_columns(["text", "__index_level_0__"])
    test = custom_dataset_test.map(lambda e: tokenizer(e['text'], truncation=True, padding="max_length"), batched=True)
    test = test.remove_columns(["text"])

    #instantiate model, replace first param to use pre-trained model 
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    #set training arguments for model
    arguments = TrainingArguments(
        output_dir="sample_SD_trainer",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        evaluation_strategy="epoch", # run validation at the end of each epoch
        save_strategy="epoch",
        learning_rate=2e-5,
        load_best_model_at_end=True,
        seed=224
    )

    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=mixup_aug(train, lam),     # mix-up augmentation
        eval_dataset=test,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=1, early_stopping_threshold=0.0))
    trainer.add_callback(LoggingCallback("sample_SD_trainer/log.jsonl"))

    #training call
    trainer.train()

    #Evaluation
    predictions = trainer.predict(test)

    preds = np.argmax(predictions.predictions, axis=-1)

    metric = load_metric("glue", "mrpc")
    print(metric.compute(predictions=preds, references=test['label']))


for lam in [0.2, 0.5, 0.8]:
    print(f"Starting experiment: MixUp w/ Lambda = {lam}")
    run_mixup_experiments(l=lam)
    print(f"Experiment finished")
