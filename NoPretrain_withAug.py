import pandas as pd
import numpy as np
from datasets import Dataset, load_metric
from sklearn.model_selection import train_test_split
from transformers import TrainingArguments, Trainer
from transformers import BertForSequenceClassification, BertTokenizer
import json

train = pd.read_csv('/home/asingh11/224Nfinproj/data/train_shuffle_sentences.csv')

test = pd.read_csv('/home/asingh11/224Nfinproj/data/test.csv') 
custom_dataset_train = Dataset.from_pandas(train)
custom_dataset_test = Dataset.from_pandas(test)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


train = custom_dataset_train.map(lambda e: tokenizer(e['text'], truncation=True, padding=True), batched=True)
test = custom_dataset_test.map(lambda e: tokenizer(e['text'], truncation=True, padding=True), batched=True)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)


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


# def compute_metrics(eval_pred):
#     """Called at the end of validation. Gives accuracy"""
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     # calculates the accuracy
#     return {"accuracy": np.mean(predictions == labels)}
def compute_metrics(eval_preds):
    metric = load_metric("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


trainer = Trainer(
    model=model,
    args=arguments,
    train_dataset=train,
    eval_dataset=test, # change to test when you do your final evaluation!
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

from transformers import TrainerCallback, EarlyStoppingCallback

class LoggingCallback(TrainerCallback):
    def __init__(self, log_path):
        self.log_path = log_path
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(logs) + "\n")


trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=1, early_stopping_threshold=0.0))
trainer.add_callback(LoggingCallback("train_shuffle/log.jsonl"))

print("TRAINING____")
trainer.train()
print("DONE TRAINING____")
#Evaluation
predictions = trainer.predict(test)

preds = np.argmax(predictions.predictions, axis=-1)

metric = load_metric("glue", "mrpc")
print(metric.compute(predictions=preds, references=test['label']))
