#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
from huggingface_hub import notebook_login
import transformers
from datasets import load_dataset
from transformers import BertTokenizer, AutoTokenizer
from transformers import Trainer, TrainingArguments
import math
from transformers import BertForPreTraining, AutoModelForMaskedLM
from transformers import DataCollatorForLanguageModeling


# In[2]:


def tokenize_function(examples):
    return tokenizer(examples["text"])


# In[3]:


block_size = 128

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


# In[4]:


print("starting login")
notebook_login()


# In[5]:


print("done with login")
datasets = load_dataset('wikitext', 'wikitext-2-raw-v1')


# In[6]:


datasets["train"][10]


# In[7]:


#model_checkpoint = "distilroberta-base"
model_checkpoint = "bert-base-uncased"
#tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])


# In[8]:


lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=1000,
    num_proc=4,
)


# In[9]:


model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
#model = BertForPreTraining.from_pretrained(model_checkpoint)


# In[10]:


model_name = model_checkpoint.split("/")[-1]
training_args = TrainingArguments(
    f"{model_name}-pretrained-wikitext2",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    push_to_hub=True,
)


# In[11]:


data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)


# In[12]:


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
    data_collator=data_collator,
)


# In[13]:


print("pretraining on wiki")
trainer.train()


# In[ ]:


eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")


# In[ ]:


print("pushing wiki checkpoint")
trainer.push_to_hub()


# In[14]:


datasets = load_dataset("text", data_files={"train": "data/sports_article_data.csv", "validation": "data/sports_article_data.csv"})


# In[15]:


datasets["train"][10]


# In[16]:


#model_checkpoint = "distilroberta-base"
model_checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
#tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
model_checkpoint = "amanm27/bert-base-uncased-pretrained-wikitext2"
tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])


# In[17]:


lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=1000,
    num_proc=4,
)


# In[18]:


model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
#model = BertForPreTraining.from_pretrained(model_checkpoint)


# In[19]:


model_name = model_checkpoint.split("/")[-1]
training_args = TrainingArguments(
    f"{model_name}-pretrained-sports-articles",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    push_to_hub=True,
)


# In[20]:


print(model_name)


# In[21]:


data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)


# In[22]:


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
    data_collator=data_collator,
)


# In[23]:


print("pretraining on articles")
trainer.train()


# In[ ]:


eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")


# In[ ]:


print("pushing article checkpoint")
trainer.push_to_hub()

