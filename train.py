###Upload files to google colab###
from google.colab import files
uploaded = files.upload()

###Checking hardware information###
!nvidia-smi

import pandas as pd
import io
df = pd.read_csv(io.BytesIO(uploaded['tweets.csv']))


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils import shuffle

####Spilt train and test data###
df = df.sample(frac=1)
df = shuffle(df)
test = df.tail(300)
df.drop(df.tail(300).index, inplace=True)

labels = df.sentiment.values
text = df.tweet.values

from transformers import BertTokenizer,BertForSequenceClassification,AdamW
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case = True)

input_ids = []
attention_mask = []
for i in text:
    encoded_data = tokenizer.encode_plus(
    str(i),
    add_special_tokens=True,
    max_length= 64,
    pad_to_max_length = True,
    return_attention_mask= True,
    return_tensors='pt')
    input_ids.append(encoded_data['input_ids'])
    attention_mask.append(encoded_data['attention_mask'])
input_ids = torch.cat(input_ids,dim=0)
attention_mask = torch.cat(attention_mask,dim=0)
labels = torch.tensor(labels)

from torch.utils.data import DataLoader,SequentialSampler,RandomSampler,TensorDataset,random_split

dataset = TensorDataset(input_ids,attention_mask,labels)
train_size = int(0.9*len(dataset))
val_size = len(dataset) - train_size

train_dataset,val_dataset = random_split(dataset,[train_size,val_size])

print('Training Size - ',train_size)
print('Validation Size - ',val_size)

train_dl = DataLoader(train_dataset,sampler = RandomSampler(train_dataset),
                     batch_size = 4)
val_dl = DataLoader(val_dataset,sampler = SequentialSampler(val_dataset),
                     batch_size = 2)
len(train_dl),len(val_dl)

model = BertForSequenceClassification.from_pretrained(
'bert-base-uncased',
num_labels = 3,
output_attentions = False,
output_hidden_states = False)

import random

###Seed_val can set whatever you like###
seed_val = 7
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


###Checking graphic cards work
device = ('cuda')
model.to(device)
print(device)

optimizer = AdamW(model.parameters(),lr = 2e-5,eps=1e-8)

from transformers import get_linear_schedule_with_warmup
epochs = 6
total_steps = len(train_dl)*epochs
scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=0,
                                           num_training_steps=total_steps)


###Evaluating accuracy in test dataset###
def evaluate(dataloader_test):
    model.eval()
    loss_val_total = 0
    predictions,true_vals = [],[]
    for batch in dataloader_test:
        batch = tuple(b.to(device) for b in batch)
        inputs = {
            'input_ids':batch[0],
            'attention_mask': batch[1],
            'labels': batch[2]
        }
        with torch.no_grad():
            outputs = model(**inputs)
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()
        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
    loss_val_avg = loss_val_total / len(dataloader_test)
    predictions = np.concatenate(predictions,axis=0)
    true_vals = np.concatenate(true_vals,axis=0)
    return loss_val_avg,predictions,true_vals

def accuracy(preds,labels):
    pred_flat = np.argmax(preds,axis=1).flatten()
    label_flat = labels.flatten()
    return np.sum(pred_flat==label_flat)/len(label_flat)
vv = []
ac = []
t = []
from tqdm.notebook import tqdm
for epoch in tqdm(range(1, epochs+1)):
    
    model.train()
    
    loss_train_total = 0

    progress_bar = tqdm(train_dl, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
    for batch in progress_bar:

        model.zero_grad()
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }       

        outputs = model(**inputs)
        
        loss = outputs[0]
        loss_train_total += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
        
        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})
         
        
    tqdm.write(f'\nEpoch {epoch}')
    
    loss_train_avg = loss_train_total/len(train_dl)            
    tqdm.write(f'Training loss: {loss_train_avg}')
    t.append(loss_train_avg)
    val_loss, predictions, true_vals = evaluate(val_dl)
    val_acc = accuracy(predictions, true_vals)
    vv.append(val_loss)
    ac.append(val_acc)
    tqdm.write(f'Validation loss: {val_loss}')
    tqdm.write(f'Accuracy: {val_acc}')
