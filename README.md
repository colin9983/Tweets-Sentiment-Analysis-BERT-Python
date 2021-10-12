# Tweets-Sentiment-Analysis-BERT-Python
The aim of this release is to fine-tune a BERT model for Twitter sentiment analysis, i.e. to classify tweets as negative, neutral and positive. Training data and validation data are sourced from Kaggle:https://www.kaggle.com/gargmanas/sentimental-analysis-for-tweets


The code is built primarily on the BERT implementation, using pre-trained weights from the BERT-BASE library, and is used by pytorch to handle standard training and testing procedures.

# Requirements
Python >= 3.8

CUDA >= 10.1 for GPU training

# Parameter
Training
When fine-tuning BERT, over-fitting must be fought. The data set is too small for training BERT from scratch. Fine-tuning for too long or with too large a learning rate will make it easy for BERT to overfit on the training data, resulting in poor performance on the test set. Therefore, the learning rate of BERT should be set an order of magnitude lower than the classification head that was just initialised (see my configuration below). An additional weight decay term would not have made much difference either.

The test set performance below was adopted with this configuration:
```python
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
    
train_dl = DataLoader(train_dataset,sampler = RandomSampler(train_dataset),
                     batch_size = 4)
val_dl = DataLoader(val_dataset,sampler = SequentialSampler(val_dataset),
                     batch_size = 2)
		     
optimizer = AdamW(model.parameters(),lr = 2e-5,eps=1e-8)
```
| Accuracy | 0.82  |    
-----------| ---------
| F1-Score | 0.81  |
	
