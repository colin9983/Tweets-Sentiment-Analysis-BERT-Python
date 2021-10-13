output_dir = './'
model_to_save = model.module if hasattr(model, 'module') else model
model_to_save.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

from transformers import BertTokenizer,BertForSequenceClassification
import torch
# Load the BERT tokenizer.
print('Loading BERT tokenizer...')
output_dir = './'
tokenizer = BertTokenizer.from_pretrained(output_dir)
model_loaded = BertForSequenceClassification.from_pretrained(output_dir)


test_text = test.tweet.values
input_ids = []
attention_mask = []
for i in tqdm(test_text):
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

testset = TensorDataset(input_ids,attention_mask)
test_dl = DataLoader(testset, batch_size=1)

def evaluate2(dataloader_test):
    model_loaded =  BertForSequenceClassification.from_pretrained(output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_loaded = model_loaded.to(device)
    predictions = []
    
    for batch in dataloader_test:
        batch = tuple(b.to(device) for b in batch)
        inputs = {
            'input_ids':batch[0],
            'attention_mask': batch[1]
        }
        with torch.no_grad():
            outputs = model_loaded(**inputs)
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        predictions.append(logits)
    predictions = np.concatenate(predictions,axis=0)
    result = np.argmax(predictions, axis=1).flatten()
    
    return result

test_result = evaluate2(test_dl)

measure_accuracy = test
measure_accuracy['bert'] = test_result
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

print('f1 score', f1_score(measure_accuracy.sentiment, measure_accuracy.bert, average=None))
print(confusion_matrix(measure_accuracy.sentiment, measure_accuracy.bert))
print(precision_recall_fscore_support(measure_accuracy.sentiment, measure_accuracy.bert, average='weighted'))
print('accuarcy', accuracy_score(measure_accuracy.sentiment, measure_accuracy.bert))
