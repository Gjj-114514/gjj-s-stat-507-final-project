from transformers import DistilBertTokenizer, DistilBertForSequenceClassification,DistilBertConfig
import random 
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import numpy as np
import jsonlines as js
from torch import optim as opt
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from matplotlib import pyplot as plt
len_test=400
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

#setting random seed
def set_seed(seed):
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed) 
    torch.backends.cudnn.deterministic = True 
    np.random.seed(seed)
    random.seed(seed)
set_seed(114514)

#constructing data
class text_data(Dataset):
    def __init__(self,path):
        self.file_path = path
        f=open(path,"r")
        self.json=[x for x in js.Reader(f)]
        f.close()
    def __getitem__(self,index):
        line=self.json[index]
        text=line['text']
        text=tokenizer(text, max_length = 4096, truncation=True,return_tensors="pt")["input_ids"]
        label=[0,0]
        label[line['label']]=1
        return text,torch.FloatTensor(label)

    def __len__(self):
        return len(self.json)
test_set=text_data('../data/fake_news_test.jsonl')
part_test_set=random_split(test_set,[len_test,len(test_set)-len_test])[0]
test_loader=DataLoader(part_test_set, batch_size=1,shuffle=True)

#training and validating
config=DistilBertConfig.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
config.max_position_embeddings=4096
#testing
model=torch.load("best.pt")
model.eval()
correct_case=0
preds=[]
truths=[]
for text,label in test_loader:
    input =text.squeeze(1).to(0)
    pred=model(input).logits.to(0)
    label=label.to(0)
    preds.append(pred.argmax().cpu())
    truths.append(label.argmax().cpu())
    if (pred.argmax()==label.argmax()):
        correct_case+=1
print ("test_acc="+str(correct_case/len_test))
#confusion matrix
from sklearn import metrics 
confusion_matrix = metrics.confusion_matrix(truths, preds)
# Display the confusion matrix
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[0, 1])
cm_display.plot()
plt.show()
