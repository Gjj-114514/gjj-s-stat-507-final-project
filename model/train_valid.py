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
len_train=4000
len_valid=400
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
train_set=text_data('../data/train.jsonl')
part_train_set=random_split(train_set,[len_train,len(train_set)-len_train])[0]
train_loader=DataLoader(part_train_set, batch_size=1,shuffle=True)
valid_set=text_data('../data/fake_news_validation.jsonl')
part_valid_set=random_split(valid_set,[len_valid,len(valid_set)-len_valid])[0]
valid_loader=DataLoader(part_valid_set, batch_size=1,shuffle=True)
#training and validating
config=DistilBertConfig.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
config.max_position_embeddings=4096
model = DistilBertForSequenceClassification(config).to(0)
optimizer=opt.Adam(model.parameters(),lr=0.00001)
Loss_f=nn.CrossEntropyLoss()
epochs=10
train_acc=[]
valid_acc=[]
max_acc=0
for i in range(0,epochs):
    print("=======epoch"+str(i)+" start=======")
    x=0
    correct_case=0
    model.train()
    for text,label in train_loader:
        input =text.squeeze(1).to(0)
        optimizer.zero_grad()
        pred=model(input).logits.to(0)
        label=label.to(0)
        loss=Loss_f(pred,label).to(0)
        loss.backward()
        optimizer.step()
        if (pred.argmax()==label.argmax()):
            correct_case+=1
    print ("train_acc="+str(correct_case/len_train))
    train_acc.append(correct_case/len_train)
    correct_case=0
    model.eval()
    for text,label in valid_loader:
        input =text.squeeze(1).to(0)
        try:
            pred=model(input).logits.squeeze(0).to(0)
            label=label.to(0).squeeze(0)
            if (pred.argmax()==label.argmax()):
                correct_case+=1
        except RuntimeError as e:
                print(str(e))
                continue
    valid_acc.append(correct_case/len_valid)
    if(correct_case/len_valid>max_acc):
        max_acc=correct_case/len_valid
        torch.save(model,"best.pt")
    print ("valid_acc="+str(correct_case/len_valid))

#plotting figure
    
from matplotlib import pyplot as plt
x=np.arange(epochs)
plt.plot(x,train_acc,c='red',label="train")
plt.plot(x,valid_acc,c='blue',label="valid")
plt.legend()
plt.title("accuracy-epoch")
plt.show()

