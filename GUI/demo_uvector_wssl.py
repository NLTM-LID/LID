################## used Library  ############################################################
import torch
import torch.nn as nn
import os 
import numpy as np
import pandas as pd
from torch.autograd import Variable
# import sklearn.metrics

############ number of class and all #####################
Nc = 12 # Number of language classes 
n_epoch = 20 # Number of epochs
IP_dim = 80 # number of input dimension
##########################################

##########################################
#### Function to return data (vector) and target label of a csv (BNF/MFCC features) file
look_back1 = 20 
look_back2 = 50

def lstm_data(f):
    df = pd.read_csv(f, encoding='utf-16', usecols=list(range(0,80)))
    dt = df.astype(np.float32)
    X = np.array(dt)
    
    Xdata1=[]
    Xdata2=[] 
      
    mu = X.mean(axis=0)
    std = X.std(axis=0)
    np.place(std, std == 0, 1)
    X = (X - mu) / std 
    f1 = os.path.splitext(f)[0]     
    
    for i in range(0,len(X)-look_back1,1):    #High resolution low context        
        a=X[i:(i+look_back1),:]        
        Xdata1.append(a)
    Xdata1=np.array(Xdata1)

    for i in range(0,len(X)-look_back2,2):     #Low resolution long context       
        b=X[i+1:(i+look_back2):3,:]        
        Xdata2.append(b)
    Xdata2=np.array(Xdata2)
    
    Xdata1 = torch.from_numpy(Xdata1).float()
    Xdata2 = torch.from_numpy(Xdata2).float()
    
    return Xdata1,Xdata2
###############################################################################

#######################################################
################### uVector Class ####################
#################################################################################### Modifying e1e2inter2aa
class LSTMNet(torch.nn.Module):
    def __init__(self):
        super(LSTMNet, self).__init__()
        self.lstm1 = nn.LSTM(80, 256,bidirectional=True)
        self.lstm2 = nn.LSTM(2*256, 32,bidirectional=True)
               
        self.fc_ha=nn.Linear(2*32,100) 
        self.fc_1= nn.Linear(100,1)           
        self.sftmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x1, _ = self.lstm1(x) 
        x2, _ = self.lstm2(x1)
        ht = x2[-1]
        ht=torch.unsqueeze(ht, 0)        
        ha= torch.tanh(self.fc_ha(ht))
        alp= self.fc_1(ha)
        al= self.sftmax(alp) 
        
        T=list(ht.shape)[1]  
        batch_size=list(ht.shape)[0]
        D=list(ht.shape)[2]
        c=torch.bmm(al.view(batch_size, 1, T),ht.view(batch_size,T,D))  # Self-attention on LID-seq-senones to get utterance-level embedding (e1/e2)      
        c = torch.squeeze(c,0)        
        return (c)

class MSA_DAT_Net(nn.Module):
    def __init__(self, model1,model2):
        super(MSA_DAT_Net, self).__init__()
        self.model1 = model1
        self.model2 = model2

        self.att1=nn.Linear(2*32,100) 
        self.att2= nn.Linear(100,1)           
        self.bsftmax = torch.nn.Softmax(dim=1)

        self.lang_classifier= nn.Sequential()
        self.lang_classifier.add_module('fc1',nn.Linear(2*32, Nc, bias=True))
        
    def forward(self, x1,x2):
        u1 = self.model1(x1)
        u2 = self.model2(x2)        
        ht_u = torch.cat((u1,u2), dim=0)  
        ht_u = torch.unsqueeze(ht_u, 0) 
        ha_u = torch.tanh(self.att1(ht_u))
        alp = torch.tanh(self.att2(ha_u))
        al= self.bsftmax(alp)
        Tb = list(ht_u.shape)[1] 
        batch_size = list(ht_u.shape)[0]
        D = list(ht_u.shape)[2]
        u_vec = torch.bmm(al.view(batch_size, 1, Tb),ht_u.view(batch_size,Tb,D)) # Self-attention combination of e1 and e2 to get u-vec
        u_vec = torch.squeeze(u_vec,0)
        
        lang_output = self.lang_classifier(u_vec)      # Output layer  
        
        return (lang_output,u1,u2,u_vec)
###############################################################################################
                
######################## uVector ####################
##########################################################

def uvector_wssl(fn):
    model1 = LSTMNet()
    model2 = LSTMNet()

    model = MSA_DAT_Net(model1, model2)
    
    #model.cuda()    
    path = "./model/ZWSSL_20_50_e21.pth"  ## Load model
    #print(path)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
       
    X1, X2 = lstm_data(fn)
    
    X1 = np.swapaxes(X1,0,1)
    X2 = np.swapaxes(X2,0,1)
    x1 = Variable(X1, requires_grad=False)
    x2 = Variable(X2, requires_grad=False)
    o1,_,_,_ = model.forward(x1, x2)
    ### Get the prediction
    # output = np.argmax(o1.detach().cpu().numpy(), axis=1)
    output =  o1.detach().cpu().numpy()[0]
    pred_all = np.exp(output) / np.sum(np.exp(output))
    Pred = np.argmax(o1.detach().cpu().numpy(), axis=1)
    
    return Pred[0], pred_all
