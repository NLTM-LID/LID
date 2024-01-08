################## used Library  ############################################################
import torch
import torch.nn as nn
from extract import extract_BNF
import os 
import numpy as np
import pandas as pd
from torch.autograd import Variable
import sys
import argparse
import matplotlib.pyplot as plt
# import sklearn.metrics

############ number of class and all #####################
e_dim = 64*2
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
#######################################################

#######################################################
################### uVector Class ####################
class LSTMNet(torch.nn.Module):
    def __init__(self):
        super(LSTMNet, self).__init__()
        self.lstm1 = nn.LSTM(80, 256,bidirectional=True)
        self.lstm2 = nn.LSTM(2*256, 64,bidirectional=True)
        # self.lstm3 = nn.LSTM(2*128, 64,bidirectional=True)
        
        self.fc_ha=nn.Linear(e_dim,128) 
        self.fc_1= nn.Linear(128,1)           
        self.sftmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x, _ = self.lstm1(x) 
        x, _ = self.lstm2(x)
        # x, _ = self.lstm3(x)
        ht = x[-1]
        ht = torch.unsqueeze(ht, 0)      
        ha = torch.tanh(self.fc_ha(ht))
        alpha = self.fc_1(ha)
        al = self.sftmax(alpha) # Attention vector
        
        T = list(ht.shape)[1]  #T=time index
        batch_size = list(ht.shape)[0]
        dim = list(ht.shape)[2]
        c = torch.bmm(al.view(batch_size, 1, T),ht.view(batch_size,T,dim))
        #print('c size',c.size())        
        e = torch.squeeze(c,0)
        return e

class CCSL_Net(nn.Module):
    def __init__(self, model1,model2):
        super(CCSL_Net, self).__init__()
        self.model1 = model1
        self.model2 = model2

        self.att1=nn.Linear(e_dim,100) 
        self.att2= nn.Linear(100,1)           
        self.bsftmax = torch.nn.Softmax(dim=1)

        self.lang_classifier= nn.Sequential()
        self.lang_classifier.add_module('fc1',nn.Linear(e_dim, Nc, bias=False))     
        
        
    def forward(self, x1,x2):
        e1 = self.model1(x1)
        e2 = self.model2(x2)        
        ht_e = torch.cat((e1,e2), dim=0)  
        ht_e = torch.unsqueeze(ht_e, 0) 
        ha_e = torch.tanh(self.att1(ht_e))
        alp = torch.tanh(self.att2(ha_e))
        al= self.bsftmax(alp)
        Tb = list(ht_e.shape)[1] 
        batch_size = list(ht_e.shape)[0]
        D = list(ht_e.shape)[2]
        u_vec = torch.bmm(al.view(batch_size, 1, Tb),ht_e.view(batch_size,Tb,D))
        u_vec = torch.squeeze(u_vec,0)
        
        lang_output = self.lang_classifier(u_vec)      # Output layer   
        
        return lang_output
##########################################################
                
######################## uVector ####################
##########################################################

def uvector(fn):
    model1 = LSTMNet()
    model2 = LSTMNet()

    model = CCSL_Net(model1, model2)
    
    #model.cuda()    
    path = "./model/uVector_base_12_class_e18.pth"  ## Load model
    #print(path)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
       
    X1, X2 = lstm_data(fn)
    
    X1 = np.swapaxes(X1,0,1)
    X2 = np.swapaxes(X2,0,1)
    x1 = Variable(X1, requires_grad=False)
    x2 = Variable(X2, requires_grad=False)
    o1 = model.forward(x1, x2)
    output =  o1.detach().cpu().numpy()[0]
    pred_all = np.exp(output) / np.sum(np.exp(output))
    Pred = np.argmax(o1.detach().cpu().numpy(), axis=1)
    
    return Pred[0], pred_all

# Function to perform audio feature extraction and language identification using uVector
def classification_uvector(audio_paths):
    pred_labels = []
    for audio_path in audio_paths:
        # To get the BNF features of the given audio file
        bnf = extract_BNF(audio_path)

        # Perform language identification using uvector models
        lang, prob_all_lang = uvector(bnf)

        # Print language identification results
        # print(prob_all_lang)

        # Define language mappings for display
        lang2id = {'asm': 0, 'ben': 1, 'eng': 2, 'guj': 3, 'hin': 4, 'kan': 5, 'mal': 6, 'mar': 7, 'odi': 8, 'pun': 9, 'tel': 10, 'tam': 11}
        id2lang = {0: 'asm', 1: 'ben', 2: 'eng', 3: 'guj', 4: 'hin', 5: 'kan', 6: 'mal', 7: 'mar', 8: 'odi', 9: 'pun', 10: 'tel', 11: 'tam'}

        # Get the identified language
        Y1 = id2lang[lang]
        pred_labels.append(Y1)
        # Display pridected language information using a message box
        print("The predicted language of {} audio is {}".format(audio_path, Y1))
    
    if len(audio_paths) == 1:
        # Plot the language identification probabilities
        fig = plt.figure(figsize=(10, 5))
        plt.bar(lang2id.keys(), prob_all_lang, color='maroon', width=0.4)
        plt.yscale("log")
        plt.xlabel("Languages")
        plt.ylabel("Language Identification Probability (in log scale)")
        plt.title("Language Identification Probability of Spoken Audio using uVector")
        plt.show()
    else:
        # Create a DataFrame
        df = pd.DataFrame({'filename': audio_paths, 'predicted_language': pred_labels})
        # Specify the CSV file path
        csv_file_path = 'predicted_lang.csv'
        # Save the DataFrame to a CSV file
        df.to_csv(csv_file_path, index=False)
        print('Data has been saved to {}'.format(csv_file_path))


def main():
    parser = argparse.ArgumentParser(description='Spoken language identification (uVector) script with command line options.')

    # Command line options
    parser.add_argument('path', help='Path to file or directory')
    args = parser.parse_args()
    path = args.path

    # Get the list of all files from the path
    file_list = []
    # When path is a file
    if os.path.isfile(path) and path.endswith(".wav"):
        file_list.append(path)
    # When path is a directory
    elif os.path.isdir(path):
        for filename in os.listdir(path):
            if filename.endswith(".wav"):
                file_path = os.path.join(path, filename)
                file_list.append(file_path)
    else:
        print("Error: {} is not a valid file/directory path.".format(path))
    
    classification_uvector(file_list)


if __name__ == "__main__":
    main()
