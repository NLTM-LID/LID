################## used Library  ############################################################
import torch
import torch.nn as nn
import os 
import numpy as np
import pandas as pd
from torch.autograd import Variable
import sys
import argparse
import matplotlib.pyplot as plt
## Libraries from external python code
from extract import extract_BNF


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

# Function to perform audio feature extraction and language identification using WSSL uVector
def classification_wssl_uvector(audio_paths):
    pred_labels = []
    for audio_path in audio_paths:
        bnf = extract_BNF(audio_path)

        # Perform language identification using uvector models
        lang, prob_all_lang = uvector_wssl(bnf)

        # Print language identification results
        # print(prob_all_lang)

        # Define language mappings for display
        lang2id = {'asm': 0, 'ben': 1, 'eng': 2, 'guj': 3, 'hin': 4, 'kan': 5, 'mal': 6, 'mar': 7, 'odi': 8, 'pun': 9, 'tam': 10, 'tel': 11}
        id2lang = {0: 'asm', 1: 'ben', 2: 'eng', 3: 'guj', 4: 'hin', 5: 'kan', 6: 'mal', 7: 'mar', 8: 'odi', 9: 'pun', 10: 'tam', 11: 'tel'}

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
        plt.title("Language Identification Probability of Spoken Audio using WSSL uVector")
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
    parser = argparse.ArgumentParser(description='Spoken language identification (WSSL uVector) script with command line options.')

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

    # Call the function for classification
    classification_wssl_uvector(file_list)


if __name__ == "__main__":
    main()
    