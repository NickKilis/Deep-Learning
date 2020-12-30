#-----------------------------IMPORT LIBRARIES-------------------------------#
import torch 
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import random
import itertools
import seaborn as sns
import time,math
import matplotlib.pyplot as plt
from matplotlib import style
from collections import deque
import numpy as np
from sklearn import preprocessing
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.nn import Sequential,Dropout,LSTM,LSTMCell,BatchNorm1d
from torch.utils import checkpoint,data
from torch.autograd import Variable
from multiprocessing import cpu_count
from torch.optim.lr_scheduler import _LRScheduler
from sklearn.metrics import confusion_matrix
#----------------------------------------------------------------------------#
#-----------------------------DEFINE FUNCTIONS-------------------------------#
#-----------------------------CONFUSION MATRIX-------------------------------#
def computeAndPlotCM(y_test_batches,y_predicted_batches,model_name,class_names): 
    y_test = list(itertools.chain(*y_test_batches))
    y_predicted = list(itertools.chain(*y_predicted_batches))
    cm=confusion_matrix(y_test,y_predicted)
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names )
    fig = plt.figure(figsize = (8,8))
    plt.title('Confusion Matrix \n' + model_name) 
    Ηeatmap = sns.heatmap(df_cm, annot=True, fmt="d",cmap='OrRd',linewidths=1,linecolor='k',square=True)#,mask=False, vmin=-1, vmax=1,cbar_kws={"orientation": "vertical"},cbar=True)
    # fix for mpl bug that cuts off top/bottom of seaborn viz
    b, t = plt.ylim() # discover the values for bottom and top
    b += 0.5 # Add 0.5 to the bottom
    t -= 0.5 # Subtract 0.5 from the top
    plt.ylim(b, t) # update the ylim(bottom, top) values
    plt.show()
#----------------------------------------------------------------------------#
#-----------------------------PLOT GRAPHS------------------------------------#   
def create_acc_loss_graph(model_name):
    epochs = []
    accuracies = []
    losses = []        
    val_accuracies=[]
    val_losses=[]
    with open("dataset-cryptocurrency/log_loss_acc/"+model_name+".log", "r") as f:
        count = 0
        for line in f:
            count+=1
            if count % 2 == 0: #this is the remainder operator
                # VAL PHASE
                name, epoch, loss, acc = line.split("\t")
                val_accuracies.append(float(acc))
                val_losses.append(float(loss))
            else:
                # TRAIN PHASE
                name, epoch, loss, acc = line.split("\t")
                epochs.append(float(epoch))
                accuracies.append(float(acc))
                losses.append(float(loss))   
    fig = plt.figure()
    ax1 = plt.subplot2grid((2,1), (0,0))
    ax2 = plt.subplot2grid((2,1), (1,0), sharex=ax1)
    ax1.set_title(name)
    ax1.plot(epochs, accuracies, label="Training accuracy")
    ax1.plot(epochs, val_accuracies, label="Validation accuracy")
    ax1.legend(loc=4)
    plt.grid()
    ax1.set_xticks(np.arange(0, len(epochs), step=1))
    ax2.plot(epochs,losses, label="Training loss")
    ax2.plot(epochs,val_losses, label="Validation loss")
    ax2.legend(loc=1)
    ax2.set_xticks(np.arange(0, len(epochs), step=1))
    plt.grid()
    plt.show()
#----------------------------------------------------------------------------# 
#----------------------------PREPROCESSING-----------------------------------#
# classify each sequence to one of the 2 categories (buy/sell)
def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0
# scaling data, create sequencies, balance them and shuffle them
def preprocess_df(df,name):
    df = df.drop("future", 1)
    for col in df.columns:
        if col != "target":
            df[col] = df[col].pct_change()
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values)
    df.dropna(inplace=True)
    
    sequential_data = []
    prev_days = deque(maxlen=sequence_length)
    for i in df.values:
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == sequence_length:
            sequential_data.append([np.array(prev_days), i[-1]])
    random.shuffle(sequential_data)
    
    buys = []
    sells = []
    for seq, target in sequential_data:
        if target == 0:
            sells.append([seq, target]) 
        elif target == 1:
            buys.append([seq, target])
    random.shuffle(buys)
    random.shuffle(sells)
    # display the reduced dataset's volume
    print("Processing "+name+"...")
    print("The length of \'buys\' is : ",len(buys))
    print("The length of \'sells\' is : ",len(sells))  
    lower = min(len(buys), len(sells))
    print("For balancing them, we define a new length for both, which is : ",lower)
    buys = buys[:lower]
    sells = sells[:lower]
    sequential_data = buys+sells
    random.shuffle(sequential_data)
    X = []
    y = []
    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)
    return(np.array(X), y)
#----------------------------------------------------------------------------# 
#--------------------------CHECK FOR NANs AND INFs---------------------------#
# check if a dataframe has NaN or inf values in it
def check_if_nan_or_inf(df):
    print("")
    if np.any(np.isnan(df)):
        print("The dataframe has NaN values in it!")
        # df=df.dropna(inplace=True)
        # print("Removing NaNs...")
        # return df
    else: 
        print("The dataframe does not have NaN values in it!")
    if np.all(np.isfinite(df)):
        print("The dataframe does not have inf values in it!")    
        # df=df.replace([np.inf, -np.inf], np.nan)
        # df=df.dropna(axis=0)
        # print("Removing infs...")
        # return df
    else: 
        print("The dataframe has inf values in it!")   
#----------------------------------------------------------------------------# 
#--------------------TRASFORM Xs AND Ys INTO TENSORS-------------------------#
def transform_into_tensors(train_x,train_y,batch_size,name):
    print("The "+name+"_x has a shape          : ",train_x.shape)
    train_x=np.expand_dims(train_x, axis=2)
    print("The "+name+"_x has a new shape      : ",train_x.shape)
    
    print("The "+name+"_y has a shape          : ",np.array(train_y).shape)
    train_y=np.expand_dims(train_y, axis=1)
    print("The "+name+"_y has a new shape      : ",np.array(train_y).shape)
    
    tensor_x_train=torch.tensor(train_x, dtype=torch.float32)#.unsqueeze(1)
    tensor_y_train=torch.tensor(train_y, dtype=torch.long)
    print("The "+name+"_x tensor has a shape   : ",tensor_x_train.shape)
    print("The "+name+"_y tensor has a shape   : ",tensor_y_train.shape)
    
    train_dataset=data.TensorDataset(tensor_x_train,tensor_y_train)
    train_dataloader=data.DataLoader(train_dataset,batch_size=batch_size)
    dataiter = iter(train_dataloader)
    inputs, labels = dataiter.next()
    print("The "+name+" x batches have a shape : ",inputs.shape)
    print("The "+name+" y batches have a shape : ",labels.shape)
    print("")
    return(train_x,tensor_x_train,tensor_y_train,train_dataset,train_dataloader)
#----------------------------------------------------------------------------# 
#--------------------LR SCHEDULER AND LR FINDER------------------------------#
class CyclicLR(_LRScheduler):
    def __init__(self, optimizer, schedule, last_epoch=-1):
        assert callable(schedule)
        self.schedule = schedule
        super().__init__(optimizer, last_epoch)
    def get_lr(self):
        return [self.schedule(self.last_epoch, lr) for lr in self.base_lrs]
    
def cosine(t_max, eta_min=0):
    def scheduler(epoch, base_lr):
        t = epoch % t_max
        return eta_min + (base_lr - eta_min)*(1 + np.cos(np.pi*t/t_max))/2
    return scheduler

def cyclical_lr(stepsize, min_lr=0.00005, max_lr=0.005):
    # Scaler: we can adapt this if we do not want the triangular CLR
    scaler = lambda x: 1.
    # Lambda function to calculate the LR
    lr_lambda = lambda it: min_lr + (max_lr - min_lr) * relative(it, stepsize)
    # Additional function to see where on the cycle we are
    def relative(it, stepsize):
        cycle = math.floor(1 + it / (2 * stepsize))
        x = abs(it / stepsize - 2 * cycle + 1)
        return max(0, (1 - x)) * scaler(cycle)
    return(lr_lambda)

def find_lr(model,train_dataloader,num_epochs,optimizer,criterion,sched):
    lr_find_loss = []
    lr_find_lr = []
    iter = 0
    smoothing = 0.05
    for i in range(num_epochs):
        print("epoch {}".format(i))
        for inputs, labels in train_dataloader:
            # Send to device
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Training mode and zero gradients
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            # Get outputs to calc loss
            outputs = model(inputs)
            loss = criterion(outputs, labels.squeeze())
            #torch.nn.utils.clip_grad_value_(model.parameters(), 0.2)
            # Backward pass
            loss.backward()
            optimizer.step()
            # Update LR
            sched.step()
            lr_step = optimizer.state_dict()["param_groups"][0]["lr"]
            lr_find_lr.append(lr_step)
            # smooth the loss
            if iter==0:
              lr_find_loss.append(loss)
            else:
              loss = smoothing  * loss + (1 - smoothing) * lr_find_loss[-1]
              lr_find_loss.append(loss)
            iter += 1
    return(lr_find_loss,lr_find_lr)
#----------------------------------------------------------------------------# 
#-------------------------MODEL ARCHITECTURE---------------------------------#
def select_loss_function(select_loss):
    if select_loss=='CrossEntropyLoss':
        loss_criterion = nn.CrossEntropyLoss()
    elif select_loss=='BCELoss':
        loss_criterion = nn.BCELoss()
    elif select_loss=='MSELoss':
        loss_criterion = nn.MSELoss(size_average=True)    
    elif select_loss=='BCEWithLogitsLoss':
        loss_criterion = nn.BCEWithLogitsLoss()  
    elif select_loss=='NLLLoss':
        loss_criterion = nn.NLLLoss(reduction='sum') # use together with F.log_softmax (this combination is identical to CrossEntropyLoss)  
    else:
        print("Loss function with this name was not found!")
    return(loss_criterion)

def select_optim(select_optimizer):
    if select_optimizer=='Adam':
        optimizer=optim.Adam(model.parameters(),lr=learning_rate)
    elif select_optimizer=='Adagrad':
        optimizer=optim.Adagrad(model.parameters(),lr=learning_rate)
    elif select_optimizer=='SGD':
        optimizer=optim.SGD(model.parameters(),lr=learning_rate)
    elif select_optimizer=='RMSprop':
        optimizer=optim.RMSprop(model.parameters(),lr=learning_rate)
    elif select_optimizer=='LBFGS':
        optimizer=optim.LBFGS(model.parameters(),lr=learning_rate)
    else:
        print("Optimizer with this name was not found!")    
    return(optimizer)

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_layer,dropout_lstm, num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,dropout=dropout_lstm, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.drop_layer = nn.Dropout(p=dropout_layer)
    def forward(self, x):
        h0, c0 = self.init_hidden(x)
        out, (hn, cn) = self.rnn(x, (h0, c0))
        #out=self.bn1(out)
        out = self.drop_layer(out)
        out = self.relu(out)
        out = self.fc(out[:, -1, :])
        return out
    def init_hidden(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        return [t for t in (h0, c0)]
#----------------------------------------------------------------------------#
#------------------------------TRAINING-TESTING------------------------------#
def train_model(model_name,train_dataloader,valid_dataloader,criterion,opt,num_epochs,patience,lr,toggle_cyclic,sched,clip):
    train_losses=[]
    valid_losses=[]
    train_accs=[]
    valid_accs=[]
    y_true=[]
    y_predicted=[]
    best_acc = 0
    trials =  0
    overfit_stop=False
    print('Start model training...')
    since = time.time()
    with open("./dataset-cryptocurrency/log_loss_acc/"+model_name+".log", "a") as f:
        for epoch in range(0, num_epochs):
            #print("Epoch %d / %d" % (epoch,num_epochs))
            #print("-"*15)
            correct_train, total_train = 0, 0
            for i, (x_batch, y_batch) in enumerate(train_dataloader):
                #model.init_hidden(x_batch)
                model.train()
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                
                model.zero_grad()
                preds = model(x_batch)
                loss = criterion(preds, y_batch.squeeze())
    
                opt.zero_grad()
                loss.backward()
                
                # # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                # for p in model.parameters():
                #     p.data.add_(-lr, p.grad)
                
                # enable-disable gradient clipping
                #torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=clip)
                
                opt.step()
                if toggle_cyclic:
                    sched.step()
                
                preds = F.log_softmax(preds, dim=1).argmax(dim=1)
                total_train += y_batch.size(0)
                correct_train += (preds == y_batch.squeeze()).sum().item()
                y_true.append(y_batch.data.cpu().numpy())
                y_predicted.append(preds.cpu().numpy())
            acc_train = correct_train / total_train
            train_losses.append(loss.item())
            train_accs.append(acc_train)
            f.write("{}\t{}\t{:.4f}\t{:.4f}\n".format(model_name,epoch,train_losses[epoch], train_accs[epoch]) )
            if epoch % 1 == 0:
                print('Epoch {:3d} Training Loss  : {:.4f}\t Training accuracy: {:.4f}'.format(epoch, loss, acc_train))
                #print(f'Epoch: {epoch:3d}. Training Loss: {loss.item():.4f}. Training accuracy: {acc_train:2.2%}')    
                
            model.eval()
            correct_val, total_val = 0, 0
            for x_val, y_val in valid_dataloader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                preds = model(x_val)
                loss_val = criterion(preds,y_val.squeeze())
                
                preds = F.log_softmax(preds, dim=1).argmax(dim=1)
                total_val += y_val.size(0)
                correct_val += (preds == y_val.squeeze()).sum().item()
            acc_val = correct_val / total_val
            valid_accs.append(acc_val)
            valid_losses.append(loss_val.item())
            f.write("{}\t{}\t{:.4f}\t{:.4f}\n".format(model_name,epoch,valid_losses[epoch], valid_accs[epoch]) )
            if epoch % 1 == 0:
                print('Epoch {:3d} Validation Loss: {:.4f}\t Validation accuracy: {:.4f}'.format(epoch, loss_val, acc_val))
                #print(f'Epoch: {epoch:3d}. Loss: {loss_val.item():.4f}. Acc.: {acc_val:2.2%}')
            # OVERFIT STOP
            if epoch > 0:    
                if valid_losses[-2]+0.005<valid_losses[epoch]:
                    #print("Training stopped because the previous validation loss: "+str(valid_losses[-2]+0.005)+ " is less than the next: "+str(valid_losses[epoch]))
                    print("OVERFIT WARNING - the previous validation loss: "+str(valid_losses[-2]+0.005)+ " is less than the next: "+str(valid_losses[epoch]))
                    overfit_stop=True
            # if overfit_stop:
            #     print(f'OVERFIT STOP - on epoch {epoch}')
            #     break
             # SAVE BEFORE PLATEU
            if acc_val > best_acc:
                trials = 0
                best_acc = acc_val
                torch.save(model.state_dict(), 'dataset-cryptocurrency/best_model_save/best_model.pth')
                print('SAVE - on Epoch {:3d} best model with validation accuracy: {:.4f}'.format(epoch,best_acc))
                #print(f'Epoch {epoch} best model saved with validation accuracy: {best_acc:.4f')
            else:
                trials += 1
                if trials >= patience:
                    print(f'EARLY STOP - on epoch {epoch}')
                    break
            
    # compute and plot the confusion matrix
    class_names=["buy","sell"]
    computeAndPlotCM(y_true,y_predicted,model_name+" TRAIN",class_names)
    print('The training is finished! Restoring the best model weights')
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print("")
    return(train_losses,valid_losses,train_accs,valid_accs,y_true,y_predicted)

def test_model(model_name,model,test_dataloader):
    with open("./dataset-cryptocurrency/log_loss_acc_test/"+model_name+".log", "a") as f:
        model.eval()
        y_test2=[]
        y_predicted2=[]
        accuracy = 0
        model.eval()
        correct_test, total_test = 0, 0
        for x_test, y_test in test_dataloader:
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            preds = model(x_test)
            y_test2.append(y_test.data.cpu().numpy())
            y_predicted2.append(preds.max(1)[1].cpu().numpy())
            preds = F.log_softmax(preds, dim=1).argmax(dim=1)
            total_test += y_test.size(0)
            correct_test += (preds == y_test.squeeze()).sum().item()               
        acc_test = correct_test / total_test
        print("Test accuracy: {:.4f}".format(acc_test))
        f.write("{}\t{:.4f}\n".format(model_name,acc_test))
        class_names=["buy","sell"]
        # compute and plot the confusion matrix
        computeAndPlotCM(y_test2,y_predicted2,model_name+" TEST",class_names)
#----------------------------------------------------------------------------#        
#---------------------------------MAIN---------------------------------------#
if __name__ == "__main__":
#----------------------------------------------------------------------------# 
#----------------------------DEVICE SELECTION--------------------------------#
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
        print("Running on the GPU device!")
        print("The number of devices you have is : ",torch.cuda.device_count())
        print("")
    else:
        device = torch.device("cpu")
        print("Running on the CPU device!")      
#----------------------------------------------------------------------------# 
#----------------------------CONFIGURATIONS----------------------------------#
    # CONSTANTS    
    ratio_to_predict = "LTC-USD"
    #ratio_to_predict = "BTC-USD"
    # ratio_to_predict = "BCH-USD"
    # ratio_to_predict = "ETH-USD"
    num_classes = 2
    input_size = 1
    
    # VARIABLES TO BE TUNED
    # architecture 
    hidden_size = 128
    dropout_layer=0
    dropout_lstm=0
    num_layers = 2
    sequence_length = 128
    future_length = 2 
    # training
    num_epochs = 50
    batch_size = 32
    select_loss='CrossEntropyLoss'
    select_optimizer='Adam'
    patience = 5                                    
    learning_rate = 0.0001
    clip=0.5
    toggle_lr_find=False
    toggle_cyclic=True
    toggle_train=True
    toggle_test=True
    toggle_plot=True
    
    model_name = f"{ratio_to_predict}-SEQ_LEN-{sequence_length}-FUT_LEN-{future_length}-HS-{hidden_size}-NL-{num_layers}-DRL-{dropout_layer}-DRLSTM-{dropout_lstm}-BS-{batch_size}-LR-{str(learning_rate).replace('.', '')}-{int(time.time())}"
    print("")
    print(model_name)
    print("NUMBER OF EPOCHS     : ",num_epochs)
    print("SEQUENCE LENGTH      : ",sequence_length)
    print("FUTURE LENGTH        : ",future_length)
    print("INPUT SIZE           : ",input_size)
    print("HIDDEN SIZE          : ",hidden_size)
    print("NUMBER OF LAYERS     : ",num_layers)
    print("NUMBER OF CLASSES    : ",num_classes)
    print("BATCH SIZE           : ",batch_size)
    print("LEARNING RATE (fixed): ",learning_rate)
    print("DROPOUT (LAYER)      : ",dropout_layer)
    print("DROPOUT (LSTM)       : ",dropout_lstm)
    print("PATIENCE             : ",patience)
    print("LOSS FUNCTION        : ",select_loss)
    print("OPTIMIZER            : ",select_optimizer)
    print("TOGGLE TRAIN         : ",toggle_train)
    print("TOGGLE TEST          : ",toggle_test)
    print("TOGGLE PLOT          : ",toggle_plot)
    print("TOGGLE LR FINDER     : ",toggle_lr_find)
    print("TOGGLE CYCLIC        : ",toggle_cyclic)
    
    main_df = pd.DataFrame()
    ratios = ["BTC-USD", "LTC-USD", "BCH-USD", "ETH-USD"]
    for ratio in ratios:
        #print(ratio)
        # get the full path to the file.
        dataset = f'dataset-cryptocurrency/{ratio}.csv'  
        # read in specific file
        df = pd.read_csv(dataset, names=['time', 'low', 'high', 'open', 'close', 'volume'])  
        # rename volume and close to include the ticker so we can still which close/volume is which:
        df.rename(columns={"close": f"{ratio}_close", "volume": f"{ratio}_volume"}, inplace=True)
        df.set_index("time", inplace=True)
        df = df[[f"{ratio}_close", f"{ratio}_volume"]]
        if len(main_df)==0:
            main_df = df
        else:
            main_df = main_df.join(df)
    main_df.fillna(method="ffill", inplace=True)
    main_df.dropna(inplace=True)
    main_df['future'] = main_df[f'{ratio_to_predict}_close'].shift(-future_length)
    main_df['target'] = list(map(classify, main_df[f'{ratio_to_predict}_close'], main_df['future']))
    main_df.replace([np.inf, -np.inf], np.nan).dropna(inplace=True,axis=0)
    main_df.dropna(inplace=True)
    times = sorted(main_df.index.values)
    # check for non valid numbers
    check_if_nan_or_inf(main_df)

#----------------------------------------------------------------------------# 
#-------------------SPLIT INTO TRAIN VALIDATION TEST-------------------------#
    # create a threshold to separate the training data and validation data
    last_5pct = times[-int(0.05*len(times))]                                   # get the last 5% of the times
    last_10pct = times[-int(0.1*len(times))]                                   # get the last 10% of the times
    
    # split into training and validation and test
    training_main_df = main_df[(main_df.index < last_10pct)]   
    validation_main_df =main_df.loc[last_10pct:last_5pct]                                       
    test_main_df=main_df[(main_df.index >= last_5pct)]
    
    fig1=plt.figure(figsize=(8,8))
    plt.plot(main_df[ratio_to_predict+"_close"][(main_df.index < last_10pct)],"k",label="Training set")
    plt.plot(main_df[ratio_to_predict+"_close"].loc[last_10pct:last_5pct],"g",label="Validation set")
    plt.plot(main_df[ratio_to_predict+"_close"][(main_df.index >= last_5pct)],"r",label="Test set")
    plt.grid()
    plt.legend()
    plt.show()
    
    train_x,train_y=preprocess_df(training_main_df,"train set")
    valid_x,valid_y=preprocess_df(validation_main_df,"validation set")
    test_x,test_y=preprocess_df(test_main_df,"test set")
    print("")
    print("The length of the training set is   : ",len(train_x))
    print("The length of the validation set is : ",len(valid_x))
    print("The length of the test set is       : ",len(test_x))
    print("The length of the class \'buy\' in training set is        : ",train_y.count(0))
    print("The length of the class \'dont buy\' in training set is   : ",train_y.count(1))
    print("The length of the class \'buy\' in validation set is      : ",valid_y.count(0))
    print("The length of the class \'dont\' buy in validation set is : ",valid_y.count(1))
    print("The length of the class \'buy\' in test set is            : ",test_y.count(0))
    print("The length of the class \'dont\' buy in test set is       : ",test_y.count(1))
    print("")
    #----------------------------------------------------------------------------# 
    #--------------SELECT THE COLUMN OF Xs FOR THE SELECTED COIN-----------------#
    if ratio_to_predict=="BTC-USD":
        train_x=train_x[:,:,0]
        valid_x=valid_x[:,:,0]
        test_x=test_x[:,:,0]
    if ratio_to_predict=="LTC-USD":
        train_x=train_x[:,:,2]
        valid_x=valid_x[:,:,2]
        test_x=test_x[:,:,2]
    if ratio_to_predict=="BCH-USD":
        train_x=train_x[:,:,4]
        valid_x=valid_x[:,:,4]
        test_x=test_x[:,:,4]
    if ratio_to_predict=="ETH-USD":
        train_x=train_x[:,:,6]
        valid_x=valid_x[:,:,6]
        test_x=test_x[:,:,6]
    # transform into tensors    
    train_x,tensor_x_train,tensor_y_train,train_dataset,train_dataloader=transform_into_tensors(train_x,train_y,batch_size,"train")
    valid_x,tensor_x_valid,tensor_y_valid,valid_dataset,valid_dataloader=transform_into_tensors(valid_x,valid_y,batch_size,"valid")
    test_x,tensor_x_test,tensor_y_test,test_dataset,test_dataloader=transform_into_tensors(test_x,test_y,batch_size,"test")

    # define the model and print its parameters
    model = LSTMClassifier(input_size, hidden_size, num_layers,dropout_layer,dropout_lstm, num_classes).to(device)
    
    print(model)
    print(len(list(model.parameters())))
    for i in range(len(list(model.parameters()))):
        print(list(model.parameters())[i].size())
    print("")
    
    loss_criterion=select_loss_function(select_loss)
    optimizer=select_optim(select_optimizer)
    
    if toggle_lr_find:
        num_epochs = 2
        start_lr=0.005
        end_lr=0.00005
        lr_find_epochs=num_epochs
        lr_lambda = lambda x: math.exp(x * math.log(end_lr / start_lr) / (lr_find_epochs * len(train_dataloader)))
        sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        lr_find_loss,lr_find_lr=find_lr(model,train_dataloader,num_epochs,optimizer,loss_criterion,sched)
        
        tmp_list=[]
        for i in range(len(lr_find_loss)):
            tmp_list.append(lr_find_loss[i].data.cpu().numpy())
        print(len(tmp_list))
        
        plt.figure()
        plt.title("Learning Rate finder")
        plt.plot(tmp_list)
        plt.ylabel("Training loss")
        plt.xlabel("Learning rate [log]")
        plt.grid()
        plt.xscale("log")
        plt.show()    
        
        plt.figure()
        plt.plot(lr_find_lr)
        plt.show()  
    
    # if toggle_cyclic:
    #     # # # scheduler config
    #     n = 100
    #     sched = cosine(n)
    #     lrs = [sched(t, 1) for t in range(n * 4)]
    #     plt.figure()
    #     plt.plot(lrs)
    #     plt.show
        
    if toggle_train:
        if toggle_cyclic:
            step_size = 4*len(train_dataloader)
            end_lr=0.05
            factor=10
            clr = cyclical_lr(step_size, min_lr=end_lr/factor, max_lr=end_lr)
            sched = torch.optim.lr_scheduler.LambdaLR(optimizer, [clr])
            
            #iterations_per_epoch = len(train_dataloader)
            #sched = CyclicLR(optimizer, cosine(t_max=iterations_per_epoch * 2, eta_min=learning_rate/100))
            train_losses,valid_losses,train_accs,valid_accs,y_true,y_predicted=train_model(model_name,train_dataloader,valid_dataloader,loss_criterion,optimizer,num_epochs,patience,learning_rate,toggle_cyclic,sched,clip)
        else:
            sched=None
            train_losses,valid_losses,train_accs,valid_accs,y_true,y_predicted=train_model(model_name,train_dataloader,valid_dataloader,loss_criterion,optimizer,num_epochs,patience,learning_rate,toggle_cyclic,sched,clip)
            
    if toggle_plot:
        style.use("ggplot")
        create_acc_loss_graph(model_name)
        
    if toggle_test:
        test_model(model_name,model,test_dataloader)