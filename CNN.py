#----------------------------------------------------------------------------#
# HOW TO USE:
#
#   1. root_dir : choose the folder, which contains the spectrograms splitted into:
#                   a. train,
#                   b. val and
#                   c. test subfolders 
#                 and each subfolder contains folders of the 10 classes.
#   2. img_size         : resize the size of the spectrograms into a new one.
#   3. num_classes      : choose the number of classes.
#   4. bs               : choose a batch size.
#   5. learning_rate    : choose a learning rate.
#   6. num_epochs       : choose the number of epochs for training.
#   7. select_model     : select of of the pretrained models.
#   8. select_loss      : select a loss function.
#   9. select_optimizer : select an optimizer.
#   10. toggle_train    : enable training of the model.
#   11. toggle_test     : enable testing of the model.
#   12. toggle_plot     : enable ploting the training loss and accuracy.
#   13. toggle_load     : enable a model to be loaded.
#  
#----------------------------------------------------------------------------#
#----------------------------PRIOR INSTALLATIONS-----------------------------#
## pip install opencv-python
## pip install tqdm
#----------------------------------------------------------------------------#
#----------------------------IMPORT LIBRARIES--------------------------------#
from torchvision.datasets import ImageFolder
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor,Resize,Normalize
from torchvision import models,transforms,datasets
import os,time
from tqdm import tqdm
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
import torch
import copy
import math
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix
#----------------------------------------------------------------------------#
#----------------------------DEFINE FUNCTIONS--------------------------------#
def imshow(inp, title=None):
    '''
    # Grab some of the training data to visualize
    inputs, classes = next(iter(dataloaders['train']))
    
    # Now we construct a grid from batch
    out = torchvision.utils.make_grid(inputs)
    
    imshow(out, title=[class_names[x] for x in classes])
    '''
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # Pause a bit so that plots are updated

def visualize_model(model,data_loader,class_names, num_images=6):
    '''
    visualize_model(model,data_loader,class_names)
    plt.show()
    '''
    was_training = model.training
    model.eval()
    images_handeled = 0
    fig = plt.figure()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data_loader['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_handeled += 1
                ax = plt.subplot(num_images//2, 2, images_handeled)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_handeled == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

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

def create_acc_loss_graph(model_name):
    epochs = []
    accuracies = []
    losses = []        
    val_accuracies=[]
    val_losses=[]
    with open("saved_models/"+model_name+".log", "r") as f:
        count = 0
        for line in f:
            count+=1
            if count % 2 == 0: #this is the remainder operator
                # VAL PHASE
                name, epoch,phase, loss, acc = line.split("\t")
                val_accuracies.append(float(acc))
                val_losses.append(float(loss))
            else:
                # TRAIN PHASE
                name, epoch,phase, loss, acc = line.split("\t")
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

def select_pretrained_model(select_model):
    if select_model=='resnet18':
        model = models.resnet18(pretrained=True)
    elif select_model=='alexnet':
        model= models.alexnet(pretrained=True)
    elif select_model=='squeezenet':
        model= models.squeezenet1_0(pretrained=True)
    elif select_model=='vgg16':
        model= models.vgg16(pretrained=True)
    elif select_model=='densenet':
        model= models.densenet161(pretrained=True)
    elif select_model=='inception':
        model= models.inception_v3(pretrained=True)
    elif select_model=='googlenet':
        model= models.googlenet(pretrained=True)
    elif select_model=='shufflenet':
        model= models.shufflenet_v2_x1_0(pretrained=True)
    elif select_model=='mobilenet':
        model= models.mobilenet_v2(pretrained=True)
    elif select_model=='resnet34':
        model= models.resnet34(pretrained=True)
    elif select_model=='wide_resnet50_2':
        model= models.wide_resnet50_2(pretrained=True)
    elif select_model=='mnasnet':
        model= models.mnasnet1_0(pretrained=True)
    elif select_model=='resnet50':
        model = models.resnet50(pretrained=True)
    else:
        print("Pretrained model not found!")
    return(model)

def select_loss_function(select_loss):
    if select_loss=='CrossEntropyLoss':
        loss_criterion = nn.CrossEntropyLoss()
    else:
        print("Loss function not found!")
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
    return(optimizer)
#----------------------------------------------------------------------------#
#-------------------------LOAD A PRETRAINED MODEL----------------------------#
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def train_model(model,model_name,data_loaders,criterion,optimizer,epochs=5):
    train_losses=[]
    valid_losses=[]
    y_test=[]
    y_predicted=[]
    early_stop=False
    since = time.time()
    with open("./saved_models/"+model_name+".log", "a") as f:
        stop_loss=10000
        best_acc = 0.0
        for epoch in range(epochs):
            print("Epoch %d / %d" % (epoch,epochs-1))
            print("-"*15)
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode
                running_loss=0.0
                correct=0
                # Iterate over data.
                for batch_idx, (inputs, labels) in enumerate(data_loaders[phase]):
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)
                        y_test.append(labels.data.cpu().numpy())
                        y_predicted.append(preds.cpu().numpy())
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            train_losses.append(loss.item())
                            optimizer.step()
                                                  
                    running_loss +=loss.item()*inputs.size(0)
                    correct +=torch.sum(preds == labels.data)                  
                # total epoch loss and total epoch accuracy
                epoch_loss = running_loss / len(data_loaders[phase].dataset)
                epoch_acc = correct.double() / len(data_loaders[phase].dataset)
                
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                f.write("{}\t{}\t{}\t{:.4f}\t{:.4f}\n".format(model_name,epoch,phase, epoch_loss, epoch_acc) )
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                if phase== 'val' and epoch_loss > stop_loss:
                    print("")
                    print("Early stop at epoch : ",epoch)
                    early_stop=True
                else:
                    stop_loss=epoch_loss
            if early_stop:
                break    
        # print best validation accuracy of all epochs
        print('Best valid Accuracy of all epochs: {:4f}'.format(best_acc))
    time_elapsed = time.time() - since
    print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print("")
    # compute and plot the confusion matrix
    computeAndPlotCM(y_test,y_predicted,model_name+" TRAIN",class_names)
    #load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model,"./saved_models/"+model_name)
    return(model,train_losses,valid_losses,y_test,y_predicted)
    
if __name__ == "__main__":
#----------------------------------------------------------------------------#
#----------------------------CHECK DEVICE------------------------------------#
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
        print("Running on the GPU device!")
        print("The number of devices you have is : ",torch.cuda.device_count())
        print("")
    else:
        device = torch.device("cpu")
        print("Running on the CPU device!")
#----------------------------------------------------------------------------#
#----------------------------DATA TRANSFORMS---------------------------------#
    #select between 2 datasets
    root_dir="spec_6320_red_train_test_val"
    #root_dir="spec_6320_red_train_test_val"
    if root_dir =="spec_6320_green_train_test_val":
        dataset_name="green"
    elif root_dir =="spec_6320_red_train_test_val": 
        dataset_name="red"
    os.listdir(root_dir)
    img_size=224
    num_classes=10
    bs=70
    learning_rate=0.001
    num_epochs=15
    select_model='resnet50'
    select_loss='CrossEntropyLoss'
    select_optimizer='Adam'
    toggle_train=True
    toggle_test=True
    toggle_plot=True
    toggle_load=False
    if toggle_load:
        name="./saved_models/"+"model-resnet50-red-4epochs-70bs-0001lr-1589721252"
    model_name =f"model-{select_model}-{dataset_name}-{num_epochs}epochs-{bs}bs-{str(learning_rate).replace('.', '')}lr-{int(time.time())}"
    print("")
    print(model_name)
    print("DATASET NAME     : ",dataset_name) 
    print("MODEL NAME       : ",select_model) 
    print("IMAGE SIZE INPUT : ",str(img_size)+"x"+str(img_size))
    print("NUMBER OF CLASSES: ",num_classes)
    print("BATCH SIZE       : ",bs)
    print("LEARNING RATE    : ",learning_rate)
    print("LOSS FUNCTION    : ",select_loss)
    print("OPTIMIZER        : ",select_optimizer)
    print("NUMBER OF EPOCHS : ",num_epochs)
    print("TOGGLE TRAIN     : ",toggle_train)
    print("TOGGLE TEST      : ",toggle_test)
    print("TOGGLE PLOT      : ",toggle_plot)
    print("TOGGLE LOAD      : ",toggle_load)
    
    image_transforms = {
    'train': transforms.Compose([
        transforms.Resize((img_size,img_size)),
        #transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        #transforms.RandomRotation(degrees=15),
        #transforms.RandomHorizontalFlip(),
        #transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((img_size,img_size)),
        #transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((img_size,img_size)),
        #transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    }
    # create a data generator
    data_generator ={k: datasets.ImageFolder(os.path.join(root_dir,k),image_transforms[k]) for k in ["train", "val"]}
    class_names = data_generator['train'].classes
    # create a data loader
    data_loader= {k :DataLoader(data_generator[k],batch_size=bs,shuffle=True,num_workers=0) for k in ["train", "val"]}
    
    # if in phase load an existing model
    if toggle_load:
        print("You have selected the saved model : ",name.split("/")[-1])
        model=torch.load(name)
        #print(model)
        learning_rates_exp=np.linspace(1e-4,1e-2,5)**2
        learning_rates_linear=np.linspace(1e-8,1e-4,5)
        lr=learning_rates_linear
        optimizer = optim.Adam(
        [
            {"params": model.layer1.parameters(), "lr": lr[0]},
            {"params": model.layer2.parameters(), "lr": lr[1]},
            {"params": model.layer3.parameters(), "lr": lr[2]},
            {"params": model.layer4.parameters(), "lr": lr[3]},
            {"params": model.fc.parameters()    , "lr": lr[4]},
        ],
        lr=5e-4,
        )
        unfreeze_layers=[model.layer1,model.layer2,model.layer3,model.layer4,model.fc]
        for layer in unfreeze_layers:
            for param in layer.parameters():
                param.requires_grad=True
        loss_criterion=select_loss_function(select_loss)
        model,train_losses,valid_losses,y_test,y_predicted=train_model(model,model_name,data_loader,loss_criterion,optimizer,num_epochs)       
        
    # if in training phase, freeze all the layers except the last and update the weights only for the last
    if toggle_train:
        # select a pretrained model and a loss function
        model=select_pretrained_model(select_model)
        loss_criterion=select_loss_function(select_loss)
        set_parameter_requires_grad(model,True)
        # get the number of features from the model    
        num_features=model.fc.in_features
        
        # select a last layer for the model
        #model.fc=nn.Linear(num_features,num_classes)  # SIMPLE LINEAR
        dimension=512
        dropout_percentage=0.5
        print("LAST LAYER'S DIM : ",dimension)
        print("DROPOUT PERC     : ",dropout_percentage)
        print("")
        model.fc = nn.Sequential(nn.Linear(num_features, dimension),nn.ReLU(),nn.Dropout(dropout_percentage),nn.Linear(dimension,num_classes)) # WITH DROPOUT
        
        # load the model to the device
        model.to(device)
        # select an optimizer
        optimizer=select_optim(select_optimizer)
        
        params_to_update = []
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                #print("\t",name)
        # MODEL TRAIN-VALIDATE
        model,train_losses,valid_losses,y_test,y_predicted=train_model(model,model_name,data_loader,loss_criterion,optimizer,num_epochs)
    else:
        set_parameter_requires_grad(model,False)

    if toggle_test:
        # MODEL TEST
        with open("./saved_models/"+model_name+"_test.log", "a") as f:
            model.eval()
            y_test2=[]
            y_predicted2=[]
            accuracy = 0
            data_generator_test ={k: datasets.ImageFolder(os.path.join(root_dir,k),image_transforms[k]) for k in ["test"]}
            data_loader_test= {k :DataLoader(data_generator_test[k],batch_size=bs,shuffle=True,num_workers=0) for k in ["test"]}
            for inputs, labels in data_loader_test['test']:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                # Class with the highest probability is our predicted class
                equality = (labels.data == outputs.max(1)[1])
                # Accuracy is number of correct predictions divided by all predictions
                accuracy += equality.type_as(torch.FloatTensor()).mean()
                #print("The ranking of labels is : ")
                #print(outputs:topk(10))    
                y_test2.append(labels.data.cpu().numpy())
                y_predicted2.append(outputs.max(1)[1].cpu().numpy())
            print("Test accuracy: {:.4f}".format(accuracy/len(data_loader_test['test'])))
            #print(F.softmax(outputs,dim=1).data.squeeze())
            #print("")
            #print(torch.topk(F.softmax(outputs,dim=1).data.squeeze(),1))
            f.write("{}\t{:.4f}\n".format(model_name,accuracy/len(data_loader_test['test'])) )
            # compute and plot the confusion matrix
            computeAndPlotCM(y_test2,y_predicted2,model_name+" TEST",class_names)
   
    if toggle_plot:
        style.use("ggplot")
        create_acc_loss_graph(model_name)
    