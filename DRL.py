#----------------------------------------------------------------------------#
#----------------------------IMPORT LIBRARIES--------------------------------#
import matplotlib.pyplot as plt
import gym
import time
import os
import numpy as np
from tqdm import tqdm, trange
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
#----------------------------------------------------------------------------#
#----------------------------PRIOR INSTALLATIONS-----------------------------#
## pip install gym
## pip install tqdm
#----------------------------------------------------------------------------#
#--------------------------DEFINE FUNCTIONS----------------------------------#
def plot_and_save_figures(history_dictionary,model_name,SHOW_EVERY):
    # POSITIONS
    plt.figure(1, figsize=[10,5])
    p = pd.Series(history_dictionary["position"])
    ma = p.rolling(10).mean()
    plt.plot(p, alpha=0.8,label="Positions")
    plt.plot(ma,label="Rolling mean")
    plt.plot([first_success_index[0]],[history_dictionary["position"][first_success_index[0]]], 'ro',label="First success")
    plt.plot([0,max_episodes],[0.5,0.5],"--g",label="Success threshold")
    plt.xlabel('Episode')
    plt.ylabel('Position')
    plt.title(model_name+'\nCar final position for each episode'+"\nFirst success in episode :"+str(first_success_index[0])+" with value : "+str(history_dictionary["position"][first_success_index[0]]))
    plt.legend(loc=0)
    plt.grid()
    plt.savefig('./experiments_plots/'+model_name+'_1.png')
    plt.show()
         
    # REWARDS (MIN-MAX-AVG)      
    plt.figure(2, figsize=[10,5])
    plt.plot(avg_reward,label="average")
    plt.plot(min_reward,label="min")
    plt.plot(max_reward,label="max")
    if np.argmax(aggr_ep_rewards["min"])*SHOW_EVERY !=0:
        plt.title(model_name+'\nMin,max,average reward history every '+str(SHOW_EVERY)+' steps\nThe earliest best score was found in episode : '+str(np.argmax(aggr_ep_rewards["min"])*SHOW_EVERY)+" ,with value : "+str(max(aggr_ep_rewards["min"])))
        plt.plot([np.argmax(aggr_ep_rewards["min"])*SHOW_EVERY],[max(aggr_ep_rewards["min"])], 'ro')
    else:
        plt.title(model_name+'\nMin,max,average reward history every '+str(SHOW_EVERY)+' steps')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid()
    plt.legend(loc=0)
    plt.savefig('./experiments_plots/'+model_name+'_2.png')
    plt.show()
    
    # REWARDS
    plt.figure(3, figsize=[10,5])
    plt.plot([item[0] for item in history_dictionary["reward"]])
    plt.title(model_name+'\nReward history')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid()
    plt.show()
    plt.savefig('./experiments_plots/'+model_name+'_3.png')
    
    # EPSILON
    plt.figure(4, figsize=[10,5])
    plt.plot([item[0] for item in history_dictionary["epsilon"]])
    plt.title(model_name+'\nEpsilon history')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.grid()
    plt.show()
    plt.savefig('./experiments_plots/'+model_name+'_4.png')
    
    # LOSS
    plt.figure(5, figsize=[10,5])
    plt.plot([item[0] for item in history_dictionary["loss"]])
    plt.title(model_name+'\nLoss history')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.grid()
    plt.show()
    plt.savefig('./experiments_plots/'+model_name+'_5.png')
    
    # WEIGHT
    plt.figure(6, figsize=[10,5])
    plt.plot([item[0] for item in history_dictionary["weight"]])
    plt.title(model_name+'\nWeight history')
    plt.xlabel('Episode')
    plt.ylabel('Weight ')
    plt.grid()
    plt.show()
    plt.savefig('./experiments_plots/'+model_name+'_6.png')
    
    # LEARNING RATE
    plt.figure(7, figsize=[10,5])
    plt.plot([item[1] for item in history_dictionary["lr"]],[item[0] for item in history_dictionary["lr"]])
    plt.title(model_name+'\nLearning rate history'+' - Initial value : '+str(learning_rate)+' - Decay started in episode : '+str(history_dictionary["lr"][0][1]))
    plt.xlabel('Episode')
    plt.ylabel('Learning rate')
    plt.grid()
    plt.show()
    plt.savefig('./experiments_plots/'+model_name+'_7.png')
    
    # MAX POSITIONS
    plt.figure(8, figsize=[10,5])
    plt.plot([item[1] for item in history_dictionary["max_pos"]],[item[0] for item in history_dictionary["max_pos"]],label="Max position")
    plt.plot([min([item[1] for item in history_dictionary["max_pos"]]),max([item[1] for item in history_dictionary["max_pos"]])],[0.5,0.5],"--g",label="Success threshold")
    plt.plot([first_success_index[0]],[history_dictionary["position"][first_success_index[0]]], 'ro',label="First success")
    plt.title(model_name+'\nMax position history')
    plt.xlabel('Episode')
    plt.ylabel('Max position')
    plt.legend(loc=0)
    plt.grid()
    plt.show()
    plt.savefig('./experiments_plots/'+model_name+'_8.png')

def select_loss_function(select_loss):
    if select_loss=='MSE':
        loss_criterion = nn.MSELoss()   
    elif select_loss=='L1Loss':    
        loss_criterion = nn.L1Loss()
   
    else:
        print("Loss function with given name not found!")
    return(loss_criterion)

def select_optim(select_optimizer):
    if select_optimizer=='Adam':
        optimizer=optim.Adam(policy.parameters(),lr=learning_rate)
    elif select_optimizer=='Adagrad':
        optimizer=optim.Adagrad(policy.parameters(),lr=learning_rate)
    elif select_optimizer=='SGD':
        optimizer=optim.SGD(policy.parameters(),lr=learning_rate)
    elif select_optimizer=='SGD_M':
        optimizer=optim.SGD(policy.parameters(),lr=learning_rate,momentum=0.9)
    elif select_optimizer=='SGD_N':
        optimizer=optim.SGD(policy.parameters(),lr=learning_rate,momentum=0.9,dampening =0,nesterov=True)
    elif select_optimizer=='ASGD':
        optimizer=optim.ASGD(policy.parameters(),lr=learning_rate)
    else:
        print("Optimizer with given name not found!")
    return(optimizer)

def create_dir(dirname):
    if os.path.exists(dirname):
        print("The directory "+dirname+" already exists!")
        toggle_train=False
        toggle_plot=False
    else:
        os.makedirs(dirname)
        print("The directory "+dirname+" was created!")
        
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, 0, 1)


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n
        self.hidden = 150
        self.l1 = nn.Linear(self.state_space, self.hidden, bias=False)
        #self.l2 = nn.ReLU()
        self.l3 = nn.Linear(self.hidden, self.action_space, bias=False)
        #self.drop_layer = nn.Dropout(p=0.3)
    def forward(self, x):    
        #model = torch.nn.Sequential(self.l1,self.l2,self.l3)
        #self.l1.weight.data.normal_(0.0, 1)
        #self.l3.weight.data.normal_(0.0, 1)      
        model = torch.nn.Sequential(self.l1,self.l3)
        return model(x)
    
def train_model(max_episodes,state,steps,learning_rate,epsilon,discount_factor,decay,max_position,loss_fn,optimizer,scheduler,toggle_scheduler):
    history_dictionary ={"max_pos":[], "epsilon":[], "lr":[], "success":[], "failure":[], "loss":[], "reward":[], "position":[]}
    successes = 0
    for episode in trange(max_episodes):
        episode_loss = 0
        episode_reward = 0
        state = env.reset()
        for s in range(steps):
            # Uncomment to render environment
            #if episode % 100 == 0 and episode > 0:
            #    env.render()
            
            # Get first action value function
            Q = policy(Variable(torch.from_numpy(state).type(torch.FloatTensor)))
            # Choose epsilon-greedy action
            if np.random.rand(1) < epsilon:
                action = np.random.randint(0,3)
            else:
                _, action = torch.max(Q, -1)
                action = action.item()
            # Step forward and receive next state and reward
            state_1, reward, done, _ = env.step(action)
            # Find max Q for t+1 state
            Q1 = policy(Variable(torch.from_numpy(state_1).type(torch.FloatTensor)))
            maxQ1, _ = torch.max(Q1, -1)
            # Create target Q value for training the policy
            Q_target = Q.clone()
            Q_target = Variable(Q_target.data)
            Q_target[action] = reward + torch.mul(maxQ1.detach(), discount_factor)
            # Calculate loss
            loss = loss_fn(Q, Q_target)
            # Update policy
            policy.zero_grad()
            loss.backward()
            optimizer.step()
            # Record history
            episode_loss += loss.item()
            episode_reward += reward
            # Keep track of max position
            if state_1[0] > max_position:
                max_position = state_1[0]
                history_dictionary["max_pos"].append((max_position,episode))
            if done:
                if state_1[0] >= 0.5:
                    # On successful epsisodes, adjust the following parameters
                    # Adjust epsilon
                    if epsilon>0.01:
                        epsilon *= decay
                    else:
                        epsilon=0.01
                    history_dictionary["epsilon"].append((epsilon,episode))
                    # Adjust learning rate
                    if toggle_scheduler:
                        scheduler.step()
                    history_dictionary["lr"].append((optimizer.param_groups[0]['lr'],episode))
                    # Record successful episode
                    successes += 1
                    history_dictionary["success"].append((successes,episode))
                
                elif state_1[0] < 0.5:
                    history_dictionary["failure"].append(episode)
                # Record history
                history_dictionary["loss"].append((episode_loss,episode))
                history_dictionary["reward"].append((episode_reward,episode))
                weights = np.sum(np.abs(policy.l2.weight.data.numpy()))+np.sum(np.abs(policy.l1.weight.data.numpy()))
                history_dictionary["weight"].append((weights, episode))
                history_dictionary["position"].append(state_1[0])
                break
            else:
                state = state_1
            if episode % 100 == 0:
                np.save(f"qtables/{episode}-qtable.npy", Q_target)
    print('successful episodes: {:d} ,success percentage: {:.4f}%'.format(successes, successes/max_episodes*100))
    return(history_dictionary)
#----------------------------------------------------------------------------#
#----------------------------------------------------------------------------#
if __name__ == "__main__":
#----------------------------------------------------------------------------#
#-------------------------------HYPERPARAMETERS------------------------------#   
    env = gym.make('MountainCar-v0')
    #env.seed(1); torch.manual_seed(1); np.random.seed(1)
    # Parameters
    state = env.reset()
    SHOW_EVERY=100
    max_episodes = 2000
    steps = 1000
    env._max_episode_steps=500
    learning_rate = 0.001
    epsilon = 0.5
    discount_factor = 0.99
    max_position = -0.4
    decay=0.99
    gamma=0.9
    select_loss="MSE"
    select_optimizer="SGD"
    toggle_train=True
    toggle_plot=True
    toggle_scheduler=True
    # Initialize Policy
    policy = Policy()
    loss_fn =select_loss_function(select_loss)
    optimizer =select_optim(select_optimizer)
    create_dir("qtables")
    create_dir("experiments_plots")
    
    model_name = f'''max_episodes-{max_episodes}-loss-{select_loss}-opt{select_optimizer}-df-{str(discount_factor).replace('.', '')}-lr-{str(learning_rate).replace('.', '')}-e-{str(round(epsilon,5)).replace('.', '')}-decay-{str(decay).replace('.', '')}-{int(time.time())}'''
    print("MODEL NAME       : ",model_name) 
    print("LEARNING RATE    : ",learning_rate)
    print("LOSS FUNCTION    : ",select_loss)
    print("OPTIMIZER        : ",select_optimizer)
    print("MAX EPISODES     : ",max_episodes)
    print("TOGGLE TRAIN     : ",toggle_train)
    print("TOGGLE PLOT      : ",toggle_plot)
    print("TOGGLE LOAD      : ",toggle_scheduler)
    print("")
  
    if toggle_train:
        if toggle_scheduler:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)
        else:
            scheduler=None
        print("Training in progress...")
        since = time.time()
        history_dictionary=train_model(max_episodes,state,steps,learning_rate,epsilon,discount_factor,decay,max_position,loss_fn,optimizer,scheduler,toggle_scheduler)
        time_elapsed = time.time() - since
        print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print("")
    
    print('failures: {:d} ,failure percentage: {:.4f}%'.format(len(history_dictionary["failure"]), len(history_dictionary["failure"])/max_episodes*100))
    first_success_index = [i for i,x in enumerate(history_dictionary["position"]) if x >= 0.5]
    print('first successful episode :',first_success_index[0])
    
    avg_reward=[]
    min_reward=[]
    max_reward=[]
    history_rewards_split = np.array_split(history_dictionary["reward"], len(history_dictionary["reward"])//SHOW_EVERY)
    for array in history_rewards_split:
       avg_reward.append(sum([item[0] for item in array])/len([item[0] for item in array]))
       min_reward.append(min([item[0] for item in array]))
       max_reward.append(max([item[0] for item in array]))
       
    aggr_ep_rewards ={"ep":[x for x in range(0,max_episodes,SHOW_EVERY)], "avg":avg_reward, "min":min_reward, "max":max_reward}

    if toggle_plot:
        plot_and_save_figures(history_dictionary,model_name,SHOW_EVERY)
