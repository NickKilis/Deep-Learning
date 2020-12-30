# Deep Learning Projects - [Python v3.7.6, PyTorch v1.5, CUDA v10.1]:

1.Convolutional Neural Network (CNN) 
  -Classification of images containing spectrograms from audio files
  -10 classes total:
    1 anechoic class
    3 classes with different levels of noise
    3 classes with different levels of reverb
    3 classes with different levels of noise and reverb
  -Data augmentation via delta-Mel spectrograms
  -Transfer learning from the pretrained ResNet50 model
  -Replacement of ResNet50's last layer with a sequence of layers and train only them
  -Systematic search of hyperparameters:
    -Tuning
    -Fine tuning

2.Recurrent Neural Network (LSTM)
  -Binary Classification of 1-dimensional sequences extracted from the history values of four cryptocurrency coins
    Class "0" : falling price expected
    Class "1" : rising  price expected 
  -Data cleaning
  -Class Imbalance removed
  -Data scaling
  -Automated best Learning Rate region extraction
  -Systematic search of hyperparameters:
    -Tuning
    -Fine tuning

3.Deep Reinforcement Learning (DRL)
  -Mountain Car problem of OpenAI Gym library
    -Agent with three available actions
    -Train an agent via rewards, on how he can gain momentum using a sequence of actions, in order to surpass a steep mountain
  -Systematic search of hyperparameters:
    -Hyperparameter Tuning
    -Architecture - Policy tuning
  -Visualization of learning via a gif of Q-tables (every 100 episodes) 
