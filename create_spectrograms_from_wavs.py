# HOW TO USE:
#   1. audio_path : choose a name of the folder, that contains the wav files
#   2. csv_name   : choose a name for the csv file to be created, which is going to contain the file names and labels
#
#   3a. name_of_folder_with_spectrograms  : choose a name of the folder, in which the spectrograms will be saved
#   3b. name_of_folder_with_spectrograms2 : choose a name of the folder, in which the spectrograms of the second dataset will be saved
#
#   4a. output_folder_name  : choose a name for the output folder, which is going to contain the splitted files (train-valid-test)
#   4b. output_folder_name2 : choose a name for the output folder, which is going to contain the splitted files of the second dataset (train-valid-test)
#----------------------------------------------------------------------------#
#----------------------------PRIOR INSTALLATIONS-----------------------------#
## pip install tqdm
#----------------------------------------------------------------------------#
#----------------------------IMPORT LIBRARIES--------------------------------#
import os,librosa,csv
import librosa.display
import numpy as np
import pandas as pd
from scipy.io import wavfile
from tqdm import tqdm
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib
from os import path
from os import walk
import split_folders
#----------------------------------------------------------------------------#
#--------------------------GET FILENAMES FROM WAVS---------------------------#
def get_file_names(audio_path):
    f = []
    for (dirpath, dirnames, filenames) in walk(audio_path):
        f.extend(filenames)
        break
    return(filenames)
#----------------------------------------------------------------------------#
#---------------------CREATE CSV WITH FILENAMES AND LABELS-------------------#
def create_csv_names_labels(f,csv_name):
    if path.exists(csv_name):
        print("")
        print('The '+csv_name+' file already exists!')
        print("")
    else:    
        with open(csv_name, 'w',newline='') as f2:
            filewriter = csv.writer(f2)
            filewriter.writerow(['Name', 'Label'])
            for i in range(0,len(f)):
                if i < 632:
                    filewriter.writerow([f[i], 'anechoic'])
                if i >= 632 and i < 2*632:
                    filewriter.writerow([f[i], 'noise10'])
                if i >= 2*632 and i < 3*632:
                    filewriter.writerow([f[i], 'noise15'])
                if  i >= 3*632 and i < 4*632:
                    filewriter.writerow([f[i], 'noise20'])
                if i >= 4*632 and i < 5*632:
                    filewriter.writerow([f[i], 'rev0.3'])
                if  i >= 5*632 and i < 6*632:
                    filewriter.writerow([f[i], 'rev0.5'])
                if i >= 6*632 and i < 7*632:
                    filewriter.writerow([f[i], 'rev1.3'])
                if  i >= 7*632 and i < 8*632:
                    filewriter.writerow([f[i], 'rev0.3_noise10'])
                if i >= 8*632 and i < 9*632:
                    filewriter.writerow([f[i], 'rev0.5_noise15'])
                if  i >= 9*632:
                    filewriter.writerow([f[i], 'rev1.3_noise20'])
#----------------------------------------------------------------------------#
#----------------------CREATE SPECTROGRAMS FROM WAVS-------------------------#
# y, sr = librosa.core.load(path="wavs2\\rev_noise20_SX396.wav",sr=24000)
# # plt.figure()
# # librosa.display.waveplot(y, sr=sr)
# # plt.title('Saple audio')

# S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40,n_fft=2048, hop_length=1024, win_length=1024, window='blackman')

# plt.figure()
# S_dB = librosa.power_to_db(S, ref=np.max)
# librosa.display.specshow(S_dB, sr=sr)
# plt.show()
# plt.savefig('spec\\filename.jpg',bbox_inches ='tight', pad_inches=-0.05)

# plt.figure()
# S_dB = librosa.power_to_db(S, ref=np.max)
# librosa.display.specshow(S_dB, x_axis='time',y_axis='mel', sr=sr)
# plt.colorbar(format='%+2.0f dB')
# plt.title('Mel-frequency spectrogram')
# plt.tight_layout()
# plt.show()
# plt.savefig('filename.jpg')
#----------------------------------------------------------------------------#
def create_dir(dirname):
    if os.path.exists(dirname):
        pass
    else:
        os.makedirs(dirname)
        
# FIRST DATASET
def save_spectogram(file, output, figsize=(4,4)):
    # make images 400x400
    freq, sound = wavfile.read(file)
    freq, t, spectro = signal.spectrogram(sound)
    spectro = 10*np.log(spectro.astype(np.float32))
    fig = plt.figure(figsize=figsize, frameon=False) 
    ax = plt.Axes(fig, [0.,0.,1.,1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.pcolormesh(t/100, freq, spectro)
    fig.savefig(output, dpi=100)
    plt.close()

# FIRST DATASET
def create_spectrograms(filenames,audio_path,destin_folder,df_names_labels):
    matplotlib.use('Agg') #stop display output in ipython
    if path.exists(destin_folder):
        print("")
        print('The '+destin_folder+' folder already exists!')
        print("")
    else:
        print("")
        print('The wavs are being converted into spectrograms...')
        for item in tqdm(filenames):
            #name=item.split('\\')[-1]
            dirname='./'+destin_folder+'/' + df_names_labels.loc[item].Label
            create_dir(dirname)
            out_file = dirname+ '/' + item.split('.wav')[0] + '.jpg'
            save_spectogram('./'+audio_path+ '/'+item, out_file)    
        print("")
        print('All spectrograms have been made (400x400 pixels) and saved in folder '+destin_folder+' !')    

# MODIFIED DATASET
def save_spec(path_in,path_out,name, sr=24000):
    #matplotlib.use('Agg') #stop display output in ipython
    y, sr = librosa.core.load(path_in,sr)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40,n_fft=2048, hop_length=1024, win_length=1024, window='blackman')
    plt.figure()
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, sr=sr)
    #plt.show()
    plt.savefig(path_out,bbox_inches ='tight', pad_inches=-0.05)
    plt.close()        
    
# MODIFIED DATASET
def create_spectrograms2(filenames,audio_path,destin_folder,df_names_labels):
    matplotlib.use('Agg') #stop display output in ipython
    if path.exists(destin_folder):
        print('The '+destin_folder+' folder already exists!')
    else:
        print("")
        print('The wavs are being converted into spectrograms...')
        for item in tqdm(filenames):
            name=item.split('\\')[-1]
            #print(name)
            #destin_folder="spec_6320_red"
            dirname='./'+destin_folder+'/' + df_names_labels.loc[name].Label
            create_dir(dirname)
            out_file = dirname+ '/' + name.split('.wav')[0] + '.jpg'  
            save_spec('./'+audio_path+ '/'+name,out_file,name.split('.wav')[0]) 
        print("")
        print('All spectrograms have been made (490x364 pixels) and saved in folder '+destin_folder+' !')
        
def split_files_into_train_test_valid(input_folder_name,output_folder_name):
    # Split with a ratio.
    # To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
    split_folders.ratio(input_folder_name, output=output_folder_name, seed=1337, ratio=(.7, .2, .1)) # default values
    
    # Split val/test with a fixed number of items e.g. 100 for each set.
    # To only split into training and validation set, use a single number to `fixed`, i.e., `10`.
    #split_folders.fixed('input_folder', output="output", seed=1337, fixed=(100, 100), oversample=False) # default values
#----------------------------------------------------------------------------#
#--------------FIND FILENAMES-CREATE CSV-CREATE DATAFRAME--------------------#
#name of folder containing all the wav files
audio_path="wavs_6320"
# get filenames from folder with wavs
f_names=get_file_names(audio_path)
# create a csv with filenames and their labels
csv_name="labels.csv"
create_csv_names_labels(f_names,csv_name)
# create a dataframe from the csv file
df_names_labels = pd.read_csv(csv_name, usecols=['Name', 'Label'])
df_names_labels=df_names_labels.set_index('Name')
#----------------------------------------------------------------------------#
#-------CREATE SPECTROGRAMS FOR EACH CLASS AND STORE THEM INTO A FOLDER------#
# FIRST DATASET
name_of_folder_with_spectrograms="spec_6320_green"
create_spectrograms(f_names,audio_path,name_of_folder_with_spectrograms,df_names_labels)
# MODIFIED DATASET
name_of_folder_with_spectrograms2="spec_6320_red"
create_spectrograms2(f_names,audio_path,name_of_folder_with_spectrograms2,df_names_labels)
#----------------------------------------------------------------------------#
#-------------SPLIT SPECTROGRAMS INTO TRAIN TEST VALIDATION------------------#
# FIRST DATASET
output_folder_name="spec_6320_green_train_test_val"
split_files_into_train_test_valid(name_of_folder_with_spectrograms,output_folder_name)
# MODIFIED DATASET
output_folder_name2="spec_6320_red_train_test_val"
split_files_into_train_test_valid(name_of_folder_with_spectrograms2,output_folder_name2)
#----------------------------------------------------------------------------#