import pandas as pd
from Import_data import import_df_description, Make_path_list
from Parameters import subject_to_pop, path_directory, dict_category_all, path_excel, th_bad_channel, std_bad_epoch
from Tools import  find_participant,get_only_eeg, df_to_mne_events2
import mne
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr




def find_bad_epoch(df,threshold):
    df_bad_epoch = df[df['latency'] > threshold]
    bad_epoch = list(df_bad_epoch.index)
    return bad_epoch


def find_bad_channels(evoked, threshold):

    def find_pic(channel,threshold = threshold) :
        for el in channel:
            if el > threshold or el < -threshold :
                return True
        return False
    
    data = evoked.get_data()
    ch_names = evoked.ch_names
    
    bad_channels = []
    for i ,ch in enumerate(data):
        if find_pic(ch, threshold * 1e-6) :
            print(f"channel trop puissant {ch_names[i]}")
            bad_channels.append(ch_names[i])

    return bad_channels

def compute_intra_rmse(epochs):
    data = epochs.get_data()
    mean = np.mean(data, axis=0)
    
    rmse_epochs = np.sqrt(np.mean((data - mean) ** 2, axis=(1, 2)))
    
    return np.var(rmse_epochs)

def make_evoked(path_data,df,tmax, threshold_epoch = std_bad_epoch,threshold_channel = th_bad_channel):
    
    """
    threshold_channel = si un canal a au moins une valeur qui depasse ce seuil, il est considere comme mauvais (a exprimer en micro Volt --> threshold = threshold x 10e-6 V)
    tmax = temps d'une epoch
    threshold_epoch = combien de fois on ajoute l'ecart type a la moyenne pour definir la taille des epochs (mean + threshold_epoch * std)
    """

    raw = mne.io.read_raw_bdf(path_data, preload=True)
    subject = find_participant(path_data,signe = 'S')

    meta_data = df[df['subjects'] == subject]
    meta_data.index = np.arange(0,len(meta_data))

    raw_eeg = get_only_eeg(raw)
    raw_eeg.set_montage("biosemi128")

    mne_events, stim = df_to_mne_events2(raw_eeg,meta_data)
    epochs_eeg = mne.Epochs(raw_eeg,events = mne_events,event_id = stim,tmin = -0.2,tmax=tmax,preload=True) # On utilise le temps de latence moyen (sans les essais trop lents) pour mettre une limite a la taille de la fenetre 
    
    epochs_eeg.set_eeg_reference('average', projection=False)   # on rereference la baseline par rapport a la moyenne
    epochs_eeg.apply_baseline((None, 0))                        # on retire le bruit de la baseline

    mean = df['latency'].mean()
    std = df['latency'].std()
    th = mean + threshold_epoch * std
    bad_epochs = find_bad_epoch(meta_data,th)

    epochs_eeg_clean =  epochs_eeg.copy()
    epochs_eeg_clean = epochs_eeg_clean.drop(bad_epochs)

    intra_RMSE = compute_intra_rmse(epochs_eeg_clean)

    average = epochs_eeg_clean.average()

    bad_ch = find_bad_channels(average, threshold_channel)      # seuil etablit a (+ ou -) 75 micro volt
    average.info['bads'] = bad_ch
    average_clean = average.copy()
    average_clean.interpolate_bads()                            # on interpole les cannaux trop bruites

    good_answer_rate = list(meta_data['Good_answers_rate'].unique())
    mean_latency = list(meta_data['mean_latency'].unique())
    return average_clean, mean_latency[0], good_answer_rate[0], bad_ch, intra_RMSE

def make_evoked_split(path_directory,subject_to_pop,subject_split):
    path_data, path_stim, path_answers = Make_path_list(path_directory,subject_to_pop)
    df = import_df_description(path_stim,path_answers,dict_category_all,path_excel,subject_to_pop)
    
    mean = df['latency'].mean()
    std = df['latency'].std()
    th_bad_epoch = mean + 2.5 * std                 # critere de selection pour definir ce qu'est un 'bad epoch' sur la base du temps de latence entre stimulation et reponse
 
    df_clean = df[df['latency'] < th_bad_epoch]
    tmax = df_clean['latency'].mean()               # utile pour etablir la fenetre de comparaison entre le groupe et l'individu

    group = []
    individual_average = None
    for path in path_data:
        if subject_split == find_participant(path):
            individual_average, mean_latency, good_answer_rate, bad_ch, intra_RMSE = make_evoked(path,df,tmax = tmax,threshold_epoch=th_bad_epoch)
        else :
            part_of_group, useless_1, useless_2, useless_3, useless_4 = make_evoked(path,df,tmax = tmax, threshold_epoch=th_bad_epoch)
            group.append(part_of_group)
    
    group_average = mne.grand_average(group)
    return group_average, individual_average, mean_latency, good_answer_rate, bad_ch, intra_RMSE



path_data, path_stim, path_answers = Make_path_list(path_directory,subject_to_pop)

list_subject = []
for sub_path in path_data:
    list_subject.append(find_participant(sub_path,signe = 'S'))


sub_list = []
rmse_intra_list = []
mean_latency_list = []
good_answer_rate_list = []
bad_ch_list = []

for sub in list_subject:
    
    group_average , subject_average, mean_latency, good_answer_rate, bad_ch, intra_RMSE = make_evoked_split(path_directory=path_directory,subject_to_pop=subject_to_pop,subject_split= sub)
    
    group_average.save(f"group_average_{sub}.fif",overwrite=True)
    subject_average.save(f"subject_average_{sub}.fif",overwrite=True)
    
    sub_list.append(sub)
    rmse_intra_list.append(intra_RMSE)
    mean_latency_list.append(mean_latency)
    good_answer_rate_list.append(good_answer_rate)
    # bad_ch_list.append(bad_ch)


dico = {
    "sub" : np.array(sub_list),
    "rmse_inra" : np.array(rmse_intra_list) * 10e6,
    "mean_latency" :  np.array(mean_latency_list),
    "good_answer_rate" : np.array(good_answer_rate_list),
    # "bad_ch" : np.array(bad_ch_list),
}

for el in dico.values():
    print(len(el))

df = pd.DataFrame(dico)
df.to_csv("meta_data_subjects.csv")




