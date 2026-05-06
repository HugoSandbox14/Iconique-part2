import pandas as pd
from Import_data import import_df_description, Make_path_list
from Parameters import subject_to_pop, path_directory, dict_category, path_excel, std_bad_epoch, fmin, fmax
from Tools import  find_participant,get_only_eeg, df_to_mne_events2, get_sub_df, get_tmax, filtering
import mne
from scipy.stats import pearsonr
from autoreject import AutoReject


def make_epochs(path_data,df_sub,tmax):
    
    """ 
    df_sub = le sous Dataframe contenant uniquement les meta data du sujet en question (moment de la stimuation, temps de production...)
    tmax = temps maximum pour le decoupage des essais en epoch (moyenne des temps de latence sur les essais valides)
    """

    raw = mne.io.read_raw_bdf(path_data, preload=True)          # lecture du fichier .bdf

    raw_eeg = get_only_eeg(raw)                                 # selection des canaux eeg uniquement
    raw_eeg.set_montage("biosemi128")                           # precision du montage utilise
    filtered_eeg = filtering(raw_eeg,fmin,fmax)

    mne_events, stim = df_to_mne_events2(filtered_eeg,df_sub)   # obligatoire pour faire un objet mne.Epochs(), contient les information de debut de stiumulation et de temp de latence   
    epochs_eeg = mne.Epochs(filtered_eeg,events = mne_events,
                            event_id = stim,tmin = -0.2,
                            tmax=tmax,preload=True)             # on utilise le temps de latence moyen (sans les essais trop lents) pour mettre une limite a la taille de la fenetre 
    
    epochs_eeg.set_eeg_reference('average', projection=False)   # on rereference la baseline par rapport a la moyenne
    epochs_eeg.apply_baseline((None, 0))                        # on retire le bruit de la baseline

    # ar = AutoReject()
    # epochs_eeg_clean = ar.fit_transform(epochs_eeg)           # retire automatiquement les epochs trop bruitees et les canaux mauvais
    # log = ar.get_reject_log(epochs_eeg)                       # resuperation des informations sur le nettoyage
    # bad_epochs_auto = log.bad_epochs7                         # epochs rejetés
    # bad_channels_auto = log.labels                            # quels canaux étaient mauvais

    # # ajout des infos aux meta donnees 
    # # df_sub["n_bad_ch"] = (bad_channels_auto == 1).sum(axis=1)   # nb de canaux interpoles
    # # df_sub["rejected"] = bad_channels_auto                      # nb d'epoch supprimee car trop de canaux bruites

    # epochs_eeg_clean.metadata = df_sub

    return epochs_eeg


def make_dict_epochs(df,list_path_data,std_bad_epoch = std_bad_epoch):
                                                   
    tmax = get_tmax(df,std_bad_epoch)                           # utile pour definir la taille des epochs base sur le temps de latence moyen des essais valides

    dict_epochs = {}
    for path_data in list_path_data:
        sub = find_participant(path_data)
        sub_df = get_sub_df(df,sub)
        dict_epochs[sub] = make_epochs(path_data,sub_df,tmax = tmax)

    return dict_epochs

def make_LOO_one_sub(dict_epochs,subject_out):

    group_epochs = []
    for sub in dict_epochs.keys():
        if sub == subject_out:
            epochs_sub = dict_epochs[sub]
        else :
            part_of_group_epochs = dict_epochs[sub]
            group_epochs.append(part_of_group_epochs)
    
    all_epochs = mne.concatenate_epochs(group_epochs)
    average_group = all_epochs.average()
    return epochs_sub, average_group


def make_fif_all_sub(path_directory = path_directory,subject_to_pop = subject_to_pop):

    list_path_data, path_stim, path_answers = Make_path_list(path_directory,subject_to_pop)
    df = import_df_description(path_stim,path_answers,dict_category,path_excel,subject_to_pop)
    
    dict_epoch = make_dict_epochs(df,list_path_data)

    list_sub = []
    for sub_path in list_path_data:
        list_sub.append(find_participant(sub_path))

    for sub in list_sub:
    
        epochs_sub , average_group = make_LOO_one_sub(dict_epoch,subject_out = sub)
        
        average_group.save(f"group_average_{sub}.fif",overwrite=True)
        epochs_sub.save(f"subject_epochs_{sub}.fif",overwrite=True)
        

make_fif_all_sub()
