import numpy as np
import mne

########################################## Import_Data.py ############################################

def txt_to_table(txt,colones):
        splt = txt.split()
        
        dico = {}
        for c in colones:
            dico[c] = []
    
        for el in splt:
            data = el.split(',')
            for i,c in enumerate(colones):
                dico[c].append(float(data[i]))
        return dico

def take_off_bad_stim(df_stims):
    good_id = []
    for i in list(df_stims.index) :
        if i not in list(df_stims[df_stims["id_stim"] >9].index):
            good_id.append(i)
    
    good_index = np.arange(0,len(good_id))
    good_df = df_stims.iloc[good_id]
    good_df.index = good_index
    return good_df

def pop_list(liste,index):
    for i in sorted(index,reverse = True):
        liste.pop(i-1)
    return liste

######################################################################################



########################################## Raw_to_BIDS ############################################


def find_participant(path,signe = 'S') :
    for i, lettre in enumerate(path):
        if lettre == signe and str.isnumeric(path[i+1]):
            return path[i:i+3]
        
######################################################################################





########################################## Raw_To_Epoch ############################################

def select_max(modality,dict_category,time_window_max):

    for key,value in dict_category.items():

        if modality == key:
            return value[1]
    return time_window_max
        
def df_to_mne_events(df_event):
        
    sub_df_event = df_event[["time_stim","???_stim","id_stim"]]
    array = np.array(sub_df_event).T

    stim = {}
    for el in list(df_event["id_stim"].unique()):
        
        if el == 1:
            name = "A"
        elif el == 2:
            name = "B"
        elif el == 3:
            name = "C"
        elif el == 4:
            name = "D"
        
        stim[name] = el
    
    return array.astype(float).T, stim


def df_to_mne_events2(raw, df_event):

    stim = []
    for el in list(df_event["id_stim"]):
        
        if el == 1:
            stim.append("A") 
        elif el == 2:
            stim.append("B")
        elif el == 3:
            stim.append("C")
        elif el == 4:
            stim.append("B")
        
    raw.set_annotations(mne.Annotations(
        onset= list(df_event["time_stim"]),  # Temps en secondes
        duration= np.zeros(len(df_event)),      # Durée nulle (événement ponctuel)
        description= stim
    ))

    events, event_id = mne.events_from_annotations(raw)
    return  events, event_id







def get_only_eeg(epochs):
    non_eeg = []
    for ch_name in epochs.ch_names:
        if ch_name[0] not in ["A","B","C","D"]:
            non_eeg.append(ch_name)
    
    return epochs.copy().drop_channels(non_eeg)

def filtering(eeg_data,l_freq,h_freq):
    eeg_data.notch_filter(freqs=50, method='fir')
    eeg_data.filter(l_freq=l_freq, h_freq=h_freq, method='fir', phase='zero')

    return eeg_data




######################################################################################

########################################## Epoch_To_PSD ############################################


def is_between(x,tupl):
    if x >= tupl[0] and x <= tupl[1]:
        return True
    else : 
        return False
    



####################################### Descriptive analyse ######################################

def clean_df_time_window(df_time_window):
    
    liste_epoch = list(df_time_window["Unnamed: 0"])
    liste_label_epoch = []
    for el in liste_epoch:
        liste_label_epoch.append(f"Epoch_{el}")
    df_time_window["Epochs"] = liste_label_epoch
    df_time_window = df_time_window.drop(["Unnamed: 0","???_stim","???_reponse"], axis = 1)
    return df_time_window


def clean_df_participant(df_participants): 
    df_participants["Good_answers_rate"] = df_participants["good_answers"] / (df_participants["good_answers"] + df_participants["bad_answers"])
    return df_participants.drop(["Unnamed: 0","subjects_check","good_answers","bad_answers"],axis = 1)





######################################### Else ###################################################

def get_dict_shape(dico):
    if type(dico) == dict:
        key = list(dico.keys())[0]
        return f"{len(dico)} , " + get_dict_shape(dico[key])  

    else :
        return ""