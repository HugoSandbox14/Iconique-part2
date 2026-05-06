import mne
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os
from Parameters import path_directory, subject_to_pop
from Import_data import import_df_description, Make_path_list
from Tools import  find_participant
import seaborn as sns
import pandas as pd
import scipy as sp






######################## Fonctions pour class Requirement_Simulation() ########################


def make_dict_function(labels):

    temp_regions = []
    temp_regions_supp = []
    occ_regions = []
    broca_region = []
    region_motrice = []
    region_bruit = []


    for l in labels:
        if 'occ' in l.name or 'fusi' in l.name or 'pericalca' in l.name or 'cuneus-rh' == l.name or 'cuneus-lh' == l.name:
            occ_regions.append(l)

    for l in labels:
        if 'temp' in l.name:
            temp_regions.append(l)

    for l in labels:
        if 'marginal' in l.name:
            temp_regions_supp.append(l)
    
    for l in labels :
        if 'opercularis' in l.name or 'precentral' in l.name or 'triangularis' in l.name : #or 'pars' in l.name:
            broca_region.append(l)

    for l in labels :
        if 'postcentral' in l.name or 'insula' in l.name or 'precentral' in l.name:
            region_motrice.append(l)

    for l in labels :
        if 'frontal' in l.name or 'Pars' in l.name:
            region_bruit.append(l)

    Visuel_conceptuel = occ_regions
    lexico_seman = temp_regions[:4]
    lexico_phono = temp_regions[4:]
    lexico_phono2 = temp_regions_supp                     
    lexico_phono.extend(lexico_phono2)
    phono = broca_region
    articulation = region_motrice
    bruit = region_bruit[:4]
    bruit2 = region_bruit[6:]
    bruit.extend(bruit2)

    dico_fonction = {
        'visuo_conceptuel' : Visuel_conceptuel,
        'lexico_semantique' : lexico_seman,
        'lexico_phonologique' : lexico_phono,
        'phonologique' : phono,
        'articulation' : articulation,
        'bruit' : bruit,
    }

    return dico_fonction

def make_dict_vertices(dict_function):
    dico_vertices = {}
    for key, value in dict_function.items():
        dico_vertices[key] = {
            "lh" : [],
            "rh" : [],
        }
        for sous_region in value :
            if sous_region.hemi == 'lh' :
                dico_vertices[key]["lh"].append(sous_region.vertices)
            elif sous_region.hemi == 'rh' :
                dico_vertices[key]["rh"].append(sous_region.vertices)

        dico_vertices[key]["lh"] = np.concatenate(dico_vertices[key]["lh"])
        dico_vertices[key]["rh"] = np.concatenate(dico_vertices[key]["rh"])
    
    return dico_vertices

def make_R_matrix(dict_vertices,src):
    """ 
    permet de creer la matrice qui stoque l'information, quel vertice fait parti de quel region
    """

    n_vertices_lh = src[0]['nuse'] 
    n_vertices_rh = src[1]['nuse']
    n_regions = len(dict_vertices)

    R_matrix = np.zeros((n_vertices_lh + n_vertices_rh, n_regions))

    for i, verts in enumerate(dict_vertices.values()):

        # gauche
        common, idx_src, idx_labels = np.intersect1d(src[0]['vertno'],verts['lh'],return_indices=True)

        R_matrix[idx_src, i] = 1.0 / len(idx_src)

        # droite
        common, idx_src, idx_labels = np.intersect1d(src[0]['vertno'],verts['rh'],return_indices=True)
        R_matrix[n_vertices_lh + idx_src, i] = 1.0 / len(idx_src)

    """ 
    src[0]['vertno'] =>                             tous les indices des vertices utilises dans mon source space (depend de la resolution, ico6, ico5...)
    verts['lh'] =>                                  tous les vertices compris dans chaque region defini par la parcellation personnalisee
    quand on fait 'np.intersect1d(src,vertno)' =>   on recupere les indices de vertices compris a la fois dans la region d'interet et dans la resolution choisie
    common =>                                       sont les valeurs (indice des vertices sur l'atlas) (pas necessaire)
    idx_src =>                                      sont les position des valeurs dans vertno
    idx_labels =>                                   sont les position des valeurs dans labels (pas necessaire)
    1.0 / len(idx_src) =>                           pas seulement '1' sinon les regions plus grandes montrerons une activite plus forte artificiellement
    """

    return R_matrix


def make_bem(subject,subjects_dir) :
    model = mne.make_bem_model(subject = subject, subjects_dir = subjects_dir)
    return mne.make_bem_solution(model)

def make_info():
    montage = mne.channels.make_standard_montage("biosemi128")
    info = mne.create_info(
        ch_names= montage.ch_names,
        sfreq=512,
        ch_types="eeg"
    )
    info.set_montage(montage)
    return info

def make_fwd(info,src,bem):
    forward = mne.make_forward_solution(info,trans="fsaverage",src=src,bem=bem,eeg=True)
    forward_fixed = mne.convert_forward_solution(forward,surf_ori=True,force_fixed=True)
    return forward_fixed


######################## Fonctions pour class Simulation_Paralelle() ########################

# version nouvelle

def make_alpha_matrix(GR,eeg_group,lamb,methode = "pinv"):
    """ 
    la matrice alpha c'est la matrice qui contient toutes les sources les plus representative du signal eeg moyen
    """
    region = GR.shape[1]
    time = eeg_group.data.shape[1]
    alpha = np.zeros((region,time))

    if methode == "pinv":
        for t, eeg_t in enumerate(eeg_group.data.T):
            alpha[:, t] = np.linalg.pinv(GR.T @ GR + lamb * np.eye(GR.shape[1])) @ GR.T @ eeg_t
        return alpha
    
    elif methode == "nnls":
        for t, eeg_t in enumerate(eeg_group.data.T):
            alpha[:, t], _ = sp.optimize.nnls(GR, eeg_t)
        return alpha

######################## Fonctions pour class Simulation_Serielle() ########################

# version nouvelle

def make_alpha_matrix_seriel(GR,eeg_group,lamb,constraint,sfreq = 512, gap = 0.2,methode = "pinv"):

    def make_constraint(constraint,time,gap = gap, sfreq = sfreq):
        liste_out = []
        sf_gap = int(gap * sfreq)
        for i in range(len(constraint)):
            if i == len(constraint) -1:
                liste_out.append((int(constraint[i] * sfreq / 1000) + sf_gap,time))
            else :
                liste_out.append((int(constraint[i] * sfreq / 1000)+ sf_gap,int(constraint[i+1]* sfreq / 1000)+ sf_gap))
        return liste_out
    
    region = GR.shape[1]
    time = eeg_group.data.shape[1]
    alpha = np.zeros((region,time))
    time_constraint = make_constraint(constraint,time)

    if methode == "pinv" :
        for i, t_c in enumerate(time_constraint):
            GR_sub = GR[:,[i]]
            for t in range(t_c[0], t_c[1]):
                eeg_t = eeg_group.data[:,t]
                alpha[i, t] = np.linalg.pinv(GR_sub.T @ GR_sub + lamb * np.eye(GR_sub.shape[1])) @ GR_sub.T @ eeg_t
        return alpha
    
    elif methode == "nnls":
        for i, t_c in enumerate(time_constraint):
            GR_sub = GR[:,[i]]

            for t in range(t_c[0], t_c[1]):
                eeg_t = eeg_group.data[:,t]
                value, _ = sp.optimize.nnls(GR_sub, eeg_t)
                alpha[i, t] = value[0]
        return alpha


######################## Fonctions pour calcul du RMSE ########################

### pour simulation paralelle et serielle

def compute_RMSE_average(epochs,sim):
    average = epochs.average().get_data()
    list_err = []
    for epoch_t, sim_t in zip(average,sim):
        list_err.append(compute_err_time(epoch_t,sim_t))
    
    array_err  = np.array(list_err)
    mean_err = np.mean(array_err)

    return np.sqrt(mean_err)

def compute_err_time(epoch_t,sim_t):
    err_t =  epoch_t - sim_t
    return err_t ** 2

def compute_RMSE_signal(epoch, sim):
    list_err = []
    for epoch_t, sim_t in zip(epoch,sim):
        list_err.append(compute_err_time(epoch_t,sim_t))
    
    array_err  = np.array(list_err)
    mean_err = np.mean(array_err)

    return np.sqrt(mean_err)
    
def compute_RMSE_epochs(epochs,sim):
    list_RMSE = []
    for epoch in epochs:
        list_RMSE.append(compute_RMSE_signal(epoch, sim))
    array_RMSE = np.array(list_RMSE)

    return array_RMSE


def compute_RMSE_time(epochs_subject, sim_eeg):
    return np.sqrt(np.mean((epochs_subject - sim_eeg) ** 2,axis=(0, 1)))


### pour group simulation

def compute_group_RMSE(list_simulations, from_average = False):
    list_rmse = []
    array_mean_rmse = []
    array_std_rmse = []
    array_median_rmse = []
    array_subject = []

    for sim in list_simulations:
        if from_average :
            mean = sim.array_rmse_from_average
            std = sim.std_rmse_from_average
            median = sim.median_rmse_from_average
            rmse = sim.array_rmse_from_average
        else :
            mean = sim.array_rmse
            std = sim.std_rmse
            median = sim.median_rmse
            rmse = sim.array_rmse

   
        list_rmse.append(rmse)
        array_mean_rmse.append(mean)
        array_std_rmse.append(std)
        array_median_rmse.append(median)
        array_subject.append(sim.subject)
    
    return list_rmse ,np.array(array_mean_rmse), np.array(array_std_rmse) ,np.array(array_median_rmse) ,np.array(array_subject) 


######################## Fonctions pour calcul du corr ########################

def compute_corr_signal(epoch,sim):
    r = np.corrcoef(epoch, sim)
    return r[0][1]

def compute_corr_epochs(epochs,sim):
    list_corr = []
    for epoch in epochs:
        list_corr.append(compute_corr_signal(epoch,sim))
    return np.array(list_corr)

def compute_corr_average(epochs,sim):
    average = epochs.average().get_data()
    r = np.corrcoef(average, sim)
    return r[0][1]

def compute_group_corr(list_simulations, from_average = False):
    list_corr = []
    array_mean_corr = []
    array_std_corr = []
    array_median_corr = []
    array_subject = []

    for sim in list_simulations:
        if from_average :
            mean = sim.array_corr_from_average
            std = sim.std_corr_from_average
            median = sim.median_corr_from_average
            corr = sim.array_corr_from_average
        else :
            mean = sim.array_corr
            std = sim.std_corr
            median = sim.median_corr
            corr = sim.array_corr

        list_corr.append(corr)
        array_mean_corr.append(mean)
        array_std_corr.append(std)
        array_median_corr.append(median)
        array_subject.append(sim.subject)
    
    return list_corr,np.array(array_mean_corr), np.array(array_std_corr) ,np.array(array_median_corr) ,np.array(array_subject) 

######################## Fonctions pour class Group_Simulation ########################

def make_df_rmse(array_mean_rmse, array_std_rmse ,array_median_rmse ,array_subject):
    
    dico = {
        "mean_rmse" : np.array(array_mean_rmse) *1e6,
        "std_rmse" : np.array(array_std_rmse) *1e6,
        "median_rmse" : np.array(array_median_rmse) *1e6,
        "name_subject" : np.array(array_subject),
        }
    
    return pd.DataFrame(dico) 

def make_df_corr(array_mean_rmse, array_std_rmse ,array_median_rmse ,array_subject):
    
    dico = {
        "mean_corr" : np.array(array_mean_rmse),
        "std_corr" : np.array(array_std_rmse),
        "median_corr" : np.array(array_median_rmse),
        "name_subject" : np.array(array_subject),
        }
    
    return pd.DataFrame(dico) 



######################## Fonctions pour class Model_Moyenne() ########################


