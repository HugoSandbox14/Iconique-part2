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
        common, idx_src, idx_labels = np.intersect1d(src[0]['vertno'],verts['lh'],return_indices=True)
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

def make_alpha_matrix(GR,eeg_group,lamb):
    """ 
    la matrice alpha c'est la matrice qui contient toutes les sources les plus representative du signal eeg moyen
    """
    region = GR.shape[1]
    time = eeg_group.data.shape[1]
    alpha = np.zeros((region,time))

    for t, eeg_t in enumerate(eeg_group.data.T):
        alpha[:, t] = np.linalg.pinv(GR.T @ GR + lamb * np.eye(GR.shape[1])) @ GR.T @ eeg_t
    return alpha

# version ancienne

# def make_alpha_matrix(GR,eeg_group,lamb):
#     """ 
#     la matrice alpha c'est la matrice qui contient toutes les sources les plus representative du signal eeg moyen
#     """
#     region = GR.shape[1]                            # (n_channel, n_region)
#     time = eeg_group.data.shape[1]                       # (n_channel, n_time)
#     alpha = np.zeros((region,time))
#     base = GR.T @ GR + lamb * np.eye(GR.shape[1])

#     for t, eeg_t in enumerate(eeg_group.data.T):
#         alpha[:, t] = np.linalg.solve(base,GR.T @ eeg_t)
#     return alpha

# version avec nnls

# def make_alpha_matrix(GR,eeg_group,lamb):
#     
#     region = GR.shape[1]
#     time = eeg_group.data.shape[1]
#     alpha = np.zeros((region,time))

#     for t, eeg_t in enumerate(eeg_group.data.T):
#         alpha[:, t], _ = sp.optimize.nnls(GR, eeg_t)
#     return alpha
######################## Fonctions pour class Simulation_Serielle() ########################

# version nouvelle

def make_alpha_matrix_seriel(GR,eeg_group,lamb,constraint,sfreq = 512, gap = 0.2):

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

    for i, t_c in enumerate(time_constraint):
        GR_sub = GR[:,[i]]
        for t in range(t_c[0], t_c[1]):
            eeg_t = eeg_group.data[:,t]
            alpha[i, t] = np.linalg.pinv(GR_sub.T @ GR_sub + lamb * np.eye(GR_sub.shape[1])) @ GR_sub.T @ eeg_t
        
    return alpha

# version ancienne 

# def make_alpha_matrix_seriel(GR,eeg_group,lamb,constraint,sfreq, gap = 0.2):

#     def make_constraint(constraint,time,gap = gap, sfreq = sfreq):
#         liste_out = []
#         sf_gap = int(gap * sfreq)
#         for i in range(len(constraint)):
#             if i == len(constraint) -1:
#                 liste_out.append((int(constraint[i] * sfreq / 1000) + sf_gap,time))
#             else :
#                 liste_out.append((int(constraint[i] * sfreq / 1000)+ sf_gap,int(constraint[i+1]* sfreq / 1000)+ sf_gap))
#         return liste_out
    
#     region = GR.shape[1]
#     time = eeg_group.data.shape[1]
#     alpha = np.zeros((region,time))
#     time_constraint = make_constraint(constraint,time)

#     for i, t_c in enumerate(time_constraint):
#         GR_sub = GR[:,[i]]
#         base = GR_sub.T @ GR_sub + lamb * np.eye(GR_sub.shape[1])

#         for t in range(t_c[0], t_c[1]):
#             eeg_t = eeg_group.data[:,t]
#             alpha[i, t] = np.linalg.solve(base, GR_sub.T @ eeg_t)[0]
        
#     return alpha

# version avec nnls

# def make_alpha_matrix_seriel(GR,eeg_group,lamb,constraint,sfreq, gap = 0.2):

#     def make_constraint(constraint,time,gap = gap, sfreq = sfreq):
#         liste_out = []
#         sf_gap = int(gap * sfreq)
#         for i in range(len(constraint)):
#             if i == len(constraint) -1:
#                 liste_out.append((int(constraint[i] * sfreq / 1000) + sf_gap,time))
#             else :
#                 liste_out.append((int(constraint[i] * sfreq / 1000)+ sf_gap,int(constraint[i+1]* sfreq / 1000)+ sf_gap))
#         return liste_out
    
#     region = GR.shape[1]
#     time = eeg_group.data.shape[1]
#     alpha = np.zeros((region,time))
#     time_constraint = make_constraint(constraint,time)

#     for i, t_c in enumerate(time_constraint):
#         GR_sub = GR[:,[i]]

#         for t in range(t_c[0], t_c[1]):
#             eeg_t = eeg_group.data[:,t]
#             value, _ = sp.optimize.nnls(GR_sub, eeg_t)
#             alpha[i, t] = value[0]
#     return alpha


######################## Fonctions pour calcul du RMSE ########################

### pour simulation paralelle et serielle

def compute_RMSE(epochs_subject, sim_eeg):

    list_rmse = []
    for epoch in epochs_subject:
        rmse_epoch = np.sqrt(np.mean((epoch - sim_eeg) ** 2))
        list_rmse.append(rmse_epoch)

    return np.array(list_rmse)


def compute_RMSE_time(epochs_subject, sim_eeg):
    return np.sqrt(np.mean((epochs_subject - sim_eeg) ** 2,axis=(0, 1)))


### pour group simulation

def compute_group_RMSE(list_simulations):
    list_rmse = []
    array_mean_rmse = []
    array_std_rmse = []
    array_median_rmse = []
    array_subject = []

    for sim in list_simulations:
        list_rmse.append(sim.array_rmse)
        array_mean_rmse.append(sim.mean_rmse)
        array_std_rmse.append(sim.std_rmse)
        array_median_rmse.append(sim.median_rmse)
        array_subject.append(sim.subject)
    
    return list_rmse ,np.array(array_mean_rmse), np.array(array_std_rmse) ,np.array(array_median_rmse) ,np.array(array_subject) 


######################## Fonctions pour class Group_Simulation ########################

def make_df_rmse(array_mean_rmse, array_std_rmse ,array_median_rmse ,array_subject):
    
    dico = {
        "mean_rmse" : np.array(array_mean_rmse) *1e6,
        "std_rmse" : np.array(array_std_rmse) *1e6,
        "median_rmse" : np.array(array_median_rmse) *1e6,
        "name_subject" : np.array(array_subject),
        }
    
    return pd.DataFrame(dico) 



######################## Fonctions pour class Model_Moyenne() ########################


