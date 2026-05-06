import mne
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os
from Tools_simulation import make_dict_function, make_dict_vertices, make_R_matrix, make_bem, make_info, make_fwd, make_alpha_matrix, make_alpha_matrix_seriel, compute_RMSE, compute_group_RMSE, make_df_rmse, compute_RMSE_time
import seaborn as sns
import pandas as pd
from Parameters import sfreq, parcellation


# Mise en place du dossier FreeSurfer avec toutes les informations relative model de tete average

fs_dir = mne.datasets.fetch_fsaverage(verbose=True)
mne.datasets.fetch_aparc_sub_parcellation(subjects_dir=fs_dir)

subjects_dir = Path(fs_dir).parent
subject='fsaverage'

class Requirement_Simulation():
    
    def __init__(self,subjects_dir = subjects_dir ,subject = subject):
        self.labels = mne.read_labels_from_annot(subject=subject,parc=parcellation,subjects_dir=subjects_dir)           # nom des regions et decoupage utilise dans le model aparc (Desikan Killliany)
        self.dict_vertices = make_dict_vertices(make_dict_function(self.labels))                                        # structuration de ses regions avec fonction cognitive pour les utiliser plus facilement 
        self.src = mne.setup_source_space(subject=subject,spacing='ico5',subjects_dir=subjects_dir,add_dist=False)      # espace des sources
        self.R = make_R_matrix(self.dict_vertices,self.src)                                                             # Matrice qui met en lien les vertex et les regions (les vertex qui definissent une region sont regroupe ensemble comme etant un seul objet)
        self.bem = make_bem(subject,subjects_dir)                                                                       # Model BEM ????
        # self.info = make_info()                                                                                         # info de type mne.info avec nom des canaux, type de montage...
        self.forward = make_fwd(make_info(),self.src,self.bem)                                                          # solution au probleme directe
        self.G = self.forward["sol"]["data"]                                                                            # matrice de gain (permet de passer d'un signal EEG a une estimation des sources et inversement)
        self.G_mean_reference = self.G - np.mean(self.G, axis=0, keepdims=True)                                         # matrice de gain reference a la moyenne

class True_EEG():
    
    def __init__(self,epochs_individual, average_group, subject):
        self.epochs_individual = epochs_individual                                                                     # raw du fichier .fif contenant l'objet Epochs du sujet (plusieur epoch d'un sujet non moyene)
        self.average_group = average_group                                                                              # raw du fichier .fif contenant l'objet Evoked du sujet (plusieur epoch de plusieur sujet moyene)                                   
        self.subject = subject                                                                                          # numero du sujet

class Model_Moyenne():
    def __init__(self,true_eeg):
        self.epochs_individual = true_eeg.epochs_individual                                                                     # raw du fichier .fif contenant l'objet Epochs du sujet (plusieur epoch d'un sujet non moyene)
        self.average_group = true_eeg.average_group                                                                              # raw du fichier .fif contenant l'objet Evoked du sujet (plusieur epoch de plusieur sujet moyene)                                   
        self.subject = true_eeg.subject
        self.array_rmse = compute_RMSE(self.epochs_individual,self.average_group.get_data())

        self.mean_rmse = np.mean(self.array_rmse)
        self.std_rmse = np.std(self.array_rmse)
        self.median_rmse = np.median(self.array_rmse)


class Simulation_Parallele():
    
    def __init__(self,Requirement_Simulation,True_EEG,Lambda):
        
        self.G = Requirement_Simulation.G_mean_reference
        self.G_mean_reference  = Requirement_Simulation.G_mean_reference                                                                # matrice de gain (n_channel, n_vertex)
        self.R = Requirement_Simulation.R                                                                               # matrice (n_vertex, n_region)
        
        self.Average_group = True_EEG.average_group
        self.Epochs_subject = True_EEG.epochs_individual
        self.subject = True_EEG.subject

        self.GR = self.G @ self.R                                                                                       # matrice de gain par rapport aux regions (n_channel, n_region)
        self.Alpha_matrix = make_alpha_matrix(self.GR,self.Average_group,Lambda)                                        # matrice qui contient l'information sur l'activite des regions dans le temps (n_region, n_time)
        self.sim_eeg = self.GR @ self.Alpha_matrix

        self.raw_sim_eeg = mne.io.RawArray(self.sim_eeg,self.Average_group.info)                                        # simulation d'un Raw eeg
        self.evoked_sim = mne.EvokedArray(self.sim_eeg, self.Average_group.info, tmin=-0.2)                             # pour pouvoir plot dse la meme maniere que les evoked (avec couleur...)

        self.array_rmse = compute_RMSE(self.Epochs_subject,self.sim_eeg)
        self.array_rmse_time = compute_RMSE_time(self.Epochs_subject,self.sim_eeg)

        self.mean_rmse = np.mean(self.array_rmse)
        self.std_rmse = np.std(self.array_rmse)
        self.median_rmse = np.median(self.array_rmse)

    def plot_simulation(self,eeg = None,title = "simulation du model paralelle"):
        self.evoked_sim.plot(titles=title)
        if eeg is not None:
            self.Epochs_subject[eeg].average().plot()
        

    
    # def get_info(self):
    #     print(f"shape de average group = {self.Average_group.get_data().shape}")
    #     print(f"shape de epoch subject = {self.Epoch_subject.get_data().shape}")
    #     print(f"shape de simulation raw = {self.raw_sim_eeg.get_data().shape}")
    #     print(f"shape de simulation evoked = {self.evoked_sim.get_data().shape}")



class Simulation_Serielle():
    
    def __init__(self,Requirement_Simulation,True_EEG,Lambda, constraint = [0,200,275,350,600,900]):
        
        self.G = Requirement_Simulation.G_mean_reference
        self.G_mean_reference = Requirement_Simulation.G_mean_reference                                                                # matrice de gain (n_channel, n_vertex)
        self.R = Requirement_Simulation.R                                                                               # matrice (n_vertex, n_region)
        
        self.Average_group = True_EEG.average_group
        self.Epochs_subject = True_EEG.epochs_individual
        self.subject = True_EEG.subject

        self.GR = self.G @ self.R                                                                                       # matrice de gain par rapport aux regions (n_channel, n_region)
        self.Alpha_matrix = make_alpha_matrix_seriel(self.GR,self.Average_group,Lambda,constraint,sfreq=sfreq)          # matrice qui contient l'information sur l'activite des regions dans le temps (n_region, n_time)
        self.sim_eeg = self.GR @ self.Alpha_matrix

        self.raw_sim_eeg = mne.io.RawArray(self.sim_eeg,self.Average_group.info)                                        # simulation d'un Raw eeg
        self.evoked_sim = mne.EvokedArray(self.sim_eeg, self.Average_group.info, tmin=-0.2)                             # pour pouvoir plot dse la meme maniere que les evoked (avec couleur...)

        self.array_rmse = compute_RMSE(self.Epochs_subject,self.sim_eeg)
        self.array_rmse_time = compute_RMSE_time(self.Epochs_subject,self.sim_eeg)

        self.mean_rmse = np.mean(self.array_rmse)
        self.std_rmse = np.std(self.array_rmse)
        self.median_rmse = np.median(self.array_rmse)

    def plot_simulation(self,eeg = None,title = "simulation du model seriel"):
        self.evoked_sim.plot(titles=title)
        if eeg is not None:
            self.Epochs_subject[eeg].average().plot()


    # def get_info(self):
    #     print(f"shape de average group = {self.Average_group.get_data().shape}")
    #     print(f"shape de epoch subject = {self.Epoch_subject.get_data().shape}")
    #     print(f"shape de simulation raw = {self.raw_sim_eeg.get_data().shape}")
    #     print(f"shape de simulation evoked = {self.evoked_sim.get_data().shape}")


class Group_simulation():
    def __init__(self,list_simulations,model_type = None) :
        self.simulations = list_simulations
        self.model_type = model_type
        self.rmse = compute_group_RMSE(self.simulations)
        self.list_rmse = self.rmse[0]
        self.array_mean_rmse = self.rmse[1]
        self.array_std_rmse = self.rmse[2]
        self.array_median_rmse = self.rmse[3]
        self.array_subject = self.rmse[4]
        self.df_rmse = make_df_rmse(self.array_mean_rmse,self.array_std_rmse,self.array_median_rmse,self.array_subject)
    
    def set_model_type(self,model_type):
        self.model_type = model_type

    def plot_rmse(self,sort = 'mean', info_type = "std"):
        
        if sort in ['mean' , 'median', 'std']:
            df = self.df_rmse.sort_values(f"{sort}_rmse")
        else :
            print("sort ne peut etre seulement 'mean', 'median', ou 'std'")
           
        sns.set(rc={"figure.figsize": (15, 8)})
        plt.title(f"Score RMSE par sujet pour le model {self.model_type} trie en fonction de {sort}")
        ax = sns.barplot(data = df,x = "name_subject", y = f"{sort}_rmse")
        
        if info_type == "std":
        
            plt.errorbar(
                df["name_subject"],
                df[f"{sort}_rmse"],
                yerr=df["std_rmse"],   
                fmt='none',
                ecolor='black',
                capsize=5
            )
        elif info_type == "value":

            ax.bar_label(ax.containers[0], fontsize=10, fmt='%d')

        plt.show()    

    def display_main_info(self):
        print(f"mean rmse = {np.mean(self.array_mean_rmse) * 1e6}") 
        print(f"std rmse = {np.std(self.array_std_rmse)* 1e6}")
        print(f"median rmse = {np.median(self.array_median_rmse)* 1e6}")