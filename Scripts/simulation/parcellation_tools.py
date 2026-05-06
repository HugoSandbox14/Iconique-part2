import numpy as np
import mne 

class Parcellation():
    def __init__(self,subjects_dir,subject):

        print("Parcellation...")
        self.labels = mne.read_labels_from_annot(subject=subject,parc='aparc',subjects_dir=subjects_dir)
        print("label ok...")
        self.dict_function = make_dict_function(self.labels)
        print("dict_function ok...")
        self.dict_vertices = make_dict_vertices(self.dict_function)
        print("dict_vertices ok...")

class Forward():
    def __init__(self,subjects_dir,subject):
        
        self.bem = make_bem(subject,subjects_dir)
        self.src = mne.setup_source_space(subject=subject,spacing='ico5',subjects_dir=subjects_dir,add_dist=False)
        self.info = make_info()
        self.forward = make_fwd(self.info,self.src,self.bem)
        self.G = self.forward["sol"]["data"]
        self.G_mean_reference = self.G - np.mean(self.G, axis=0, keepdims=True)

####################### Fontions pour Forward ##########################

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

######################## Fonctions pour Parcellation ########################


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
    lexico_phono = lexico_phono.extend(temp_regions_supp)
    phono = broca_region
    articulation = region_motrice
    bruit = region_bruit[:4]
    bruit = bruit.extend(region_bruit[6:])

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
