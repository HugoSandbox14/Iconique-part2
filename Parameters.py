
path_directory = "D:\\Data_stage\\EEG data\\"
path_excel = "D:\\Data_stage\\participants_S5.xlsx"

path_directory = "C:\\Users\\hugop\\Desktop\\Stage Iconique\\Data_stage\\EEG data\\"
path_excel = "C:\\Users\\hugop\\Desktop\\Stage Iconique\\Data_stage\\participants_S5.xlsx"

fmin = 0.1
fmax = 30
time_window_max = 3
subject_to_pop = [14,17,28]


# pour nettoyage des epoch/channel
std_bad_epoch = 2.5    # --> mean + 'std_bad_epoch' x std   | permet de definirt le seuil d'exclusion d'une epoch par rapport au temps de latence entre stimulation et reponse 
th_bad_channel = 75    # --> en Micro Volt                  | permet de definir le seuil de puissance a depasser pour considerer un cannal comme mauvais


# !! dict_category must have the category "too_long" for window which are too_long for the analyse !! and no space inside names !!

dict_category = {
    "(500-650)" : (0.5,0.65),
    "(650-700)" : (0.65,0.7),
    "(700-800)" : (0.7,0.8),
    "(800-900)" : (0.8,0.9),
    "(900-1000)": (0.9,1),
    "(1000-1100)" : (1,1.1),
    "(1100-1200)" : (1.1,1.2),
    "(1200-1300)" : (1.2,1.3),
    "(1300-1400)" : (1.3,1.4),
    "(1400-1600)" : (1.4,1.6),
    "too_long" : (1.6,time_window_max)
}

# dict_category = {
#     "(500-800)" : (0.5,0.8),
#     "(800-1000)" : (0.8,1),
#     "(1000-1200)" : (1,1.2),
#     "(1200-1400)" : (1.2,1.4),
#     "too_long" : (1.4,time_window_max)
# }

dict_category_all = {
    'good' : (0,time_window_max)
}

# dict band permet de définir les bands fréquences en fonction de ce que l'on veut

dico_bands = {
    "theta" : (4,8),
    "alpha" : (8,12),
    "beta" : (12,fmax),
}

# le mapping des régions cérébrales en fonction des numéros des éléctrodes

dico_map_area = {
    "frontal": list(range(98,103)) +list(range(66,96)),
    "temporalG": list(range(103,106)) + list(range(115,122)),
    "temporalD" : list(range(45,48)) + list(range(54,61)),
    "parietal": list(range(1,7)) +list(range(48,54)) +list(range(61,66)) +list(range(106,115)) +list(range(31,36)) + [18,97,96]+ list(range(122,124)) ,
    "occipital": list(range(36,45)) + list(range(7,17)) + list(range(19,30)) + list(range(124,128)),
}
