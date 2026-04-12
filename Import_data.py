import pandas as pd
import mne
import numpy as np
import Tools

def import_stim_response(path_stim,path_reponse, category):
    """
    this function is about creating a DataFrame (pandas object) of the information contained in stimulation and answer files

    path_stim is one single path of stimulation file
    path_reponse is one single path of answer file of the same subject as path_stim

    category is a dict which class the time windows in size category depending on latency between stimulation and answer
    dict = {
        "category A" : (min, max),
        "category B" : (min, max),
        "category C" : (min, max) ...
    }
    min and max are latency lower and upper limits (in second) to considere a trial in category A,B,C...
    
    for example : 
    
    category = {
        "S" : (0.5,0.8),
        "M" : (0.8,1.0),
        "L" : (1.0,1.2),
        "XL": (1.2,1.4),
        "Too_Long" : (1.4,3)
    }
    """

    with open(path_stim, "r", encoding="utf-8") as f:
        stims = f.read()

    with open(path_reponse, "r", encoding="utf-8") as f:
        reponses = f.read()

    dict_stims = Tools.txt_to_table(stims,["id_stim","time_stim","???_stim"])
    dict_reponses = Tools.txt_to_table(reponses,["id_answer","time_answer","???_answer"])

    df_stims = pd.DataFrame(Tools.take_off_bad_stim(pd.DataFrame(dict_stims)))
    df_reponses = pd.DataFrame(dict_reponses)
    df = pd.concat([df_stims,df_reponses],axis=1)

    df["id_stim"] = df["id_stim"].astype(int)
    df["latency"] = df["time_answer"] - df["time_stim"]

    

    df["size_category"] = "too_long"
    for key in category.keys():
        index = df[df["latency"].between(left = category[key][0],right = category[key][1],inclusive='left')].index
        df.loc[index,"size_category"] = key

    df["time_answer"] = (df["time_answer"] * 512).astype(int)
    df["time_stim"] = (df["time_stim"] * 512).astype(int)
    df["latency"] = (df["latency"] * 512).astype(int)
    return df

def Make_path_list(path,subject_to_pop,nb_sub=None):
    """
    this function is about create a list of each subject eeg path, stimulation path and answer path to use get_df_description() and raw_to_epoch_data()
    easily with all the data
    
    path = is the path of the folder containing all the data (stimulation = sub_cleaned.txt | answer = sub_cleaned_reponse.txt | eeg = sub_DS.bdf)
    subject_to_pop = is a list of integer representing subject rejected 14 and 17 can not be analysed
    nb_sub = is if you want to select the x first subject only
    """
    data_files = []
    stim_files = []
    reponse_files = []
    for i in range(1,31):
        if i < 10:
            participant = "S0" + str(i)
        else :
            participant = "S" + str(i)

        stim_files.append(path + participant + "_cleaned.txt")
        reponse_files.append(path + participant + "_cleaned_response.txt")
        data_files.append(path + participant + "_DS.bdf")

    data_files = Tools.pop_list(data_files,subject_to_pop)
    stim_files = Tools.pop_list(stim_files,subject_to_pop)
    reponse_files = Tools.pop_list(reponse_files,subject_to_pop)
    if nb_sub is None:
        return data_files,stim_files,reponse_files
    else:
        return data_files[:nb_sub+1],stim_files[:nb_sub+1],reponse_files[:nb_sub+1]
    





####################################### Analyse descriptive #############################################

def import_df_description(list_path_stim,list_path_reponse,category, path_excel_participants, subject_to_pop) :

    def get_df_description(list_path_stim,list_path_reponse,category):
        """
        this function is about creating a DataFrame (pandas object) of the information contained in stimulation and answer files

        list_path_stim is path of each stimulation file
        path_reponse is path of each answer file of the same subject as path_stim

        category is a dict which class the time windows in size category depending on latency between stimulation and answer
        dict = {
            "category A" : (min, max),
            "category B" : (min, max),
            "category C" : (min, max) ...
        }
        min and max are latency lower and upper limits (in second) to considere a trial in category A,B,C...
        
        for example : 
        
        category = {
            "S" : (0.5,0.8),
            "M" : (0.8,1.0),
            "L" : (1.0,1.2),
            "XL": (1.2,1.4),
            "Too_Long" : (1.4,3)
        }
        """

        def get_one_description(path_file_stim,path_file_reponse,category=category):

            def find_participant(path,signe = 'S') :
                for i, lettre in enumerate(path):
                    if lettre == signe and str.isnumeric(path[i+1]):
                        return (i, i+3)
                
                print(f"aucun participant trouve sous le nom de {signe}XX")
                return
                
            with open(path_file_stim, "r", encoding="utf-8") as f:
                stims = f.read()

            with open(path_file_reponse, "r", encoding="utf-8") as f:
                reponses = f.read()
            
            dict_stims = Tools.txt_to_table(stims,["id_stim","time_stim","???_stim"])
            dict_reponses = Tools.txt_to_table(reponses,["id_reponse","time_reponse","???_reponse"])

            df_stims = pd.DataFrame(dict_stims)
            df_stims = Tools.take_off_bad_stim(df_stims)
            df_reponses = pd.DataFrame(dict_reponses)

            df = pd.concat([df_stims,df_reponses],axis=1)
            df["latency"] = df["time_reponse"] - df["time_stim"]

            df["size category"] = "too_long"
            for key in category.keys():
                index = df[df["latency"].between(left = category[key][0],right = category[key][1],inclusive='left')].index
                df.loc[index,"size category"] = key
            start,end = find_participant(path_file_stim,'S')
            df["participant"] = path_file_stim[start:end]

            mapp2 = {
            "id_reponse" : "id_answer",
            "time_reponse" : "time_answer",
            "size category" : "size_category",
            "participant" : "subjects"
            }

            df = df.rename(columns = mapp2)

            return df, path_file_stim[start:end]

        liste_df = []
        for stim,reponse in zip(list_path_stim,list_path_reponse) :
            df , sub = get_one_description(stim,reponse)
            liste_df.append(df)


        df = pd.concat(liste_df,ignore_index=True)
        return df


    def clean_p_file(path,subject_to_pop):
        """
        use to create and clean the excel file of subject features
        path = path ofthe excel file
        """
        df = pd.read_excel(path)
        df = df[['participant', 'liste','genre', 'age','sommail','trouble néuro','médicament']]
        df = df.iloc[0:len(df)-2]

        map = {
        "mauvais" : 0,
        "plus court que d'habitude" : 1,
        "habituel" : 2,
        "excellent" : 3
        }

        def function_a(x):
            if x == "NON":
                return 0
            else : 
                return 1

        df["sommail"] = df["sommail"].replace(map)
        df["trouble néuro"] =  df["trouble néuro"].apply(function_a)
        df["médicament"] = df["médicament"].apply(function_a)
        to_pop = np.array(subject_to_pop)-1 
        df = df.drop(to_pop)

        return df
        
    def clean_df_time_window(df_time_window):
    
        liste_epoch = list(df_time_window.index)
        liste_label_epoch = []
        for el in liste_epoch:
            liste_label_epoch.append(f"Epoch_{el}")
        df_time_window["Epochs"] = liste_label_epoch
        df_time_window = df_time_window.drop(["???_stim","???_reponse"], axis = 1)
        return df_time_window


    def clean_df_participant(df_participants,df_time_window): 

        moyennes = []
        stds = []
        nb = []
        participants = []
        maxi = []
        mini = []
        for p in df_time_window["subjects"].unique():
            sub_df = df_time_window[df_time_window["subjects"] == p]
            moyennes.append(sub_df["latency"].mean())
            stds.append(sub_df["latency"].std())
            maxi.append(sub_df["latency"].max())
            mini.append(sub_df["latency"].min())
            nb.append(len(sub_df))
            participants.append(p)

        df_participants["participant_check"] = participants
        df_participants["mean_latency"] = moyennes
        df_participants["std_latency"] = stds

        df_participants["nb_reponse"] = nb
        df_participants["Good_answers_rate"] = df_participants["nb_reponse"] / 160
        df_participants["max_latency"] = maxi
        df_participants["min_latency"] = mini

        map = {
        "genre" : "gender",
        "participant" : "subjects",
        "sommail" : "sleep",
        "trouble néuro" : "neural_disease",
        "participant_check" : "subjects_check",
        "médicament" : "medication"
        }

        df_participants = df_participants.rename(columns = map)

        return df_participants.drop("nb_reponse",axis = 1)
    
    def concatenation_participant_time_window(df_participants,df_time_window):
        df_out = None
        for sub in df_participants["subjects"]:
            # print("#############################################")
            # print(f"subject = {sub}")
            sub_df_tw = df_time_window[df_time_window["subjects"] == sub]
            sub_df_p = df_participants[df_participants["subjects"] == sub]

            # print(f"sub_df_tw.head() = {sub_df_tw.head()}")
            # print(f"sub_df_p = {sub_df_p}")
            # print("#############################################")

            for col in sub_df_p.drop(["subjects"],axis = 1).columns:
                # print(f"=========== col {col} =================")
                # print(f"sub_df_p[{col}] = {list(sub_df_p[col])[0]}")
                sub_df_tw[col] = list(sub_df_p[col])[0]
                # print(f"sub_df_tw = {sub_df_tw.columns}")
            if df_out is None:
                df_out = sub_df_tw
            else : 
                df_out = pd.concat([df_out,sub_df_tw])
        return df_out.drop("subjects_check",axis = 1)
        

    df_participants = clean_p_file(path_excel_participants,subject_to_pop=subject_to_pop)
    df_time_window = get_df_description(list_path_stim,list_path_reponse,category = category)

    description_tw  = clean_df_time_window(df_time_window)
    description_p = clean_df_participant(df_participants,description_tw)

    df_all = concatenation_participant_time_window(description_p,description_tw)
    
    return df_all