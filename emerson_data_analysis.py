import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


""" 
Data analysis of the Emerson cohort immune repertoires 

Functions
=========
    reduce_data : function to reduce and clean the original datasets 
    top_clone_creator : function to create the set of top clones   
"""


def special_data(string):
    """ 
    This function receives the aminoacid sequence of the TCR and
    chekcs whether it contains special characters or not 
    """
    
    var = False
    for let in string:
        if let == '*' or let == '_':
            var = True
    return var

def reduce_data(file_name, read_path, save_path):
    """ 
    This function takes the original dataset, cleans it for the problem 
    and saves it in the same format in a different folder 
        
    Parameters
    ==========
        file_name : string
            name of the patient's dataset file
        read_path : string
            path where the original datasets are stored
        save_path : string
            path where the resulting datasets will be stored
    """
    
    frame = pd.read_csv(read_path + file_name, sep='\t', low_memory = False)
    
    
    ## Rename the aminoacid, clone size and frequency fields and remove the rest ##
    
    frame.insert(0, 'aa', frame['aminoAcid'], True)
    del frame['aminoAcid']
    frame.insert(1, 'count', frame['count (templates/reads)'], True)
    del frame['count (templates/reads)']
    frame.insert(2, 'frequency', frame['frequencyCount (%)'], True)
    del frame['frequencyCount (%)']
    frame = frame[['aa', 'count', 'frequency']]    
    
    
    ## Remove the NaN sequences and sequences with special characters ##
    
    frame = frame[np.invert(pd.isna(frame['aa']))]
    special = []
    for amino in frame['aa']:
        spec_char = special_data(amino)
        special.append(spec_char)

    frame['special'] = special
    frame = frame[frame['special'] == False]
    del frame['special']
    
    
    ## Sum the frequencies of repeated sequences, renormalise and save ##
    
    frame = frame.groupby(['aa'], as_index=False).sum()
    frame['frequency'] = frame['frequency']/np.sum(frame['frequency'])
    
    frame.to_csv(save_path + file_name, sep='\t', index=False) 
        
            
def top_clone_creator(read_path, frame_patients, L, M = 1):
    """
    This function receives a patient dataset and selects the M top clones
    
    Parameters
    ==========
        read_path : string
            path where the reduced datasets and are stored
        frame_patients : pandas DataFrame
            single column DataFrame of strings where each row is a patients' dataset name 
        M : integer
            order statsitics (we pick the top M clones). Default is M = 1
        L : integer
            number of top clones that wants to be collected. For all patients send len(frame_patients)
            
    Output
    ======
        top_clones : pandas DataFrame
            frame with the aminoacid, counts and frequency of the first L top clones
    """       
    
    
    ## The first patient needs to be read separately ##
    
    patient_0 = pd.read_csv(read_path + frame_patients['file_patients'][0], sep='\t', low_memory=False)
    top_clones = patient_0.sort_values(['frequency'], ascending=False).head(M)

    
    ## The top clone of the remaining patients is picked and added to the DataFrame ##
    
    for file_name in frame_patients.file_patients[1:L]:
        new_patient = pd.read_csv(read_path + file_name, sep='\t', low_memory=False)
        new_top_clone = new_patient.sort_values(['frequency'], ascending=False).head(M)
        
        top_clones = pd.concat([top_clones, new_top_clone])

    return top_clones
        
        
       