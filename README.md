# Extreme value statistics of T cell clone sizes
This project is my M1 internship.
It is aimed at studying the statistics of the more frequent clonotypes (top clones) of T cell repertoires in the framework of the theory of extremes (EVS).

Here I provide the more important codes used in the process. The user can find:
    - emerson_data_analysis.py : code to clean and rewrite the original datasets containing the repertoires
    - EVS_functions.py : code containing all the important statistical functions representing the problem (PDFs and CDFs mainly)
    - normalised_plaw_generator.py : code to numerically generate sequences of clonotypes with normalisation over the frequency sum
    - power_law_analysis.py : code to analyse and characterise the power law behaviour of the repertoires
    - RL_integrator.py : advanced integrator (modified version of the original: https://github.com/differint)
    
Additionally, some example notebooks are provided, in order to see how to apply these codes. Namely:
    - example_data_analysis : patient data analysis
    - example_nEVS : normalised EVS evaluation and fit
    - example_mEVS : mixture EVS evaluation and fit
    - example_simulation : simulation of numerically generated data
    - example_plaw_analysis : power law analysis of some patients
    
along with some data files needed to make these codes work:
    - HIP00110 : test patient
    - reduced_patients : list with all the patients' names
    - top_1_emerson_clones : DataFrame of all the top clones
