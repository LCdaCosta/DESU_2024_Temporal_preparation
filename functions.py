# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 15:40:10 2024
@author: ContactM2
"""


import numpy as np
import pandas as pd


def normalize_trial_numbers(df):
    """
    Remplace les numéros d'essai d'origine par une séquence de 1 à max pour chaque sujet.
    Parameters:
    - df: DataFrame contenant les colonnes 'Subject', 'Trial', 'Foreperiod', 'Clean_RT'
    Returns:
    - DataFrame avec les numéros d'essai normalisés
    
    """
    # Assure que les données sont triées par sujet et essai
    df = df.sort_values(by=['Subject', 'Trial'])
    
    # Créer un DataFrame vide pour stocker les résultats
    normalized_dfs = []
    
    # Traiter chaque sujet individuellement
    for subject in df['Subject'].unique():
        subject_df = df[df['Subject'] == subject].copy()
        
        # Trouver le nombre d'essais pour le sujet
        num_trials = len(subject_df)
        
        # Créer une séquence de 1 à num_trials pour ce sujet
        subject_df['Normalized_Trial'] = np.arange(1, num_trials + 1)
        
        # Ajouter le DataFrame avec les numéros d'essai normalisés à la liste
        normalized_dfs.append(subject_df)
    
    # Concaténer les DataFrames pour obtenir le DataFrame final
    df_normalized = pd.concat(normalized_dfs)
    
    return df_normalized



def create_lag_features(df,num_trials):
    """
    Crée des caractéristiques basées sur les essais précédents pour chaque sujet,
    en adaptant automatiquement le nombre d'essais précédents en fonction des données disponibles.

    Parameters:
    - df: DataFrame contenant les colonnes 'Subject', 'Trial', 'Foreperiod', 'Clean_RT'
      num_trials : int = nombre de lags à intégrer dans le nouveau dataset

    Returns:
    - DataFrame avec les caractéristiques des essais précédents
      Value : nombre de lag max 
      Liste : liste des mse de l'OLS pour chaque lag supplémentaire'
    
    """
    df = df.sort_values(by=['Subject', 'Normalized_Trial'])
    subjects = df['Subject'].unique()
    Foreperiod_stand=600
    
    
    # Pour chaque sujet, créer des colonnes pour les essais précédents
    for subject in subjects:
        subject_df = df[df['Subject'] == subject]
        
        ## max_lags
        #num_trials = len(subject_df)
        
        mean_RT=subject_df['Clean_RT'].mean()

        
        for lag in range(1, num_trials):
            df.loc[df['Subject'] == subject, f'Foreperiod_prev_{lag}'] = subject_df['Foreperiod'].shift(lag)
            # Compléter les NaN par la valeur standard de la Foreperiod (ici 600ms)
            df=df.fillna(Foreperiod_stand)
            df.loc[df['Subject'] == subject, f'Clean_RT_prev_{lag}'] = subject_df['Clean_RT'].shift(lag)
            # Compléter les NaN par la valeur moyenne du RT du sujet
            df=df.fillna(mean_RT)
            
            

    return df

# Fonction de filtrage
def filter_outliers_std(df):
    
    """
    Filtre les outliers de df en fonction de Clean_RT

    Parameters:
    - df: DataFrame contenant les colonnes ''Foreperiod' et 'Clean_RT'

    Returns:
    - DataFrame filtré selon la méthode des std si les valeurs de Clean_RT sont supérieures à 3std 
    
    """
    
    mean_RT = df["Clean_RT"].mean()
    std_RT = df["Clean_RT"].std()
    lower_bound = mean_RT - 3 * std_RT
    upper_bound = mean_RT + 3 * std_RT
    
    # Retourner les lignes sans outliers
    return df[(df["Clean_RT"] >= lower_bound) & (df["Clean_RT"] <= upper_bound)]

def replace_outliers_with_mean(df):
    
    """
    Remplace les outliers de df en fonction de 'Clean_RT' par la moyenne.

    Parameters:
    - df: DataFrame contenant les colonnes 'Foreperiod' et 'Clean_RT'

    Returns:
    - DataFrame avec les outliers de 'Clean_RT' remplacés par la moyenne
      selon la méthode des 3 écarts-types.
    """
    
    # Calcul de la moyenne et de l'écart-type de Clean_RT
    mean_RT = df["Clean_RT"].mean()
    std_RT = df["Clean_RT"].std()
    
    # Calcul des bornes inférieure et supérieure
    lower_bound = mean_RT - 3 * std_RT
    upper_bound = mean_RT + 3 * std_RT
    
    # Remplacer les valeurs hors des bornes par la moyenne
    df["Clean_RT"] = df["Clean_RT"].apply(lambda x: mean_RT if x < lower_bound or x > upper_bound else x)
    
    return df

def replace_nan_with_group_mean(df, group_col, target_col):
    """
    Remplace les valeurs NaN dans target_col par la moyenne des valeurs non-NaN 
    pour chaque groupe défini par group_col.
    
    Parameters:
    - df: DataFrame contenant les colonnes spécifiées.
    - group_col: Colonne sur laquelle les groupes sont basés (par exemple 'Foreperiod').
    - target_col: Colonne dans laquelle les NaN doivent être remplacés (par exemple 'Clean_RT').
    
    Returns:
    - DataFrame avec les NaN remplacés par la moyenne des valeurs du groupe.
    """
    # Calculer la moyenne pour chaque groupe
    group_means = df.groupby(group_col)[target_col].transform('mean')
    
    # Remplacer les NaN par la moyenne du groupe
    df[target_col] = df[target_col].fillna(group_means)
    
    return df
