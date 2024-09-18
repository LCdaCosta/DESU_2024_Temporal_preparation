# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 09:55:02 2024
@author: Louis-Clément
"""

# Custom import
from functions import normalize_trial_numbers, create_lag_features, filter_outliers_std

# Import
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


##############################
### Chargement des données ###
##############################

# Lire le fichier Excel
file_path = 'C:\\Users\ContactM2\Documents\LC\DESU_LC\Projet\RT_DATA_Jenny\Ruby TempExp single helper.xlsx'

# Charger toutes les feuilles du fichier Excel
xls = pd.ExcelFile(file_path)

# Liste pour stocker les DataFrames de chaque feuille
dfs = []

# Pour chaque feuille dans le fichier Excel
for sheet_name in xls.sheet_names:
    # Lire la feuille
    df = pd.read_excel(xls, sheet_name=sheet_name, header=None)
    
    # Extraire les lignes 14 à 113 et colonnes 1 à 76 (correspondant à A à BX)
    df_extrait = df.iloc[13:113, 0:76]  # Attention, pandas utilise des index 0-based (donc ligne 14 = index 13)
    df_extrait = df_extrait.drop(df_extrait.columns[55], axis=1)
    
    df_extrait.columns = df_extrait.iloc[0]
    df_extrait=df_extrait.drop(13)
    # Ajouter le DataFrame extrait à la liste
    dfs.append(df_extrait)

del(dfs[0])
# Optionnel : concaténer tous les DataFrames extraits en un seul
df_final = pd.concat(dfs, ignore_index=True)

# Analyse par sujet 

#####

##############################
### Preprocessing          ###
##############################


# Sélectionner les variables 
variables=["Subject","Age","Sex","Trial","Foreperiod","Running[Trial]","Clean_RT"]
df_final_red=df_final[variables]

#Renommer Running[Trial]
df_final_red = df_final_red.rename(columns={'Running[Trial]': 'Block_number'})

#Droper les NaN, Attention cela drop donc les "Catch trials" !!
df_final_red=df_final_red.dropna()

### Encodage des variables catégorielles : One-Hot Encoding 
df_final_red = pd.get_dummies(df_final_red, columns=["Sex","Block_number"], drop_first=True)
df_final_red = df_final_red.rename(columns={'Sex_male': 'Sex'})


##############################
### Exploration           ###
##############################


palette="coolwarm"


# Pairplot
df_final_red = df_final_red.apply(pd.to_numeric, errors='coerce')
df_numeric = df_final_red.select_dtypes(include=['number'])
sns.pairplot(df_numeric,kind="scatter")
plt.show()

#Age des sujets
df_final_red= df_final_red.sort_values(by="Subject")
plt.scatter(df_final_red["Subject"],df_final_red["Age"])
plt.xlabel("Subject")
plt.ylabel("Age")
plt.show()

# Distribution des ages
plt.figure(figsize=(10, 6))
sns.histplot(df_final_red['Age'], bins=range(df_final_red['Age'].min(), df_final_red['Age'].max() + 5, 5), color='skyblue', kde=True)
plt.xlabel('Âge')
plt.ylabel('Fréquence')
plt.title('Distribution des Âges')
plt.show()

#RT en fonction des ages
df_final_red= df_final_red.sort_values(by="Age")
plt.scatter(df_final_red["Age"],df_final_red["Clean_RT"])
plt.xlabel("Age")
plt.ylabel("Clean_RT")
plt.show()

## Distribution des RT en fonction des Foreperiod avec hue ="Age"
plt.scatter(df_final_red["Foreperiod"], df_final_red["Clean_RT"], c=df_final_red["Age"], cmap='coolwarm')
plt.colorbar(label='Age')
plt.xlabel('Foreperiod')
plt.ylabel('Clean_RT')
plt.title('Scatter Plot coloré par Age')
plt.show()

## Distribution des RT en fonction des Foreperiod et du sexe"
plt.scatter(df_final_red["Foreperiod"], df_final_red["Clean_RT"], c=df_final_red["Sex"], cmap='coolwarm')
plt.xlabel('Foreperiod')
plt.ylabel('Clean_RT')
plt.title('Scatter Plot coloré par sexe')
plt.show()


############## Séparation du dataset en 2 groupes (jeunes - vieux)

df_final_red['young'] = df_final_red['Age'] < 15


fig, ax = plt.subplots(figsize=(8, 6))

# Barplot des RT en fonction du groupe jeune ou vieux par foreperiods
sns.scatterplot(data=df_final_red, x='Foreperiod', y='Clean_RT', hue='young', palette=palette, ax=ax)
ax.set_title('Scatter plot of RT by group')
ax.set_xlabel('Foreperiod')
ax.set_ylabel('Clean_RT')
ax.tick_params(axis='x', rotation=45)


# Ajuster la mise en page pour afficher les deux graphiques proprement
plt.tight_layout()
plt.show()


############## ############## ############## 

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),sharey=True)

# Barplot des RT en fonction du groupe jeune ou vieux par foreperiods
sns.barplot(data=df_final_red, x='Foreperiod', y='Clean_RT', hue='young', palette=palette, ax=ax1)
ax1.set_title('Barplot of Clean_RT grouped by young or old')
ax1.set_xlabel('Foreperiod')
ax1.set_ylabel('Clean_RT')
ax1.tick_params(axis='x', rotation=45)

# Violinplot des RT en fonction du groupe jeune ou vieux par foreperiods
sns.violinplot(data=df_final_red, x='Foreperiod', y='Clean_RT', hue='young', palette=palette, ax=ax2)
ax2.set_title('Violinplot of Clean_RT grouped by young or old')
ax2.set_xlabel('Foreperiod')
ax2.set_ylabel('Clean_RT')
ax2.tick_params(axis='x', rotation=45)
#ax2.legend(title='Young', bbox_to_anchor=(1.05, 1), loc='upper left')

# Ajuster la mise en page pour afficher les deux graphiques proprement
plt.tight_layout()
plt.show()


##########################################################################################
############################## Analyses ########################"
##########################################################################################


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Fonction qui réindex les numéros des essais (enlève les catch trials)
df_final_red_norm = normalize_trial_numbers(df_final_red)

# Liste pour stocker les mse en fonction du groupe
mse_lags_young=[]
mse_lags_old=[]
R2_old_list=[]
R2_young_list=[]

for i in range (0,11):
    
    df_lags= create_lag_features(df_final_red_norm,i)
    
    ################################################################################
    
    ## Diviser le dataset en 2 groupes (young and old)
    
    df_lags_old = df_lags[df_lags['Age'] >= 10]
    df_lags_young =df_lags[df_lags['Age'] < 10]
    
    ################################################################################
    ## Attention enlever les Clean_RT qui sont des outliers
    
    ##############################
    ### Filtrage des outliers  ###
    ##############################
    
    # Appliquer la fonction de filtrage à chaque groupe de Foreperiod (filtrage si sup ou inf à 3std)
    filtered_df_lags_old = df_lags_old.groupby("Foreperiod").apply(filter_outliers_std)
    filtered_df_lags_young = df_lags_young.groupby("Foreperiod").apply(filter_outliers_std)
    
    # Réinitialiser l'index si nécessaire
    filtered_df_lags_old = filtered_df_lags_old.reset_index(drop=True)
    # Réinitialiser l'index si nécessaire
    filtered_df_lags_young = filtered_df_lags_young.reset_index(drop=True)


    ##########################################################################################
    ############################## Modelisation ########################"
    ##########################################################################################
    
    
    ##################    Modèle fit (linear regression) Clean RT old    ################## 
    
    # Définir les caractéristiques (features) et la cible (target)
    X = filtered_df_lags_old.drop(["Subject","Sex","Trial","Block_number_Block2","Block_number_Block3","young","Clean_RT"],axis=1)
    # X = df_lags_old[['Foreperiod',"Trial"]]
    y = filtered_df_lags_old['Clean_RT']
    
    # Séparer les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    
    #Scaler les données train par Robust scaler
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_scaled_df=pd.DataFrame(X_train_scaled, columns=X.columns)
    
    # Créer et entraîner le modèle de régression
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    #Scaler les données tests avec le même scaler
    X_test_scaled=scaler.transform(X_test)
    
    # Prédictions
    y_pred = model.predict(X_test_scaled)
    
    # Évaluer le modèle
    mse1 = mean_squared_error(y_test, y_pred)
    mse_lags_old.append(mse1)
    R2_old=model.score(X_test_scaled, y_test, sample_weight=None)
    R2_old_list.append(R2_old)
    
    print(f"Mean Squared Error old: {mse1}")
    print(f"R2 old: {R2_old}")
    
    y_test_old=y_test
    y_pred_old=y_pred
    
    # Afficher les prédictions vs valeurs réelles
    # plt.scatter(y_test, y_pred)
    # plt.xlabel('True Values')
    # plt.ylabel('Predictions')
    # plt.title(f'True Values vs Predictions df_lags_old - MSE : {mse1}')
    # plt.show()
    
    
    ####################     Modèle fit Clean RT young
    
    # # Définir les caractéristiques (features) et la cible (target)
    X = filtered_df_lags_young.drop(["Subject","Sex","Trial","Block_number_Block2","Block_number_Block3","young","Clean_RT"],axis=1)
    y = filtered_df_lags_young['Clean_RT']
    # X = df_lags_young[['Foreperiod',"Trial"]]
    
    
    # Séparer les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    
    #Scaler les données train par Robust scaler
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_scaled_df=pd.DataFrame(X_train_scaled, columns=X.columns)
    
    # Créer et entraîner le modèle de régression
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    #Scaler les données tests avec le même scaler
    X_test_scaled=scaler.transform(X_test)
    
    # Prédictions
    y_pred = model.predict(X_test_scaled)
    
    # Évaluer le modèle
    mse2 = mean_squared_error(y_test, y_pred)
    mse_lags_young.append(mse2)
    R2_young=model.score(X_test_scaled, y_test, sample_weight=None)
    R2_young_list.append(R2_young)
    
    print(f"Mean Squared Error young: {mse2}")
    print(f"R2 young: {R2_young}")
    
    ### Visualisation résultats
    
    # Afficher les prédictions vs valeurs réelles
    plt.scatter(y_test, y_pred, c="red", label="Group young")
    plt.scatter(y_test_old, y_pred_old, c="blue", label="Group old")
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(f'True Values vs Predictions \nlags : {i}')
    plt.legend()
    plt.show()
    
   

### Visualisation résultats


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### •Résultats modélisation régression linéaire en fonction des Lags
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


# MSE et R^2 en fonction du nombre de lags 

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

lin = np.arange(0, len(mse_lags_old))

# Trouver les indices des meilleurs lags pour MSE et R^2
best_lag_old_mse_index = mse_lags_old.index(min(mse_lags_old))
best_lag_young_mse_index = mse_lags_young.index(min(mse_lags_young))
best_lag_old_r2_index = R2_old_list.index(max(R2_old_list))
best_lag_young_r2_index = R2_young_list.index(max(R2_young_list))

# Tracer MSE
ax[0].plot(lin, mse_lags_old, c='blue', label="MSE group old")
ax[0].plot(lin, mse_lags_young, c='red', label="MSE group young")

# Ajouter des lignes verticales et horizontales pour le meilleur lag (MSE)
ax[0].axvline(x=best_lag_old_mse_index, color='blue', linestyle='--', alpha=0.6)
ax[0].axhline(y=mse_lags_old[best_lag_old_mse_index], color='blue', linestyle='--', alpha=0.6)
ax[0].axvline(x=best_lag_young_mse_index, color='red', linestyle='--', alpha=0.6)
ax[0].axhline(y=mse_lags_young[best_lag_young_mse_index], color='red', linestyle='--', alpha=0.6)
ax[0].scatter(best_lag_old_mse_index, mse_lags_old[best_lag_old_mse_index], c='blue', marker='o', label=f"Best lag old: {best_lag_old_mse_index}, MSE: {mse_lags_old[best_lag_old_mse_index]:.2f}")
ax[0].scatter(best_lag_young_mse_index, mse_lags_young[best_lag_young_mse_index], c='red', marker='o', label=f"Best lag young: {best_lag_young_mse_index}, MSE: {mse_lags_young[best_lag_young_mse_index]:.2f}")

ax[0].set_xlabel('Number of lag')
ax[0].set_ylabel('MSE from the linear regression')
ax[0].legend()

# Tracer R^2
ax[1].plot(lin, R2_old_list, c='blue', label="R^2 group old")
ax[1].plot(lin, R2_young_list, c='red', label="R^2 group young")

# Ajouter des lignes verticales et horizontales pour le meilleur lag (R^2)
ax[1].axvline(x=best_lag_old_r2_index, color='blue', linestyle='--', alpha=0.6)
ax[1].axhline(y=R2_old_list[best_lag_old_r2_index], color='blue', linestyle='--', alpha=0.6)
ax[1].axvline(x=best_lag_young_r2_index, color='red', linestyle='--', alpha=0.6)
ax[1].axhline(y=R2_young_list[best_lag_young_r2_index], color='red', linestyle='--', alpha=0.6)
ax[1].scatter(best_lag_old_r2_index, R2_old_list[best_lag_old_r2_index], c='blue', marker='o', label=f"Best lag old: {best_lag_old_r2_index + 1}, R²: {R2_old_list[best_lag_old_r2_index]:.2f}")
ax[1].scatter(best_lag_young_r2_index, R2_young_list[best_lag_young_r2_index], c='red', marker='o', label=f"Best lag young: {best_lag_young_r2_index + 1}, R²: {R2_young_list[best_lag_young_r2_index]:.2f}")

ax[1].set_xlabel('Number of lag')
ax[1].set_ylabel("Rsquared")
ax[1].legend()

plt.tight_layout()
plt.show()


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### •Résultats Moyennes en fonction de Foreperiods / groupes
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


# Calculer la moyenne de Clean_RT pour chaque Foreperiod
mean_RT_foreperiod_old = filtered_df_lags_old.groupby("Foreperiod")["Clean_RT"].mean().reset_index()
mean_RT_foreperiod_young = filtered_df_lags_young.groupby("Foreperiod")["Clean_RT"].mean().reset_index()
std_RT_foreperiod_old = filtered_df_lags_old.groupby("Foreperiod")["Clean_RT"].std().reset_index()
std_RT_foreperiod_young = filtered_df_lags_young.groupby("Foreperiod")["Clean_RT"].std().reset_index()

    # Tracer les moyennes avec barres d'erreur pour les deux groupes (old et young)
plt.errorbar(mean_RT_foreperiod_old["Foreperiod"], 
             mean_RT_foreperiod_old["Clean_RT"], 
             yerr=std_RT_foreperiod_old["Clean_RT"],   # Barres d'erreur pour le groupe "Old"
             label='Old', 
             fmt='-o', capsize=5)  # 'fmt' pour le format du tracé et 'capsize' pour la taille des barres

plt.errorbar(mean_RT_foreperiod_young["Foreperiod"], 
             mean_RT_foreperiod_young["Clean_RT"], 
             yerr=std_RT_foreperiod_young["Clean_RT"], # Barres d'erreur pour le groupe "Young"
             label='Young', 
             fmt='-o', capsize=5)

# Ajouter des labels et un titre
plt.xlabel("Foreperiod")
plt.ylabel("Mean Clean_RT")
plt.title("Mean Clean_RT for Each Foreperiod (Old vs Young) avec std")
plt.legend()
# Ajouter des xticks uniquement pour les valeurs uniques de Foreperiod
unique_foreperiods = df_lags_old["Foreperiod"].unique()
plt.xticks(unique_foreperiods)
plt.show()


### Avec l'erreur standard   

std_RT_foreperiod_old = filtered_df_lags_old.groupby("Foreperiod")["Clean_RT"].std().reset_index(name="std")
std_RT_foreperiod_young = filtered_df_lags_young.groupby("Foreperiod")["Clean_RT"].std().reset_index(name="std")

# Calculer la taille de l'échantillon pour chaque Foreperiod
count_RT_foreperiod_young = filtered_df_lags_young.groupby("Foreperiod")["Clean_RT"].count().reset_index(name="count")
# Fusionner les deux DataFrames pour avoir la taille de l'échantillon et l'écart-type ensemble
merged_data_young = pd.merge(std_RT_foreperiod_young, count_RT_foreperiod_young, on="Foreperiod")
# Calculer l'erreur standard pour chaque Foreperiod
merged_data_young["SE"] = merged_data_young["std"] / np.sqrt(merged_data_young["count"])

# Calculer la taille de l'échantillon pour chaque Foreperiod (groupe Old)
count_RT_foreperiod_old = filtered_df_lags_old.groupby("Foreperiod")["Clean_RT"].count().reset_index(name="count")    

# Fusionner les deux DataFrames pour avoir la taille de l'échantillon et l'écart-type ensemble
merged_data_old = pd.merge(std_RT_foreperiod_old, count_RT_foreperiod_old, on="Foreperiod")
# Calculer l'erreur standard pour chaque Foreperiod (groupe Old)
merged_data_old["SE"] = merged_data_old["std"] / np.sqrt(merged_data_old["count"])

# Tracer avec les barres d'erreur pour le groupe Old
plt.errorbar(mean_RT_foreperiod_old["Foreperiod"], 
             mean_RT_foreperiod_old["Clean_RT"], 
             yerr=merged_data_old["SE"],  # Utilisation de l'erreur standard (SE)
             label='Old', 
             fmt='-o', capsize=5)
plt.errorbar(mean_RT_foreperiod_young["Foreperiod"], 
         mean_RT_foreperiod_young["Clean_RT"], 
         yerr=merged_data_young["SE"],  # Utilisation de l'erreur standard (SE)
         label='Young', 
         fmt='-o', capsize=5)

# Afficher le graphique
plt.xlabel("Foreperiod")
plt.ylabel("Mean Clean_RT")
plt.title("Mean Clean_RT for Each Foreperiod (Old vs Young) avec SE")
plt.legend()
unique_foreperiods = df_lags_old["Foreperiod"].unique()
plt.xticks(unique_foreperiods)
plt.show()