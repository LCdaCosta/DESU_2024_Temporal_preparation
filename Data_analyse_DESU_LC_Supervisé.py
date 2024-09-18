# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 10:38:27 2024

@author: ContactM2
"""

# Custom import
from functions import normalize_trial_numbers, create_lag_features, filter_outliers_std, replace_outliers_with_mean
from functions import replace_nan_with_group_mean

# Import
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

##############################
### Chargement des données ###
##############################

# Lire le fichier Excel
file_path = 'C:\\Users\ContactM2\Documents\LC\DESU_LC\Projet\RT_DATA_Jenny\Ruby TempExp single helper.xlsx'

# Charger toutes les feuilles du fichier Excel
xls = pd.ExcelFile(file_path)

# Liste pour stocker les DataFrames de chaque feuille
dfs = []
len_df=[]

# Pour chaque feuille dans le fichier Excel
for sheet_name in xls.sheet_names:
    # Lire la feuille
    df = pd.read_excel(xls, sheet_name=sheet_name, header=None)
    
    # Extraire les lignes 14 à 113 et colonnes 1 à 76 (correspondant à A à BX)
    df_extrait = df.iloc[13:113, 0:76]  # Attention, pandas utilise des index 0-based (donc ligne 14 = index 13)
    df_extrait = df_extrait.drop(df_extrait.columns[55], axis=1)
    
    df_extrait.columns = df_extrait.iloc[0]
    df_extrait=df_extrait.drop(13)
    
    # Sélectionner les variables 
    variables=["Subject","Age","Trial","Foreperiod","Clean_RT"]
    df_extrait=df_extrait[variables]

    ##############################
    ### Preprocessing          ###
    ##############################
    
    # Remplacer les valeurs NAn correspondant aux catch Trials par la valeur moyenne de RT 
    df_extrait = replace_nan_with_group_mean(df_extrait, 'Foreperiod', 'Clean_RT')
    
    # Remplacer les valeurs aberrantes par la moyenne des Clean_RT en fonction de la foreperiod
    filtered_df_extrait=df_extrait.groupby("Foreperiod").apply(replace_outliers_with_mean)

    # Ajouter le DataFrame extrait à la liste
    len_df.append(len(df_extrait))
    dfs.append(df_extrait)
    

# On drop le premier dataset qui était un template (redondant)
del(dfs[0])

# Concaténer tous les DataFrames extraits en un seul
df= pd.concat(dfs, ignore_index=True)


###################################################
### Vérifiaction de l'imputation par la moyenne ###
###################################################


# On vérifie que l'imputation par la moyenne ne change pas les résultats obtenus en droppant les NaN et les outliers
df_old=df[df["Age"]>15]
df_young=df[df["Age"]<15]

# Calculer la moyenne de Clean_RT pour chaque Foreperiod
mean_RT_foreperiod_old = df_old.groupby("Foreperiod")["Clean_RT"].mean().reset_index()
mean_RT_foreperiod_young = df_young.groupby("Foreperiod")["Clean_RT"].mean().reset_index()

# Tracer les moyennes pour les deux groupes (old et young)
plt.plot(mean_RT_foreperiod_old["Foreperiod"], mean_RT_foreperiod_old["Clean_RT"], label='Old')
plt.plot(mean_RT_foreperiod_young["Foreperiod"], mean_RT_foreperiod_young["Clean_RT"], label='Young')

# Ajouter des labels et un titre
plt.xlabel("Foreperiod")
plt.ylabel("Mean Clean_RT")
plt.title("Mean Clean_RT for Each Foreperiod (Old vs Young)")
plt.legend()
    
# Afficher le graphique
plt.show()

# Matrice correlation
correlation_matrix = df[df.columns].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='viridis', fmt='.2f')
plt.show()


#• Listes pour stocker les résultats des différents modèles
test_accuracy_list_SVC=[]

test_accuracy_list_Logreg=[]
LR_CV_mean_list=[]

test_accuracy_list_RF=[]
meanCV_accuracy_list_RF=[]


# On va itérer sur le nombre de lags (pas le plus économe mais bon..)
lags=np.arange(0,98,1)
for lag in lags:
    
    #############################################
    ### Préparation data pour les différents modèles          ###
    #############################################
    
    df=normalize_trial_numbers(df)
    df=create_lag_features(df,lag)
    df["young"] = df['Age'] < 10
    
    # But = Prédire si groupe = jeune ou vieux -> on définit la target y 
    X = df.drop(columns=['Subject', 'young',"Age","Trial","Normalized_Trial"])  
    y = df['young'].astype(int)  # Convertir les booléens en 0/1 pour le modèle
    
    # ---- Division des données en hold-out set (ensemble de test) et entraînement ----
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    
    #############################################
    ### Preprocessing          ###
    #############################################
    
    # On scale les données d'entrainement
    scaler = MinMaxScaler()
    #scaler = StandardScaler()
    
    X_train_scaled = scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_train=X_train_scaled
    
    #On scale les données test
    X_test_scaled = scaler.transform(X_test)
    X_test = X_test_scaled

    
    #############################################
    ###  Modelisation Régression logistique  ###
    #############################################
    
    from sklearn.linear_model import LogisticRegression
    
    # ---- Validation croisée sur l'ensemble d'entraînement ----
    model = LogisticRegression()
    
    # Utilisation de cross_val_score pour la validation croisée (ici 5 folds) sur l'ensemble d'entraînement
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    
    # Résultats de la cross-validation
    print("Cross-validation scores (accuracy for each fold):", cv_scores)
    print(f"Mean cross-validation accuracy: {cv_scores.mean():.2f}")
    print(f"Standard deviation of accuracy: {cv_scores.std():.2f}")
    
    LR_CV_mean_list.append(cv_scores.mean())
    
    # ---- Entraîner le modèle final sur tout l'ensemble d'entraînement ----
    model.fit(X_train, y_train)
    
    # ---- Évaluer le modèle final sur l'ensemble de test ----
    
    #############################################
    ## Prédiction sur le test set ## 
    #############################################

    y_pred_test = model.predict(X_test)
    test_accuracy_LR = accuracy_score(y_test, y_pred_test)
    
    print(f"\nAccuracy on hold-out test set: {test_accuracy_LR:.2f}")
    test_accuracy_list_Logreg.append(test_accuracy_LR)
    
    
    
    #############################################
    ### Modeling avec SVC          ###
    #############################################
    
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score
    
    # ---- Validation croisée sur l'ensemble d'entraînement ----
    model = SVC()
    
    # Utilisation de cross_val_score pour la validation croisée (ici 5 folds) sur l'ensemble d'entraînement
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    
    # Résultats de la cross-validation
    print("Cross-validation scores (accuracy for each fold):", cv_scores)
    print(f"Mean cross-validation accuracy: {cv_scores.mean():.2f}")
    print(f"Standard deviation of accuracy: {cv_scores.std():.2f}")
    
    # ---- Entraîner le modèle final sur tout l'ensemble d'entraînement ----
    model.fit(X_train, y_train)
    
    # ---- Évaluer le modèle final sur l'ensemble de test ----
    
    #############################################
    ## Prédiction sur le test set ## 
    #############################################
    
    y_pred_test = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    print(f"\nAccuracy on hold-out test set: {test_accuracy:.2f}")
    test_accuracy_list_SVC.append(test_accuracy)
    


    #############################################
    ### Modeling avec RandomForestClassifier ###
    #############################################
    
    from sklearn.ensemble import RandomForestClassifier


    # ---- Validation croisée sur l'ensemble d'entraînement ----
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Utilisation de cross_val_score pour la validation croisée (ici 5 folds) sur l'ensemble d'entraînement
    cv_scores_rf = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='accuracy')
    
    # Résultats de la cross-validation
    print("Cross-validation scores (accuracy for each fold - RandomForest):", cv_scores_rf)
    print(f"Mean cross-validation accuracy: {cv_scores_rf.mean():.2f}")
    print(f"Standard deviation of accuracy: {cv_scores_rf.std():.2f}")
    meanCV_accuracy_list_RF.append(cv_scores_rf.mean())
    
    # ---- Entraîner le modèle final sur tout l'ensemble d'entraînement ----
    
    rf_model.fit(X_train, y_train)
    
    # ---- Évaluer le modèle final sur l'ensemble de test ----
    
    # #############################################
    # ## Prédiction sur le test set ## 
    # #############################################
    
    y_pred_test_rf = rf_model.predict(X_test_scaled)
    test_accuracy_rf = accuracy_score(y_test, y_pred_test_rf)
    test_accuracy_list_RF.append(test_accuracy_rf)
    
    print(f"\nAccuracy on hold-out test set (RandomForest): {test_accuracy_rf:.2f}")


# ## VISU SVC modele

best_lag_index = test_accuracy_list_SVC.index(max(test_accuracy_list_SVC))
best_lag = lags[best_lag_index]
max_accuracy = max(test_accuracy_list_SVC)

plt.plot(lags, test_accuracy_list_SVC, label="SVC Accuracy on hold-out test")
#plt.plot(lags, Mean_CV_accuracy_list_SVC, label="SVC Mean CV Accuracy")

plt.axvline(x=best_lag, color='red', linestyle='--', label=f"Best lag: {best_lag}")
plt.axhline(y=max_accuracy, color='gold', linestyle='--', label=f"Max accuracy: {max_accuracy:.2f}")

plt.xlabel("Amount of lags")
plt.ylabel("Accuracy of SVC")
plt.title("SVC Accuracy in function of lags")
plt.legend()
plt.show()

## VISU LOG REG modele

best_lag_index = test_accuracy_list_Logreg.index(max(test_accuracy_list_Logreg))
best_lag = lags[best_lag_index]
max_accuracy = max(test_accuracy_list_Logreg)

plt.plot(lags,test_accuracy_list_Logreg,label="Log. Reg Accuracy on hold-out test")
# plt.plot(lags, LR_CV_mean_list, label="Mean CV")

plt.axvline(x=best_lag, color='red', linestyle='--', label=f"Best lag: {best_lag}")
plt.axhline(y=max_accuracy, color='blue', linestyle='--', label=f"Max accuracy: {max_accuracy:.2f}")

plt.xlabel("Amount of lags")
plt.ylabel("Accuracy of Logistic Reg. model")
plt.title("LogR Accuracy in function of lags")
plt.legend()
plt.show()


# ## VISU RandomForest modele

best_lag_index_rf = test_accuracy_list_RF.index(max(test_accuracy_list_RF))
best_lag_rf = lags[best_lag_index_rf]
max_accuracy_rf = max(test_accuracy_list_RF)

plt.plot(lags, test_accuracy_list_RF, label="RandomForest Accuracy on hold-out test")
# plt.plot(lags, meanCV_accuracy_list_RF, label="RandomForest Mean CV Accuracy")

plt.axvline(x=best_lag_rf, color='red', linestyle='--', label=f"Best lag: {best_lag_rf}")
plt.axhline(y=max_accuracy_rf, color='green', linestyle='--', label=f"Max accuracy: {max_accuracy_rf:.2f}")
plt.xlabel("Amount of lags")
plt.ylabel("Accuracy of RandomForest")
plt.title("RandomForest Accuracy in function of lags")
plt.legend()
plt.show()

# from sklearn.tree import export_graphviz
# import graphviz

# # Extraire un arbre de la forêt
# tree = rf_model.estimators_[0]  # Choisir le premier arbre par exemple

# # Exporter l'arbre sous forme de graphique
# dot_data = export_graphviz(tree, out_file=None, 
#                            feature_names=X.columns,  
#                            class_names=["Not young", "Young"],  
#                            filled=True, rounded=True,  
#                            special_characters=True)  

# # Utilisation de graphviz pour visualiser
# graph = graphviz.Source(dot_data)  
# graph.render("random_forest_tree")  # Sauvegarder l'image
# graph  # Afficher l'image



