# Traitement des remarques sur les variables analysées

# 1. OWN_CAR_AGE : Exploration des valeurs manquantes et des valeurs extrêmes
print("\nOWN_CAR_AGE: Analyse des valeurs manquantes et extrêmes")
missing_own_car_age = app_train['OWN_CAR_AGE'].isnull().sum()
print(f"Valeurs manquantes dans OWN_CAR_AGE: {missing_own_car_age} ({missing_own_car_age/len(app_train)*100:.2f}%)")
print(f"Valeurs max/min : {app_train['OWN_CAR_AGE'].max()} / {app_train['OWN_CAR_AGE'].min()}")
high_own_car_age = app_train[app_train['OWN_CAR_AGE'] > 70]
print(f"Nombre de lignes avec OWN_CAR_AGE > 70 : {len(high_own_car_age)}")

# Décision : supprimer ou "caper" les valeurs extrêmes et garder les valeurs manquantes pour l'instant
app_train.loc[app_train['OWN_CAR_AGE'] > 70, 'OWN_CAR_AGE'] = 70
print("Capping des OWN_CAR_AGE > 70 à 70 effectué.")

# 2. AMT_INCOME_TOTAL : Inspection des valeurs élevées et filtrage des valeurs aberrantes
print("\nAMT_INCOME_TOTAL: Analyse des valeurs extrêmes")
plt.figure(figsize=(10,4))
sns.boxplot(x=app_train['AMT_INCOME_TOTAL'])
plt.title('Boxplot: AMT_INCOME_TOTAL')
plt.show()
# Supposons qu'au-delà de 1 million c'est probablement aberrant
high_income_outliers = app_train[app_train['AMT_INCOME_TOTAL'] > 1_000_000]
print(f"Nombre de dossiers avec AMT_INCOME_TOTAL > 1 000 000 : {len(high_income_outliers)}")
print(high_income_outliers[['SK_ID_CURR', 'AMT_INCOME_TOTAL']])
# On peut soit les exclure, soit les "caper"
app_train.loc[app_train['AMT_INCOME_TOTAL'] > 1_000_000, 'AMT_INCOME_TOTAL'] = 1_000_000
print("Capping des AMT_INCOME_TOTAL > 1 000 000 à 1 000 000 effectué.")

# 3. AMT_CREDIT / AMT_GOODS_PRICE / AMT_ANNUITY : vérification des extrêmes, capping sur valeurs suspectes
monetary_cols = ['AMT_CREDIT', 'AMT_GOODS_PRICE', 'AMT_ANNUITY']
capping_values = {
    'AMT_CREDIT': 5_000_000,
    'AMT_GOODS_PRICE': 4_500_000,
    'AMT_ANNUITY': 300_000
}
for col in monetary_cols:
    print(f"\n{col}: Analyse des valeurs extrêmes")
    plt.figure(figsize=(10,4))
    sns.boxplot(x=app_train[col])
    plt.title(f'Boxplot: {col}')
    plt.show()
    high_values = app_train[app_train[col] > capping_values[col]]
    print(f"Nombre de dossiers avec {col} > {capping_values[col]} : {len(high_values)}")
    # Capping
    app_train.loc[app_train[col] > capping_values[col], col] = capping_values[col]
    print(f"Capping des {col} > {capping_values[col]} à {capping_values[col]} effectué.")

# 4. CNT_CHILDREN : traitement des valeurs élevées (supposons >10 comme suspects)
print("\nCNT_CHILDREN: Analyse des valeurs élevées (>10)")
suspect_children = app_train[app_train['CNT_CHILDREN'] > 10]
print(f"Nombre de lignes suspectes : {len(suspect_children)}")
print(suspect_children[['SK_ID_CURR', 'CNT_CHILDREN', 'TARGET']])
# Pour analyse, on peut soit exclure soit fixer à 10 pour ces cas
app_train.loc[app_train['CNT_CHILDREN'] > 10, 'CNT_CHILDREN'] = 10
print("Capping de CNT_CHILDREN > 10 à 10 effectué.")

