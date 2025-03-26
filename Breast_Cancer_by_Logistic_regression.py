import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,roc_auc_score,roc_curve,confusion_matrix
from sklearn.feature_selection import RFE
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import uniform
# load the dataset
data=load_breast_cancer()
x=data.data
y=data.target
col=data.feature_names
df=pd.DataFrame(x,columns=col)
df['target']=y
#information about the data
print("Information about the Data : \n",df.info())
print("Feature Names : ",col)
#EDA
print(df.head())
print(df.describe())
print("Class distriution : ",df["target"].value_counts())
#correlation matrix to check multicorrelation
corr_matrix=df.drop(columns=['target']).corr()
plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix,cmap='coolwarm',annot=False)
plt.title("Feature Correlation Heatmap")
plt.show()
#eliminate feeature with high collinearity
threshold=0.9
upper=corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(bool))
to_drop=[col for col in upper.columns if any(upper[col]>threshold)]
df.drop(columns=to_drop,inplace=True)
print("Column droped due to the correlation Matrix : ",to_drop)
# variation inflation factor
def remove_high_rif(df,thresh):
    x=df.copy()
    droped_feature=[]
    while True:
        vif_data=pd.DataFrame()
        vif_data['feature']=x.columns
        vif_data["vif"]=[variance_inflation_factor(x.values,i) for i in range(x.shape[1])]
        max_vif=vif_data['vif'].max()
        if max_vif<thresh:
            break
        feature_to_remove=vif_data.sort_values("vif",ascending=False).iloc[0]["feature"]
        print(f"Remove {feature_to_remove} with VIF = {max_vif:.2f}")
        x.drop(columns=[feature_to_remove],inplace=True)
        droped_feature.append(feature_to_remove)
        print("Feature Remaining after VIF : ",list(x.columns))
        return x,droped_feature
vif_df,removed_feature=remove_high_rif(df.drop(columns=["target"]),10)
print("Remaining feature after VIF : ",list(vif_df.columns))
# feature scaling
scaler=StandardScaler()
X_scaled=scaler.fit_transform(vif_df)
# Feature Engineering
poly=PolynomialFeatures(degree=2,interaction_only=True,include_bias=False)
x_poly=poly.fit_transform(X_scaled)
# Recursive Feature Elimination (RFE)
base_model=LogisticRegression(solver="liblinear")
rfe=RFE(base_model,n_features_to_select=15)
x_rfe=rfe.fit_transform(x_poly,y)
#Train Test split
x_train,x_test,y_train,y_test=train_test_split(x_rfe,y,test_size=0.2,random_state=42,stratify=y)# stratify mean to maintain balance class distribution in my target variable
#Hyper parameter tuning
tuned_grid={'C':[10**-4,10**-2,10**0,10**2,10**4]}
model_grid=GridSearchCV(LogisticRegression(solver='liblinear'),tuned_grid,scoring='f1',cv=5)
model_grid.fit(x_train,y_train)
tuned_random={'C':uniform(0.00001,10000)}
model_random=RandomizedSearchCV(LogisticRegression(solver='liblinear'),tuned_random,scoring='f1',cv=5,random_state=42,n_iter=20)
model_random.fit(x_train,y_train)
#find beat c from Grid and Randomm CV
best_c_grid=model_grid.best_params_['C']
best_c_random=model_random.best_params_['C']
best_score_grid=model_grid.best_score_
best_score_random=model_random.best_score_
if best_score_random>best_score_grid:
    best_c=best_c_random
else:
    best_c=best_c_grid
print("Best Hyperparameter(C) : ",best_c)
final_model=LogisticRegression(C=best_c,solver='liblinear')
final_model.fit(x_train,y_train)
# mode evaluation
y_pred = final_model.predict(x_test)
y_proba = final_model.predict_proba(x_test)[:, 1]
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))
# confusion matrix
cm=confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,fmt='.2f',cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_proba):.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()