import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
#load the Boston dataset
boston=fetch_openml(name="boston",version=1,as_frame=True)
df=boston.frame
#Explore the data
print("\nSample data : \n",df.head())
print("\nFeature Names : \n",boston.feature_names)
print("\Trget Name : \n",boston.target_names)
print("\nInformation about the DataSet :- \n")
print(df.info())
print("\n",boston.DESCR)
#statistical summery
print("statistical analysis :- \n",df.describe) 
#plot the correlation matrix
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(),annot=True,cmap="coolwarm",fmt=".2f")
plt.title("Feature Correlation Map")
plt.show()
#prepare the data for training 
x=df.drop("MEDV",axis=1)
y=df["MEDV"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print("shape of x_train data : ", x_train.shape)
print("shape of y_train data : ", y_train.shape)
print("shape of x_test data : ", x_test.shape)
print("shape of y_test data : ", y_test.shape)
#Feature scaling
scale=StandardScaler()
x_train=scale.fit_transform(x_train)
x_test=scale.transform(x_test)
print("Example Scaled Training Data : ",x_train[1])
#fit the linear model
model=LinearRegression()
model.fit(x_train,y_train)
#mak prediction by trained model
y_train_pred=model.predict(x_train)
y_test_pred=model.predict(x_test)
#evaluate the model 
def evaluation(actual,predicted):
    mae=mean_absolute_error(actual,predicted)
    mse=mean_squared_error(actual,predicted)
    rmse=np.sqrt(mse)
    r2=r2_score(actual,predicted)
    print(f"Mean absolute error : {mae:.2f}")
    print(f"Mean square error : {mse:.2f}")
    print(f"Root Mean square error : {rmse:.2f}")
    print(f"R2 score : {r2:.2f}")
print("\nTraining set performance :-\n")
evaluation(y_train,y_train_pred)
print("\nTest set performance :-\n")
evaluation(y_test,y_test_pred)
# train loss V/S test loss
plt.figure(figsize=(10,8))
plt.scatter(y_train,y_train_pred,alpha=0.6,color='r',label="Train")
plt.scatter(y_test,y_test_pred,alpha=0.6,color='g',label="Test")
plt.title("Train test loss")
plt.xlabel("Actual Value")
plt.ylabel("Predicted Value")
plt.legend() 
plt.show()
#optimization
ridge=Ridge(alpha=0.001)
ridge.fit(x_train,y_train)
print("\nOptimization Model Performance :-\n")
evaluation(y_test,ridge.predict(x_test))
                              