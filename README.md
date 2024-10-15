# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:

   STEP 1:Read the given Data.
   
   STEP 2:Clean the Data Set using Data Cleaning Process.
   
   STEP 3:Apply Feature Scaling for the feature in the data set.
   
   STEP 4:Apply Feature Selection for the feature in the data set.
   
   STEP 5:Save the data to the file.

# FEATURE SCALING:

  1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
  
  2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
  
  3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
  
  4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
  1.Filter Method
  
  2.Wrapper Method
  
  3.Embedded Method

# CODING AND OUTPUT :
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("bmi.csv")
df
```
![image](https://github.com/user-attachments/assets/36850ecf-c192-4a2f-ae59-ecb160d79ddb)
```
df.head()
```
![image](https://github.com/user-attachments/assets/36d919fa-b7b8-40ce-b715-0f687f785d9f)
```
df.dropna()
```
![image](https://github.com/user-attachments/assets/3782e221-6224-41e1-9cd7-b77c1fb5723e)
```
max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals
```
![image](https://github.com/user-attachments/assets/bbd3cdc0-6419-424e-adcb-455922286146)
```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/ff5eb943-f043-4aeb-83ad-bfc274e6a7fb)

```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/88e3cac1-6749-4b28-9303-bc216dc30df2)
```
from sklearn.preprocessing import Normalizer
scale=Normalizer()
df[['Height','Weight']]=scale.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/0f2b90ec-5044-479d-a214-2d26ae6b8933)
```
from sklearn.preprocessing import MaxAbsScaler
scalen=MaxAbsScaler()
df[['Height','Weight']]=scalen.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/7c981f2e-f2b4-41ad-9ad2-3f2a2fbd5e23)
```
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/99ff1188-49b9-4b3f-813c-b1b83899dcac)
```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
data=pd.read_csv('/content/income(1) (1).csv',na_values=[" ?"])
data
```
![image](https://github.com/user-attachments/assets/6c8d6191-122d-4a65-af81-b65a6cbd6f1c)
```
data.isnull().sum()
```
![image](https://github.com/user-attachments/assets/1421eec2-1a21-40a2-9eb6-8f9d2354e5ac)
```
missing=data[data.isnull().any(axis=1)]
missing
```
![image](https://github.com/user-attachments/assets/a2362763-98cb-4cde-a3e3-bdbc7c0600ab)
```
data2=data.dropna(axis=0)
data2
```
![image](https://github.com/user-attachments/assets/26749dcc-5fb1-48dd-9a26-722e3b1dc55d)
```
sal=data['SalStat']
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![image](https://github.com/user-attachments/assets/47203019-1357-451e-ac76-f2f51f00bef8)
```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![image](https://github.com/user-attachments/assets/6f1e210e-1a39-4a5a-a50b-6743f4377513)
```
new_data=pd.get_dummies(data2,drop_first=True)
new_data
```
![image](https://github.com/user-attachments/assets/ed8e168d-efc3-4cd9-9a7e-75cb7f353239)
```
columns_list=list(new_data.columns)
print(columns_list)
```
![image](https://github.com/user-attachments/assets/3f3944a9-65d0-4d4e-b984-088771170b03)
```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![image](https://github.com/user-attachments/assets/1ce962a8-9ca9-44c8-92ea-62a0ce9cc837)
```
y=new_data['SalStat'].values
print(y)
```
![image](https://github.com/user-attachments/assets/7f600480-ad4f-4b31-940c-0850591ce52b)
```
x=new_data[features].values
x
```
![image](https://github.com/user-attachments/assets/aa7d9dba-f6e4-450e-bb7f-aaf538e853dc)
```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors=5)
KNN_classifier.fit(train_x,train_y)
```
![image](https://github.com/user-attachments/assets/46c85ddf-1283-4e27-86dd-cf5eae1e1ab5)
```
prediction=KNN_classifier.predict(test_x)
confusionMmatrix=confusion_matrix(test_y,prediction)
print(confusionMmatrix)
```
![image](https://github.com/user-attachments/assets/c2a96e40-d9a2-4d92-807b-137408fd6e4e)
```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
![image](https://github.com/user-attachments/assets/002d204a-9732-4620-9cdf-df2db4d623c0)
```
print('Misclassified samples: %d' % (test_y != prediction).sum())
```
![image](https://github.com/user-attachments/assets/840244e4-93e8-46fc-b875-fb64db28ba01)
```
data.shape
```
![image](https://github.com/user-attachments/assets/73b7a49a-6fb1-4325-b981-c6fe877f31d8)
```
import pandas as pd
from sklearn.feature_selection import SelectKBest,mutual_info_classif,f_classif
data={
    'Feature1':[1,2,3,4,5],
    'Feature2':['A','B','C','A','B'],
    'Feature3':[0,1,1,0,1],
    'Target':[0,1,1,0,1]
}
df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df['Target']
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![image](https://github.com/user-attachments/assets/a61dadcf-1b40-4330-9653-9549bc7f64f7)
```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![image](https://github.com/user-attachments/assets/dda1ae41-df37-48d7-9a2f-f17d53286638)
```
contigency_table=pd.crosstab(tips['sex'],tips['time'])
print(contigency_table)
```
![image](https://github.com/user-attachments/assets/b5ff3adb-6a8a-4526-b725-712bf563da63)
```
chi2,p, _, _ =chi2_contingency(contigency_table)
print(f"Chi-Square Statistic: {chi2}")
print(f"P-value: {p}")
```
![image](https://github.com/user-attachments/assets/7cb119c5-6eff-4318-8093-aff02d79b216)

# RESULT:

Thus perform Feature Scaling and Feature Selection process and save the data to a file successfully.
