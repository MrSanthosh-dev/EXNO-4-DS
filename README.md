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

# CODING AND OUTPUT:

```python
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data=pd.read_csv("/content/income(1) (1).csv",na_values=[ " ?"])
data
```
![Screenshot 2024-10-03 105005](https://github.com/user-attachments/assets/0d9200b4-6128-4b3b-98e3-170372362ef5)

```python

data.isnull().sum()
```
![Screenshot 2024-10-03 105135](https://github.com/user-attachments/assets/b05b9502-8a24-4bfb-aa06-13ef1ae5f64f)

```python

missing=data[data.isnull().any(axis=1)]
missing
```
![image](https://github.com/user-attachments/assets/e10b79af-1d96-40f8-a8a1-5146569fcbc5)

```python

data2=data.dropna(axis=0)
data2
```
![image](https://github.com/user-attachments/assets/8c5b6868-7bcb-49aa-a739-11123eab2c47)

```python
sal=data["SalStat"]

data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![image](https://github.com/user-attachments/assets/38b17801-5354-492b-b271-48c6cfa672c6)

```python
sal2=data2['SalStat']

dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![image](https://github.com/user-attachments/assets/dbb67bf2-8038-40d6-8ca6-af150e17bb3c)
```python
data2
```
![image](https://github.com/user-attachments/assets/f230751f-8c5e-4cad-8e59-282ff7ce9f59)
```python
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
![image](https://github.com/user-attachments/assets/88a5715a-ef1f-41b8-8203-9bdb0736b189)
```python

columns_list=list(new_data.columns)
print(columns_list)
```
![image](https://github.com/user-attachments/assets/dd4277a2-343a-40a6-b13d-45eca1607533)
```python


features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![image](https://github.com/user-attachments/assets/6eb6f6ca-8487-456a-b255-8b9290630298)
```python
y=new_data['SalStat'].values
print(y)
```
![image](https://github.com/user-attachments/assets/ed297a16-efa1-4069-a0cc-f41a9f7f6a8a)
```python

x=new_data[features].values
print(x)
```
![image](https://github.com/user-attachments/assets/bbf39638-0ecf-4b3a-b0cb-d7dd7b58e332)
```python

train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)

KNN_classifier=KNeighborsClassifier(n_neighbors = 5)

KNN_classifier.fit(train_x,train_y)
```
![image](https://github.com/user-attachments/assets/2a849b36-9799-4548-a84f-881b8234501d)
```python

prediction=KNN_classifier.predict(test_x)

confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
![image](https://github.com/user-attachments/assets/537cd93a-e292-4301-bb0f-92e824d0c9b9)
```python

accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
![image](https://github.com/user-attachments/assets/b4cba651-264b-42c7-b88e-28a6ef9d8325)
```python

print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```
![image](https://github.com/user-attachments/assets/17aaf2a3-b3e2-4698-a683-7c45a4dba5ea)
```python

data.shape
```
![image](https://github.com/user-attachments/assets/e3f983f2-af6b-47ca-a855-a87e76da9a55)
```python

import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target'  : [0,1,1,0,1]
}

df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]

selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)

selected_feature_indices=selector.get_support(indices=True)

selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
```python

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![image](https://github.com/user-attachments/assets/5dcb3c9a-f8bc-4deb-bf6e-ff6065cf579e)
```python

tips.time.unique()
```
![image](https://github.com/user-attachments/assets/5979a2ce-8285-4e98-a830-f905bb7249ad)
```python

contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
![image](https://github.com/user-attachments/assets/7f8e9aaf-7fbb-4a95-b774-96b1ad926846)
```python

chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```

![image](https://github.com/user-attachments/assets/d7107e5a-4af9-4d4c-95c4-04dad4d25e7d)
# RESULT:
Thus, Feature selection and Feature scaling has been used on thegiven dataset.
