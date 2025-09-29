## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
NAME  : NAVEEN KUMAR P
REG NO: 212224240102
```
```python

 import pandas as pd
 df=pd.read_csv("Encoding Data.csv")
 df

```

<img width="1265" height="466" alt="image" src="https://github.com/user-attachments/assets/74c4442c-add6-4406-bef0-180b6bf58df1" />

```python

from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])

```
<img width="1015" height="236" alt="image" src="https://github.com/user-attachments/assets/56410681-0465-4d44-b80f-b55bb0ba3499" />

```python

df['bo2']=e1.fit_transform(df[["ord_2"]])
df

```
<img width="1221" height="469" alt="image" src="https://github.com/user-attachments/assets/f04e7a6a-dd2a-4335-96a7-8476cf065238" />

```python

 le=LabelEncoder()
 dfc=df.copy()
 dfc['ord_2']=le.fit_transform(dfc['ord_2'])
 dfc

```
<img width="1148" height="442" alt="image" src="https://github.com/user-attachments/assets/3102c6e0-a48d-45a1-b98d-9db44fa8b7dd" />

```python

from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2=pd.concat([df2,enc],axis=1)
df2

```
<img width="1196" height="463" alt="image" src="https://github.com/user-attachments/assets/457932f8-c3f9-4e93-86a7-abfb29b55b1a" />

```python

pd.get_dummies(df2,columns=["nom_0"])

```
<img width="1152" height="429" alt="image" src="https://github.com/user-attachments/assets/b7cfe2d3-cef9-4b43-a6db-57b459699931" />

```python

pip install --upgrade category_encoders

```
<img width="1618" height="662" alt="image" src="https://github.com/user-attachments/assets/f8d89732-2cfc-403c-bf6e-6dbe23c792dd" />

```python

from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
dfb=pd.concat([df,nd],axis=1)
dfb

```
<img width="1361" height="400" alt="image" src="https://github.com/user-attachments/assets/5ff86268-f65d-4194-b9e9-a835f0567d92" />

```python

 from category_encoders import TargetEncoder
 te=TargetEncoder()
 CC=df.copy()
 new=te.fit_transform(X=CC["City"],y=CC["Target"])
 CC=pd.concat([CC,new],axis=1)
 CC

```
<img width="1317" height="392" alt="image" src="https://github.com/user-attachments/assets/b5eea4e5-c606-4b9b-8b2e-eb1107edb3de" />

```python

import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df

```
<img width="1432" height="443" alt="image" src="https://github.com/user-attachments/assets/70848202-53a3-4a76-8653-05b5343bcbbc" />

```python

 df.skew()

```
<img width="1231" height="242" alt="image" src="https://github.com/user-attachments/assets/ff324e13-3132-42e3-a59d-4366fa0cde40" />

```python

 np.log(df["Highly Positive Skew"])

```
<img width="1160" height="512" alt="image" src="https://github.com/user-attachments/assets/7124c7ab-a92e-468e-b326-48b5c2340186" />

```python

np.reciprocal(df["Moderate Positive Skew"])

```
<img width="1046" height="529" alt="image" src="https://github.com/user-attachments/assets/2a826e81-ff7a-4b92-ac87-0e47658c909b" />

```python

np.sqrt(df["Highly Positive Skew"])

```
<img width="1234" height="518" alt="image" src="https://github.com/user-attachments/assets/e45cb017-4aec-4553-93cb-986e9ab9d64b" />

```python

 np.square(df["Highly Positive Skew"])

```
<img width="1184" height="514" alt="image" src="https://github.com/user-attachments/assets/99660015-36ab-40fe-b10c-4f892bf29cf9" />

```python

 df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
 df

```
<img width="1345" height="483" alt="image" src="https://github.com/user-attachments/assets/6e819468-5026-4969-99dd-d2c5700113c9" />

```python

df.skew()

```
<img width="890" height="280" alt="image" src="https://github.com/user-attachments/assets/ce0fea55-ffd4-4164-ba4d-14f45a3acc26" />

```python

 df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
 df.skew()

```
<img width="1026" height="298" alt="image" src="https://github.com/user-attachments/assets/36efff4c-2b52-45a0-b612-3c3ffa7e27bf" />

```python

 from sklearn.preprocessing import QuantileTransformer
 qt=QuantileTransformer(output_distribution='normal')
 df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
 df

```
<img width="1722" height="471" alt="image" src="https://github.com/user-attachments/assets/321011b1-2a1b-4b63-be27-3d4197bfef6e" />

```python

import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()

```
<img width="866" height="508" alt="image" src="https://github.com/user-attachments/assets/c6e236ed-f567-4be8-9455-778be36fd87f" />

```python

sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()

```
<img width="852" height="509" alt="image" src="https://github.com/user-attachments/assets/12052b6e-00a4-4c3d-b974-ba5bf7c3237b" />

```python

from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()

```
<img width="795" height="506" alt="image" src="https://github.com/user-attachments/assets/6b5b0c8c-8c92-4f2b-af4d-7ddf8a15cc99" />

```python

df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()

```
<img width="822" height="501" alt="image" src="https://github.com/user-attachments/assets/686b9777-5684-4b7b-adf9-522bf3f14c06" />


```python

dt=pd.read_csv("titanic_dataset.csv")
dt
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45') 
plt.show()

```
<img width="873" height="518" alt="image" src="https://github.com/user-attachments/assets/d3a9ddc0-2bed-40f8-88cb-77e7022de824" />

```python

 sm.qqplot(df["Highly Negative Skew_1"],line='45')
 plt.show()

```
<img width="770" height="498" alt="image" src="https://github.com/user-attachments/assets/78e7b658-bc3b-4caa-ba94-b26dec47012b" />

       # INCLUDE YOUR CODING AND OUTPUT SCREENSHOTS HERE
# RESULT:
  Thus the given data, Feature Encoding, Transformation process and save the data to a file
 was performed successfully

       
