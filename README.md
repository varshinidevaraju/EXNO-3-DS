## EXNO-3-DS

# AIM :
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM :
```
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.
```

# FEATURE ENCODING :
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation :
  # 1. FUNCTION TRANSFORMATION :
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION :
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT :
```
Developed by : Varshini D
Reg.No : 212223230234
```
```
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
```
![Screenshot 2024-09-23 132355](https://github.com/user-attachments/assets/79e46010-04cc-46a2-8a3e-626c71d76a3d)
```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![Screenshot 2024-09-23 132427](https://github.com/user-attachments/assets/c7c85282-1ff6-4b9b-a755-b80b4b17e32d)
```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![Screenshot 2024-09-23 133258](https://github.com/user-attachments/assets/0fbff9d3-552b-47ef-be9e-ca8aec2a8c95)
```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![Screenshot 2024-09-23 133332](https://github.com/user-attachments/assets/bb0a14a1-882b-4bcc-8e2b-52be28df55d7)
```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
```
![Screenshot 2024-09-23 133720](https://github.com/user-attachments/assets/125f7f25-3c9d-4a7d-8282-462e07008514)
```
df2=pd.concat([df2,enc],axis=1)
df2
```
![Screenshot 2024-09-23 133747](https://github.com/user-attachments/assets/6efbfcb7-240a-40f3-a12e-0639e8cdd243)
```
pd.get_dummies(df2,columns=["nom_0"])
```
![Screenshot 2024-09-23 133815](https://github.com/user-attachments/assets/ad11eba6-1d9f-4803-a071-216df32452be)
```
pip install --upgrade category_encoders
```
![Screenshot 2024-09-23 133858](https://github.com/user-attachments/assets/a1aaaa99-5dec-41cf-8024-dfb2fcb00133)
```
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
```
![Screenshot 2024-09-23 134713](https://github.com/user-attachments/assets/b8b833e2-e8c5-42cb-9776-d1e9b677378e)
```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```
![Screenshot 2024-09-23 134805](https://github.com/user-attachments/assets/f7c1911a-3176-4789-8834-8eca5a6dbd26)
```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
![Screenshot 2024-09-23 134833](https://github.com/user-attachments/assets/3baedd6e-aa00-45af-ba5b-c168ab410894)
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
![Screenshot 2024-09-23 134939](https://github.com/user-attachments/assets/d7bd4ff0-a787-4143-86fd-67cbd29d6e6f)
```
df.skew()
```
![Screenshot 2024-09-23 135001](https://github.com/user-attachments/assets/7d16d2d9-bf16-4d05-b574-2755df702789)
```
np.log(df["Highly Positive Skew"])
```
![Screenshot 2024-09-23 135034](https://github.com/user-attachments/assets/21cbe886-cd95-42b4-a935-09fabbbf1c9b)
```
np.reciprocal(df["Moderate Positive Skew"])
```
![Screenshot 2024-09-23 135113](https://github.com/user-attachments/assets/79121993-5413-48c8-8445-5fafb8240f3a)
```
np.sqrt(df["Highly Positive Skew"])
```
![Screenshot 2024-09-23 135140](https://github.com/user-attachments/assets/30dc0bcb-11f1-465e-bfc9-204755c05891)
```
np.square(df["Highly Positive Skew"])
```
![Screenshot 2024-09-23 135235](https://github.com/user-attachments/assets/36cdb338-451b-4dbf-92ac-bf96b7efc6df)
```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![Screenshot 2024-09-23 135308](https://github.com/user-attachments/assets/5ab118d4-7d0a-4757-a5dc-9920bff57a43)
```
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
df
```
![Screenshot 2024-09-23 135400](https://github.com/user-attachments/assets/adc460ee-5a6e-453f-ba32-f060e66291c7)
```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![Screenshot 2024-09-23 135433](https://github.com/user-attachments/assets/3d9ba190-7543-4e92-bcf3-51cc9a62facc)
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![Screenshot 2024-09-23 145007](https://github.com/user-attachments/assets/5751d252-94c1-4d5b-a3ac-0cf8d1f20bf2)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![Screenshot 2024-09-23 145031](https://github.com/user-attachments/assets/2ab42544-f001-4b9c-9c26-ce7cfce35a2f)
```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```
![Screenshot 2024-09-23 145050](https://github.com/user-attachments/assets/c59f8d8f-acf5-45ed-9844-33a4786b5873)
```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
![Screenshot 2024-09-23 145100](https://github.com/user-attachments/assets/1f316a7a-cc82-48d2-99fa-2162a3fa8221)
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![Screenshot 2024-09-23 145108](https://github.com/user-attachments/assets/2f36d584-813c-4f64-be49-c5abc8e63e91)

# RESULT :
Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.
     

       
