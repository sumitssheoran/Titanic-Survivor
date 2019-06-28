import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df1=pd.read_csv("titanic.csv")
df2=pd.read_csv("titanictest.csv")
#Adding new column "type" to keep train and test data separate
df1["type"]="train"
df2["type"]="test"
df=pd.concat([df1,df2],ignore_index=True)
#Dropping the unnecessary columns
df.drop(["PassengerId","Ticket"],axis=1,inplace = True)
df.info()
df.isnull().sum()
names=list(df["Name"])
for i in range(len(names)):
    names[i]=names[i].split(",")[1]
    names[i]=names[i].split(".")[0]
df["Name"]=names
df["Name"].value_counts()
df["Name"].unique()

df.loc[(df["Name"]==" the Countess")|
        (df["Name"]==" Sir")|
        (df["Name"]==" Mme")|
        (df["Name"]==" Jonkheer")|
        (df["Name"]==" Don")|
        (df["Name"]==" Dona"),"Name"]=0
df.loc[df["Name"]==" Mr","Name"]=1
df.loc[df["Name"]==" Miss","Name"]=2
df.loc[df["Name"]==" Mrs","Name"]=3
df.loc[df["Name"]==" Master","Name"]=4
df.loc[df["Name"]==" Dr","Name"]=5
df.loc[df["Name"]==" Rev","Name"]=6
df.loc[(df["Name"]==" Major")|
        (df["Name"]==" Mlle")|
        (df["Name"]==" Col")|
        (df["Name"]==" Capt"),"Name"]=7
df.loc[(df["Name"]==" Lady")|
        (df["Name"]==" Ms"),"Name"]=8  
        
#Visualize no. of survivors based on their names
sns.countplot(x="Name",hue="Survived",data=df)


df["Cabin"].fillna(value="Z",inplace=True)
df["Cabin"].value_counts()
#Assemble 'Cabin' based on their initial alphabet.
for i in range(len(df["Cabin"])):
    df["Cabin"][i]=df["Cabin"][i][0]
sns.countplot(x="Cabin",hue="Survived",data=df)
df["Cabin"].value_counts().plot(kind="bar")


from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df['Cabin'] = LE.fit_transform(df['Cabin'])

df.info()
df["Embarked"].value_counts()
#Using sklearn to fill missing values.
from sklearn.impute import SimpleImputer
si=SimpleImputer(missing_values=np.nan,strategy="most_frequent")
si=si.fit(df[["Embarked"]])
df[["Embarked"]]=si.transform(df[["Embarked"]])
df['Embarked'] = LE.fit_transform(df['Embarked'])
        

si=SimpleImputer(missing_values=np.nan,strategy="median")
si=si.fit(df[["Age"]])
df[["Age"]]=si.transform(df[["Age"]])
df["Age"]=df["Age"].astype(np.float64)


df["Sex"].value_counts()
df.loc[df["Sex"]=="male","Sex"]=0
df.loc[df["Sex"]=="female","Sex"]=1


df["Fare"].fillna(value=df["Fare"].mean(),inplace=True)

desc=df.describe() 
df.dtypes

#Separate train and test data based on the label "type"
X_train=df.loc[:890,df.columns!="Survived"] 
X_train.drop("type",axis=1,inplace=True) 
X_test=df.loc[891:1308,df.columns!="Survived"]
X_test.drop("type",axis=1,inplace=True)    
y_train=df.loc[:890,"Survived"]        
y_test=df.loc[891:1308,"Survived"]  


from sklearn.linear_model import LogisticRegression
model=LogisticRegression(class_weight="balanced"
                         ,penalty="l2")
model.fit(X_train,y_train)

y_pred=model.predict(X_train)

y_pred_prob=model.predict_proba(X_train)
accuracy=model.score(X_train,y_train)

from sklearn.metrics import roc_auc_score
score=roc_auc_score(y_train,y_pred_prob[:,1])

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_train,y_pred)
TP = cm[1, 1]
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]

(TP+TN)/ (TN+TP+FP+FN)# Accuracy
TP/(TP+FN)# Sensitivity
TN/(TN+FP)# Specificity
FP/(TN+FP)# FPR
TP/(TP+FP)# Precision        


from sklearn.metrics import classification_report
print(classification_report(y_train,y_pred))

from sklearn.metrics import roc_curve
fpr,tpr,thresholds=roc_curve(y_train,y_pred_prob[:,1])
len(fpr)
thresholds
plt.figure()
plt.plot(fpr,tpr,label="logistic regression(area=%0.4f)"%score)
plt.plot([0,1],[0,1],"r--")
plt.legend(loc=0)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristics")
plt.show()

# Recursive Feature Elimination 
from sklearn.feature_selection import RFE
model2=LogisticRegression()
rfe=RFE(model2,6)
rfe.fit(X_train,y_train)
print(rfe.support_)
print(rfe.ranking_)
col=list(X_train.columns)
rank=list(rfe.ranking_)
new_list=[]
for i in range(len(col)):
    if rank[i]==1:
        new_list.append(col[i])

x_new=pd.DataFrame()
for i in new_list:
    x_new[i]=X_train[i]
    
rfe=rfe.fit(x_new,y_train)  
y_pred_new=rfe.predict(x_new)   
new_accuracy=rfe.score(x_new,y_train)
y_new_pred_prob=rfe.predict_proba(x_new) 
new_roc_auc_score=roc_auc_score(y_train,y_new_pred_prob[:,1])    

#New model for test data
model=LogisticRegression(class_weight="balanced"
                         ,penalty="l2")
model.fit(X_train,y_train)
y_pred_test=model.predict(X_test)
y_pred_test=y_pred_test.astype(np.int16)
df3=df2["PassengerId"]
df3=pd.DataFrame(df3)

df3.set_index("PassengerId",inplace=True)
df3["Survived"]=y_pred_test
df3.to_csv(r'C:\Users\Admin\Desktop\ml practice\kaggle_titanic.csv')

    
    
    
    