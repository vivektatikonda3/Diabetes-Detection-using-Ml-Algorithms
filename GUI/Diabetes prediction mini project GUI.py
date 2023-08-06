#!/usr/bin/env python
# coding: utf-8

# # 1) Exploratory Data analysis

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle


# In[2]:


import warnings
warnings.simplefilter(action='ignore')


# In[3]:


df=pd.read_csv('diabetes.csv')


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


df.shape


# In[7]:


df.info()


# In[8]:


df.describe()


# In[9]:


df.describe([0.10,0.25,0.50,0.75,0.90,0.95,0.99]).T


# In[10]:


df.duplicated().sum()


# In[11]:


#unique values

df['Age'].unique()


# In[12]:


df['SkinThickness'].unique()


# In[13]:


df['Outcome'].value_counts()


# In[14]:


df['Outcome'].value_counts()*100/len(df)


# In[15]:


#Boxplot
#df[['Age']].boxplot()


# In[16]:


#The histogram of age variable
#df['Age'].hist(edgecolor='black')


# In[17]:


#df['Pregnancies'].hist(edgecolor='black')


# In[18]:


#df['Glucose'].hist(edgecolor='black')


# In[19]:


#df['Insulin'].hist(edgecolor='black')


# In[20]:


#diabetic=df[df['Outcome']==1]
#non_diabetic=df[df['Outcome']==0]


# In[21]:


''''plt.hist(diabetic['Age'],alpha=0.5,label='Diabetic')
plt.legend()
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Age with diabetic people")
plt.show()'''


# In[22]:


''''plt.hist(non_diabetic['Age'],alpha=0.5,label='Non Diabetic')
plt.legend()
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Age with non_diabetic people")
plt.show()'''


# In[23]:


max_age=df['Age'].max()
min_age=df['Age'].min()
print("Max age : ",max_age)
print("Min age : ",min_age)


# In[24]:


''''fig, ax = plt.subplots(4,2, figsize=(16,16))
sns.distplot(df.Age, bins = 20, ax=ax[0,0]) 
sns.distplot(df.Pregnancies, bins = 20, ax=ax[0,1]) 
sns.distplot(df.Glucose, bins = 20, ax=ax[1,0]) 
sns.distplot(df.BloodPressure, bins = 20, ax=ax[1,1]) 
sns.distplot(df.SkinThickness, bins = 20, ax=ax[2,0])
sns.distplot(df.Insulin, bins = 20, ax=ax[2,1])
sns.distplot(df.DiabetesPedigreeFunction, bins = 20, ax=ax[3,0]) 
sns.distplot(df.BMI, bins = 20, ax=ax[3,1])'''


# In[25]:


df.groupby('Outcome').agg({'Pregnancies':'mean'})


# In[26]:


df.groupby('Outcome').agg({'Age':'max'})


# In[27]:


df.groupby('Outcome').agg({'Age':'mean'})


# In[28]:


df.groupby('Outcome').agg({'Insulin':'mean'})


# In[29]:


df.groupby('Outcome').agg({'Insulin':'max'})


# In[30]:


df.groupby('Outcome').agg({'Glucose':'mean'})


# In[31]:


df.groupby('Outcome').agg({'Glucose':'max'})


# In[32]:


df.groupby('Outcome').agg({'BMI':'mean'})


# In[33]:


''''f,ax=plt.subplots(1,2,figsize=(18,8))
df['Outcome'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('target')
ax[0].set_ylabel('')
#sns.countplot('Outcome',data=df,ax=ax[1])
#ax[1].set_title('Outcome')
plt.show()'''


# In[34]:


df.corr()


# In[35]:


# Correlation matrix graph of the data set
f, ax = plt.subplots(figsize= [20,10])
sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax, cmap = "magma" )
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()


# # 2)Data Preprocessing

# # 2.1)Missing observation analysis

# In[36]:


df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)


# In[37]:


df.head()


# In[38]:


df.isnull().sum()


# In[39]:


import missingno as mso
#mso.bar(df)


# In[40]:


def median_target(var):   
    temp = df[df[var].notnull()]
    temp = temp[[var, 'Outcome']].groupby(['Outcome'])[[var]].median().reset_index()
    return temp
print(median_target('Age'))


# In[41]:


columns = df.columns
columns = columns.drop("Outcome")
for i in columns:
    median_target(i)
    df.loc[(df['Outcome'] == 0 ) & (df[i].isnull()), i] = median_target(i)[i][0]
    df.loc[(df['Outcome'] == 1 ) & (df[i].isnull()), i] = median_target(i)[i][1]


# In[42]:


df.head()


# In[43]:


df.isnull().sum()


# In[44]:


#mso.bar(df)


# # 2.2) Outlier observation analysis

# In[45]:


# In the data set, there were asked whether there were any outlier observations compared to the 25% and 75% quarters.
# It was found to be an outlier observation.
for feature in df:
    
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3-Q1
    lower = Q1- 1.5*IQR
    upper = Q3 + 1.5*IQR
    
    if df[(df[feature] > upper)].any(axis=None):
        print(feature,"yes")
    else:
        print(feature, "no")


# In[46]:


# The process of visualizing the Insulin variable with boxplot method was done. We find the outlier observations on the chart.
import seaborn as sns
#sns.boxplot(x = df["Insulin"]);


# In[47]:


#We conduct a stand alone observation review for the Insulin variable
#We suppress contradictory values
Q1 = df.Insulin.quantile(0.25)
Q3 = df.Insulin.quantile(0.75)
IQR = Q3-Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR
df.loc[df["Insulin"] > upper,"Insulin"] = upper


# In[48]:


import seaborn as sns
#sns.boxplot(x = df["Insulin"]);


# # 2.3) Local outlier factor(LOF)

# In[49]:


# We determine outliers between all variables with the LOF method
from sklearn.neighbors import LocalOutlierFactor
lof =LocalOutlierFactor(n_neighbors= 10)
lof.fit_predict(df)


# In[50]:


df_scores = lof.negative_outlier_factor_
np.sort(df_scores)[0:30]


# In[51]:


#We choose the threshold value according to lof scores
threshold = np.sort(df_scores)[7]
threshold


# In[52]:


#We delete those that are higher than the threshold
outlier = df_scores > threshold
df = df[outlier]


# In[53]:


df.shape


# In[54]:


df.head()


# In[55]:


import time


# In[56]:


x=df.iloc[:,0:8].values
y=df.iloc[:,-1].values

#X = df.drop('Outcome', axis=1)
#y = df['Outcome']


# In[57]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,stratify=y,test_size=0.3,random_state=0,shuffle=True)


# In[58]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


# # Naive Bayes

# In[59]:


from sklearn.naive_bayes import GaussianNB
clf=GaussianNB()
start_time=time.time()
clf.fit(x_train,y_train)
end_time=time.time()
y_pred=clf.predict(x_test)
y_pred

pickle_out = open("naive bayes.pkl", "wb") 
pickle.dump(clf, pickle_out) 
pickle_out.close()


# In[60]:


from sklearn.metrics import accuracy_score,confusion_matrix,recall_score,classification_report,roc_curve,auc,matthews_corrcoef,mean_squared_error
ac=accuracy_score(y_pred,y_test)
cm=confusion_matrix(y_pred,y_test)
sensitivity=recall_score(y_pred,y_test)
cr=classification_report(y_pred,y_test)
error_rate = (cm[0][1] + cm[1][0]) / float(cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
fpr,tpr,thresholds=roc_curve(y_pred,y_test)
auc_score=auc(fpr,tpr)
matt=matthews_corrcoef(y_pred,y_test)
mse=mean_squared_error(y_pred,y_test)

print("Accuracy score : \n",ac)
print("Confusion matrix : \n",cm)
print("Sensitivity : \n",sensitivity)
print("Error rate : \n",error_rate)
print("classification report :\n",cr)
print("auc score :\n",auc_score)
print("mathhews correlation coefficient : ",matt)
print("MSE : ",mse)


# In[61]:


plt.plot(fpr, tpr, lw=1, label='ROC (AUC = %0.2f)' % (auc_score))
plt.plot([0, 1], [0, 1], '--', color='gray', label='Random')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Naive Bayes')
plt.legend(loc="lower right")


# In[62]:




# In[ ]:





# In[63]:

# In[64]:


cm_time=end_time-start_time
cm_time


# # Logistic Regression

# In[65]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
s=time.time()
lr.fit(x_train,y_train)
e=time.time()
y_pred=lr.predict(x_test)
y_pred


# In[66]:


import pickle


# In[67]:


#Saving the Model
pickle_out = open("logisticRegr.pkl", "wb") 
pickle.dump(lr, pickle_out) 
pickle_out.close()


# In[68]:


from sklearn.metrics import accuracy_score,confusion_matrix,recall_score,classification_report,roc_curve,auc
ac=accuracy_score(y_pred,y_test)
cm=confusion_matrix(y_pred,y_test)
sensitivity=recall_score(y_pred,y_test)
cr=classification_report(y_pred,y_test)
error_rate = (cm[0][1] + cm[1][0]) / float(cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
fpr,tpr,thresholds=roc_curve(y_pred,y_test)
auc_score=auc(fpr,tpr)
matt=matthews_corrcoef(y_pred,y_test)
mse=mean_squared_error(y_pred,y_test)

print("Accuracy score : \n",ac)
print("Confusion matrix : \n",cm)
print("Sensitivity : \n",sensitivity)
print("Error rate : \n",error_rate)
print("classification report :\n",cr)
print("auc score :\n",auc_score)
print("mathhews correlation coefficient : ",matt)
print("MSE : ",mse)


# In[69]:


plt.plot(fpr, tpr, lw=1, label='ROC (AUC = %0.2f)' % (auc_score))
plt.plot([0, 1], [0, 1], '--', color='gray', label='Random')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Logistic Regression')
plt.legend(loc="lower right")


# In[70]:






# In[72]:


c=e-s
c


# # SVM

# In[73]:


from sklearn.svm import SVC
s=SVC(kernel='linear',C=1.0,probability=True)
st=time.time()
s.fit(x_train,y_train)
e=time.time()
y_pred=s.predict(x_test)
y_pred

pickle_out = open("svm.pkl", "wb") 
pickle.dump(s, pickle_out) 
pickle_out.close()


# In[74]:


from sklearn.metrics import accuracy_score,confusion_matrix,recall_score,classification_report,roc_curve,auc
ac=accuracy_score(y_pred,y_test)
cm=confusion_matrix(y_pred,y_test)
sensitivity=recall_score(y_pred,y_test)
cr=classification_report(y_pred,y_test)
error_rate = (cm[0][1] + cm[1][0]) / float(cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
fpr,tpr,thresholds=roc_curve(y_pred,y_test)
auc_score=auc(fpr,tpr)
matt=matthews_corrcoef(y_pred,y_test)
mse=mean_squared_error(y_pred,y_test)

print("Accuracy score : \n",ac)
print("Confusion matrix : \n",cm)
print("Sensitivity : \n",sensitivity)
print("Error rate : \n",error_rate)
print("classification report :\n",cr)
print("auc score :\n",auc_score)
print("mathhews correlation coefficient : ",matt)
print("MSE : ",mse)


# In[75]:


plt.plot(fpr, tpr, lw=1, label='ROC (AUC = %0.2f)' % (auc_score))
plt.plot([0, 1], [0, 1], '--', color='gray', label='Random')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for SVM')
plt.legend(loc="lower right")


# In[76]:


# In[78]:


c=e-st
c


# # Random Forest

# In[79]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=1)
s=time.time()
rf.fit(x_train,y_train)
e=time.time()
y_pred=rf.predict(x_test)

pickle_out = open("random forest.pkl", "wb") 
pickle.dump(rf, pickle_out) 
pickle_out.close()


# In[80]:


from sklearn.metrics import accuracy_score,confusion_matrix,recall_score,classification_report,roc_curve,auc
ac=accuracy_score(y_pred,y_test)
cm=confusion_matrix(y_pred,y_test)
sensitivity=recall_score(y_pred,y_test)
cr=classification_report(y_pred,y_test)
error_rate = (cm[0][1] + cm[1][0]) / float(cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
fpr,tpr,thresholds=roc_curve(y_pred,y_test)
auc_score=auc(fpr,tpr)
matt=matthews_corrcoef(y_pred,y_test)
mse=mean_squared_error(y_pred,y_test)

print("Accuracy score : \n",ac)
print("Confusion matrix : \n",cm)
print("Sensitivity : \n",sensitivity)
print("Error rate : \n",error_rate)
print("classification report :\n",cr)
print("auc score :\n",auc_score)
print("mathhews correlation coefficient : ",matt)
print("MSE : ",mse)


# In[81]:


plt.plot(fpr, tpr, lw=1, label='ROC (AUC = %0.2f)' % (auc_score))
plt.plot([0, 1], [0, 1], '--', color='gray', label='Random')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Random Forest')
plt.legend(loc="lower right")


# In[82]:


c=e-s
c




