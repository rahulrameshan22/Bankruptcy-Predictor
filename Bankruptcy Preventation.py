#!/usr/bin/env python
# coding: utf-8

# # Problem Statement
# 
# Bankruptcy prediction is the art of predicting bankruptcy and various measures of financial distress of public firms.
# The problem statement is to develop a prediction model which will predict whether a company can go bankrupt or not.
# This will help the company to take appropriate decisions.We have have got certain paramertes which govern the prediction.
# This is a classification problem as the motive of the problem statement is to predict a binary solution whether
# the comapny goes bankrupt or not. 
# 
# 

# In[1]:


#Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# In[2]:


#Importing the data set
DF=pd.read_csv('bankruptcy-prevention.csv', delimiter=';')
DF


# In[3]:


#Lets drop the target column that is "class"
df=DF.drop([' class'], axis=1)
df


# # 1) Exploratory Data Analysis

# In[4]:


#Describing the data set
df.describe()


# In[5]:


df.columns.tolist()


# All the columns are required and are the parmaetrs which predict the classification.

# In[6]:


#Checking invalid records
df.isnull().sum()


# No Null values and there are no invalid records in the data set

# In[7]:


#Missing value detection and imputation
df.info()


# # Outliers
# 
# As the given data set has categorical values and nothing to be excluded.

# # Converting the values to the actual refered meaning for better undertanding

# In[8]:


categ= df.iloc[:,0:6].replace({0:'Low',0.5:'Medium',1:'High'})
categ


# In[9]:


#Correlation
df.corr()


# In[10]:


sns.heatmap(df.corr(), vmin = -1, vmax = 1, annot = True)


# # Data Visualization

# In[11]:


sns.countplot(x = ' class', data = DF, palette = 'hls')


# In[12]:


pd.crosstab(categ.industrial_risk,DF[' class']).plot(kind='bar')


# In[13]:


pd.crosstab(categ[' management_risk'],DF[' class']).plot(kind='bar')


# In[14]:


pd.crosstab(categ[' financial_flexibility'],DF[' class']).plot(kind='bar')


# In[15]:


pd.crosstab( categ[' operating_risk'],DF[' class']).plot(kind='bar')


# In[16]:


pd.crosstab(categ[' credibility'],DF[' class']).plot(kind='bar')


# In[17]:


pd.crosstab(categ[' competitiveness'],DF[' class']).plot(kind='bar')


# # Feature Engineering

# # Chi2 test

# In[18]:


from sklearn.feature_selection import chi2
x=DF.drop(columns=[' class'],axis=1)
y=DF[' class']


# In[19]:


chi_scores=chi2(x,y)


# In[20]:


chi_scores


# In[21]:


chi_values=pd.Series(chi_scores[0],index=x.columns)
chi_values.sort_values(ascending=False,inplace=True)
chi_values.plot.bar()


# Higher the chi value higher is the importance of the feature.
# 
# From the chi2 test we can conclude that competitiveness, financial flexibility, credibility are the features which has more importance in predicting the class
# 

# # Recursive Feature Elimination

# In[22]:


from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier


# In[23]:


x=DF.drop(columns=[' class'],axis=1)
y=DF[' class']


# In[24]:


rfe=RFE(estimator=DecisionTreeClassifier(),n_features_to_select=3)
rfe.fit(x,y)


# In[25]:


for i, col in zip(range(x.shape[1]), x.columns):
    print(f"{col} selected={rfe.support_[i]} rank={rfe.ranking_[i]}")


# Higher the rank, higher is the importance of the feature.
# 
# From the RFE test we can conclude that competitiveness, financial flexibility, credibility are the features which has more importance in predicting the class

# Hence comparing both the feature selection test we consider 'competitiveness, financial flexibility, credibility' has more importance in predicting the 'class'  and we drop other columns.

# # Model Building

# In[26]:


#Converting the class column to numerical
df1= DF.iloc[:,:7].replace({'bankruptcy':0,'non-bankruptcy':1})
df1


# In[27]:


#Considering only competitiveness, financial flexibility, credibility as it has more importance in predicting the 'class'and droppping other columns


# In[28]:


df=df1.drop(['industrial_risk',' management_risk',' operating_risk'],axis=1)
df


# # Splitting the Data into train and test

# In[29]:


from sklearn.model_selection import train_test_split


# In[30]:


x=df.drop([' class'],axis=1)
x


# In[31]:


y=df[[' class']]
y


# In[32]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.20,random_state=1)
print("Shape of X_train : ",x_train.shape)
print("Shape of X_test  : ",x_test.shape)
print("Shape of y_train : ",y_train.shape)
print("Shape of y_test  : ",y_test.shape)


# # Logistic Regression

# In[33]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, classification_report


# In[34]:


logistic_model = LogisticRegression()
logistic_model.fit(x_train,y_train)


# In[35]:


logistic_model.coef_


# In[36]:


#Train Accuracy
y_pred_train=logistic_model.predict(x_train)
accuracy_score(y_train,y_pred_train)


# In[37]:


#Test Accuracy
y_pred_test = logistic_model.predict(x_test)
accuracy_score(y_test,y_pred_test)


# In[38]:


print('Training set score : {:.2f}%'.format(logistic_model.score(x_train, y_train)*100))
print('Test set score     : {:.2f}%'.format(logistic_model.score(x_test, y_test)*100))


# # KNN Model

# In[39]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# In[40]:


num_folds = 10
kfold = KFold(n_splits=10)


# In[41]:


model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train,y_train)


# In[42]:


#Predicting on Train Data
preds_train = model.predict(x_train)
accuracy_score(y_train,preds_train)


# In[43]:


#Predicting on test data
preds_test = model.predict(x_test) # predicting on test data set 
pd.Series(preds_test).value_counts() # getting the count of each category


# In[44]:


# Accuracy
accuracy_score(y_test,preds_test)


# In[45]:


print('Training set score : {:.2f}%'.format(model.score(x_train, y_train)*100))
print('Test set score     : {:.2f}%'.format(model.score(x_test, y_test)*100))


# # Decision Tree

# In[46]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score
from sklearn import tree
from sklearn.tree import plot_tree


# In[47]:


model= DecisionTreeClassifier(criterion='entropy',max_depth=3)
model.fit(x_train,y_train)


# In[48]:


#Predicting on test data
preds = model.predict(x_test) # predicting on test data set 
pd.Series(preds).value_counts() #getting the count of each category


# In[49]:


#Train Accuracy 
model.score(x_train,y_train)


# In[50]:


#Test Accuracy
model.score(x_test,y_test)


# In[51]:


print('Training set score : {:.2f}%'.format(model.score(x_train, y_train)*100))
print('Test set score     : {:.2f}%'.format(model.score(x_test, y_test)*100))


# # SVM Model

# In[52]:


from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix


# kernel='linear'

# In[53]:


model_linear = SVC(kernel = "linear",random_state=40, gamma=0.1, C=1.0)
model_linear.fit(x_train,y_train)


# In[54]:


#Train Accuracy
pred_train_linear=model_linear.predict(x_train)
accuracy_score(y_train,pred_train_linear)


# In[55]:


#Test Accuracy
pred_test_linear = model_linear.predict(x_test)
accuracy_score(y_test,pred_test_linear)


# In[56]:


print('Training set score : {:.2f}%'.format(model_linear.score(x_train, y_train)*100))
print('Test set score     : {:.2f}%'.format(model_linear.score(x_test, y_test)*100))


# kernel='Poly' 

# In[57]:


model_poly = SVC(kernel = "poly",random_state=40, gamma=0.1, C=1.0)
model_poly.fit(x_train,y_train)


# In[58]:


#Train Accuracy
pred_train_poly=model_poly.predict(x_train)
accuracy_score(y_train,pred_train_poly)


# In[59]:


#Test Accuracy
pred_test_poly= model_poly.predict(x_test)
accuracy_score(y_test,pred_test_poly)


# In[60]:


print('Training set score : {:.2f}%'.format(model_poly.score(x_train, y_train)*100))
print('Test set score     : {:.2f}%'.format(model_poly.score(x_test, y_test)*100))


# Kernel='rbf'

# In[61]:


model_rbf= SVC(kernel ="rbf",random_state=40, gamma=0.1, C=1.0)
model_rbf.fit(x_train,y_train)


# In[62]:


#Train Accuracy
pred_train_rbf=model_rbf.predict(x_train)
accuracy_score(y_train,pred_train_rbf)


# In[63]:


#Test Accuracy
pred_test_rbf= model_rbf.predict(x_test)
accuracy_score(y_test,pred_test_rbf)


# In[64]:


print('Training set score : {:.2f}%'.format(model_rbf.score(x_train, y_train)*100))
print('Test set score     : {:.2f}%'.format(model_rbf.score(x_test, y_test)*100))


# # Naive Bayes

# In[65]:


from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB


# In[66]:


#Gausian NB
GNB = GaussianNB()
Naive_GNB = GNB.fit(x_train ,y_train)


# In[67]:


#Train Accuracy
pred_train=GNB.predict(x_train)
accuracy_score(y_train,pred_train)


# In[68]:


#Test Accuracy
pred_test=GNB.predict(x_test)
accuracy_score(y_test,pred_test)


# In[69]:


print('Training set score : {:.2f}%'.format(GNB.score(x_train, y_train)*100))
print('Test set score     : {:.2f}%'.format(GNB.score(x_test, y_test)*100))


# In[70]:


#MultinomialNB
MNB =  MultinomialNB()
Naive_MNB = MNB.fit(x_train ,y_train)


# In[71]:


#Train Accuracy
pred_train=MNB.predict(x_train)
accuracy_score(y_train,pred_train)


# In[72]:


#Test Accuracy
pred_test=MNB.predict(x_test)
accuracy_score(y_test,pred_test)


# In[73]:


print('Training set score : {:.2f}%'.format(MNB.score(x_train, y_train)*100))
print('Test set score     : {:.2f}%'.format(MNB.score(x_test, y_test)*100))


# # After verifying all the scores of various algorithms we select Decision Tree algorithm to build the model

# In[74]:


model= DecisionTreeClassifier(criterion='entropy',max_depth=3)
model.fit(x_train,y_train)


# In[75]:


#Predicting on test data
preds = model.predict(x_test) # predicting on test data set 
pd.Series(preds).value_counts() #getting the count of each category


# In[76]:


fn=[' financial_flexibility', ' credibility', ' competitiveness' ]
cn=['bankruptcy','non-bankruptcy'] 
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(model,
               feature_names = fn, 
               class_names=cn,
               filled = True);


# In[82]:


import pickle
pickle_out = open ("model.pkl","wb")
pickle.dump(model,pickle_out)
pickle_out.close()


# In[ ]:





# In[ ]:




