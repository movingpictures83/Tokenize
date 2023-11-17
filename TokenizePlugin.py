#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.model_selection as ms
from sklearn.model_selection import train_test_split, cross_val_score
import os
from itertools import cycle
from textblob import TextBlob
from nltk.stem import PorterStemmer
from textblob import Word
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing, linear_model, metrics, datasets, multiclass, svm
import seaborn as sns
import numpy.random as nr
import pandas as pd 
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.max_rows', 100)
# pd.set_option('display.max_colwidth', -1)


# In[3]:


class TokenizePlugin:
 def input(self, inputfile):
  self.complaints_df = pd.read_csv(inputfile)
 def run(self):
     pass
 def output(self, outputfile):
  self.complaints_df.info()
  self.complaints_df.describe()
  self.complaints_df.head()
  self.complaints_df.columns
  prod_col = 'Product'
  narr_col = 'Consumer complaint narrative'
  no_of_valid_obs = self.complaints_df[narr_col].notnull().sum()
  print(round((no_of_valid_obs/len(self.complaints_df)*100),1))
  print((self.complaints_df[prod_col]).value_counts())
  print(pd.notnull(self.complaints_df[narr_col]).value_counts())
  self.complaints_df = self.complaints_df[[prod_col, narr_col]]
  self.complaints_df = self.complaints_df[pd.notnull(self.complaints_df[narr_col])]
  self.complaints_df.head()
  self.complaints_df.shape
  dups = [ 'Credit reporting, credit repair services, or other personal consumer reports',
       'Credit card or prepaid card',
       'Money transfer, virtual currency, or money service',
       'Payday loan, title loan, or personal loan']
  self.complaints_df = self.complaints_df[~self.complaints_df[prod_col].isin(dups)]
  fig, ax = plt.subplots(figsize=(10,8))
  ax = sns.countplot(y=prod_col, 
                   data=self.complaints_df, 
                   order=self.complaints_df[prod_col].value_counts().index,
                   palette='magma'        
                  )
  ax.set_title('Complaints in each category',size=15)
  ax.set_xlabel('# of Complaints', size=14)
  plt.show
  tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern='\w{1,}', max_features=5000)    # default {max_df=1.0(float) so its proportion of word contain in all documents,
  tfidf_vect.fit(self.complaints_df[narr_col])
  Features = tfidf_vect.transform(self.complaints_df[narr_col])

  encoder = preprocessing.LabelEncoder()
  Labels1 = encoder.fit_transform(self.complaints_df[prod_col])

  train_x, valid_x,  train_y, valid_y = train_test_split(self.complaints_df[narr_col],self.complaints_df[prod_col])    # Default it will split 25 by 75% means 25% test case and 75% training cases

  encoder = preprocessing.LabelEncoder()
  train_y = encoder.fit_transform(train_y)
  valid_y = encoder.fit_transform(valid_y)
  print(train_y)

  tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern='\w{1,}', max_features=5000)    # default {max_df=1.0(float) so its proportion of word contain in all documents,
  tfidf_vect.fit(self.complaints_df[narr_col])
  print(tfidf_vect.stop_words)
  print(tfidf_vect.vocabulary_)
  xtrain_tfidf = tfidf_vect.transform(train_x)
  #print(xtrain_tfidf)
  xvalid_tfidf = tfidf_vect.transform(valid_x)

  model = linear_model.LogisticRegression().fit(xtrain_tfidf, train_y)
  model

  def accuracy(model):
    # checking accuracy
    accuracy = metrics.accuracy_score(model.predict(xvalid_tfidf),valid_y)
    print("Accuracy: ",accuracy)
    print(metrics.classification_report(valid_y,model.predict(xvalid_tfidf), 
                                        target_names=self.complaints_df[prod_col].unique()))

  accuracy(model)


  def heat_conf(model):
    # confusion matrix
    conf_mat = metrics.confusion_matrix(valid_y,model.predict(xvalid_tfidf))
#     print(conf_mat)
    # visualizing confusion matrix
    #category_id_df = Data[['product','category_id']].drop_duplicates().sort_values('category_id')
    #category_id_df
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(conf_mat, annot=True,fmt='d',cmap='Blues',
                xticklabels=self.complaints_df[prod_col].unique(),yticklabels= self.complaints_df[prod_col].unique())
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    
  heat_conf(model)

  from sklearn.linear_model import LogisticRegression
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.svm import LinearSVC
  from sklearn.naive_bayes import MultinomialNB
  from sklearn.model_selection import cross_val_score
  models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
  ]
  CV = 2
  cv_df = pd.DataFrame(index=range(CV * len(models)))
  entries = []

  for model in models:
   
    model = model.fit(xtrain_tfidf, train_y)
    accuracy(model)
    heat_conf(model)

  exit()   # FINISH HERE

