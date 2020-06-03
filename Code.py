#!/usr/bin/env python
# coding: utf-8

# # Course project - IMDB movie rating
# 
# 
# 
# github link:  https://github.com/ynnnnnnn/Course-project---IMDB-movie-rating.git

# # 1. Data Management

# ## 1.1 Unzip and Merge dataset

# In[32]:


import tarfile
import os
def untar(fname, dirs):
    t = tarfile.open(fname)
    t.extractall(path = dirs) 

if __name__ == "__main__":
    untar("aclImdb_v1.tar.gz", ".")


# In[33]:


import pyprind
import pandas as pd
import os
pbar=pyprind.ProgBar(50000)
labels={'pos':1,'neg':0}
labeldataset={'test':'test','train':'train'}
df=pd.DataFrame()
for s in ('test','train'):
    for l in ('pos','neg'):
        path='./aclImdb/%s/%s'% (s,l)
        for file in os.listdir(path):
            with open(os.path.join(path,file),'r') as infile:
                txt=infile.read()
            df=df.append([[txt,labels[l],labeldataset[s]]],ignore_index=True)
            pbar.update()
df.columns=['review','sentiment','dataset']


# In[34]:


df.to_csv('./movie_data.csv',index=False)


# In[35]:


df=pd.read_csv('./movie_data.csv')
df.head(4)


# In[36]:


train = df.loc[df['dataset'].isin(['train'])]


# In[37]:


test = df.loc[df['dataset'].isin(['test'])]


# In[38]:


train.to_csv('./train_dataset.csv',index=False)
test.to_csv('./test_dataset.csv',index=False)
train=pd.read_csv('./train_dataset.csv')
test=pd.read_csv('./test_dataset.csv')


# In[39]:


train.head(4)


# In[40]:


test.head(4)


# ## 1.2 Preprocessing Data

# In[41]:


import re
from bs4 import BeautifulSoup

def review_to_wordlist(review):
    
    # removing HTML tags
    review_text = BeautifulSoup(review, "html.parser").get_text()
    # removing punctuations
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    # lowercase all words，and turn IMDB comments into word list.
    words = review_text.lower().split()
    # return words
    return words


# In[42]:


label = train['sentiment']
train_data = []
for i in range(len(train['review'])):
    train_data.append(' '.join(review_to_wordlist(train['review'][i])))
test_data = []
for i in range(len(test['review'])):
    test_data.append(' '.join(review_to_wordlist(test['review'][i])))


# In[43]:


print(train_data[0], '\n')
print(test_data[0])


# ## 1.3 Feature Extraction and Vectorization of text

# In[44]:


from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF

tfidf = TFIDF(min_df=2, 
           max_features=None,
           strip_accents='unicode',
           analyzer='word',
           token_pattern=r'\w{1,}',
           ngram_range=(1, 2),  # binary grammar model
            #ngram_range=(1, 3),  # Ternary grammar model
           use_idf=1,
           smooth_idf=1,
           sublinear_tf=1,
           stop_words = 'english') # Remove English stop words

# Combine training and test sets for TF-IDF vectorization
data_all = train_data + test_data
len_train = len(train_data)

tfidf.fit(data_all)
data_all = tfidf.transform(data_all)
# Restore to training set and testing set sections
train_x = data_all[:len_train]
test_x = data_all[len_train:]
print( 'Tf-idf processing finished.')


# # 2. Model Building
# ## 2.1 Naive Bayes Classification

# In[45]:


from sklearn.naive_bayes import MultinomialNB as MNB

model_NB = MNB()
model_NB.fit(train_x, label)
MNB(alpha=1.0, class_prior=None, fit_prior=True)

from sklearn.model_selection import cross_val_score
import numpy as np

print ("10-Fold cross validation: ", np.mean(cross_val_score(model_NB, train_x, label, cv=10, scoring='roc_auc')))


# ## 2.2 Logistic Regression

# In[46]:


from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import GridSearchCV
import warnings

warnings.filterwarnings("ignore")

# 设定grid search的参数
grid_values = {'C':[30]}  
# 设定打分为roc_auc
model_LR = GridSearchCV(LR(penalty = 'l2', dual = True, random_state = 0), grid_values, scoring = 'roc_auc', cv = 20)
model_LR.fit(train_x, label)
# 10折交叉验证
GridSearchCV(cv=20, estimator=LR(C=1.0, class_weight=None, dual=True,
             fit_intercept=True, intercept_scaling=1, penalty='l2', random_state=0, tol=0.0001), iid=True, n_jobs=1,
        param_grid={'C': [30]}, pre_dispatch='2*n_jobs', refit=True,
        scoring='roc_auc', verbose=0)
#输出结果
print (model_LR.cv_results_['mean_test_score'])
print (model_LR.cv_results_['std_test_score'])
print (model_LR.cv_results_['params'])


# ## 2.3 Random Forest

# In[47]:


from sklearn.ensemble import RandomForestClassifier
# Create the model with 100 trees
RF = RandomForestClassifier(n_estimators=100)
# Fit on training data
RF.fit(train_x, label)
score=cross_val_score(model_NB, train_x, label, cv=10, scoring='roc_auc')
np.mean(score)


# # 3. Outcome Evaluation

# In[48]:


#add a column ID to testing dataset to facilitate backtesting of the modeling results.
list_c=list(test.index)
test['id']=list_c
test.head(4)


# ## 3.1 Naive Bayes Classification

# In[49]:


test_predicted = np.array(model_NB.predict(test_x))
nb_output = []
nb_output = pd.DataFrame(data=test_predicted, columns=['sentiment_predict'])
nb_output['id'] = test['id']
nb_output = nb_output[['id', 'sentiment_predict']]
#merge the calssification result with the real sentiment classification. 
nb=pd.merge(test,nb_output)


# In[50]:


#Calculate accurcy 
correct=0
wrong=0
for i in nb.index:
    if nb.loc[i,'sentiment']==nb.loc[i,'sentiment_predict']:
        correct=correct+1
    else:
        wrong=wrong+1
print('correct=',correct)
print('wrong=',wrong)
print('accurcy=',correct/25000)


# ## 3.2 Logistic Regression

# In[51]:


#Predict in testing data using our fitted Logistic Regression model
test_predicted = np.array(model_LR.predict(test_x))
lr_output = pd.DataFrame(data=test_predicted, columns=['sentiment_predict'])
lr_output['id'] = test['id']
lr_output = lr_output[['id', 'sentiment_predict']]
#merge the calssification result with the real sentiment classification. 
lr=pd.merge(test,lr_output)


# In[52]:


#Calculate accurcy
correct1=0
wrong1=0
for i in lr.index:
    if lr.loc[i,'sentiment']==lr.loc[i,'sentiment_predict']:
        correct1=correct1+1
    else:
        wrong1=wrong1+1
print('correct=',correct1)
print('wrong=',wrong1)
print('accurcy=',correct1/25000)


# ## 3.3 Random Forest

# In[53]:


# Predict in testing data using our fitted Random Forest model above
rf_predictions = RF.predict(test_x)

rf_output = pd.DataFrame(data=rf_predictions, columns=['sentiment_predict'])
rf_output['id'] = test['id']
rf_output = rf_output[['id', 'sentiment_predict']]
#merge the calssification result with the real sentiment classification. 
rf=pd.merge(test,rf_output)


# In[55]:


#Calculate accurcy
correct2=0
wrong2=0
for i in rf.index:
    if rf.loc[i,'sentiment']==rf.loc[i,'sentiment_predict']:
        correct2=correct2+1
    else:
        wrong2=wrong2+1
print('correct=',correct2)
print('wrong=',wrong2)
print('accurcy=',correct2/25000)


# # 4. Conclusion

# In[63]:


conclusion=[]
conclusion = pd.DataFrame({'model':['Naive Bayes Classification','Logistic Regression','Random Forest'],
                      'AUC':[0.9499691520000001,0.96481306,0.9499691520000001],
                      'Accuracy':[0.85788,0.89048,0.84896]})
conclusion

