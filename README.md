# Course-project---IMDB-movie-rating

# 1. Data Management

## 1.1 Unzip and Merge dataset
In this section, we first downloaded the compressed packet 'aclimdb_v1.tar.gz' from http://ai.stanford.edu/~amaas/data/sentiment/. The function 'untar' is defined to unzip the 'aclImdb_v1.tar.gz' file. The detailed function is shown in the code below：




```python
import tarfile
import os
def untar(fname, dirs):
    t = tarfile.open(fname)
    t.extractall(path = dirs) 

if __name__ == "__main__":
    untar("aclImdb_v1.tar.gz", ".")
```



After unzipped, we obtained an 'aclImdb' folder. The file structure is:
Under the aclImdb directory there are directories such as test and train, and under the train and test directories there are secondary subdirectories neg and pos respectively. Neg directory contains a large number of negative rating TXT files, pos contains a large number of positive rating TXT files.

We found that each movie review was in a separate TXT file, not merged together. So I loop through the unzipped ‘aclImdb’ folder, putting all the movie reviews and their sentiment classification in one dataframe. The detailed code is shown below:




```python
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
```

    0% [##############################] 100% | ETA: 00:00:00
    Total time elapsed: 00:08:12


As you can see from the load progress shown, it takes about six minutes due to a lot of file processing. Then we store the data set in a CSV file.


```python
df.to_csv('./movie_data.csv',index=False)
```

The data after collation is as follows:


```python
df=pd.read_csv('./movie_data.csv')
df.head(4)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>review</th>
      <th>sentiment</th>
      <th>dataset</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Based on an actual story, John Boorman shows t...</td>
      <td>1</td>
      <td>test</td>
    </tr>
    <tr>
      <th>1</th>
      <td>This is a gem. As a Film Four production - the...</td>
      <td>1</td>
      <td>test</td>
    </tr>
    <tr>
      <th>2</th>
      <td>I really like this show. It has drama, romance...</td>
      <td>1</td>
      <td>test</td>
    </tr>
    <tr>
      <th>3</th>
      <td>This is the best 3-D experience Disney has at ...</td>
      <td>1</td>
      <td>test</td>
    </tr>
  </tbody>
</table>
</div>



As you can see, in our data, the Positive Sentiment Classification is marked as 1 and the Negative sentiment Classification as 0. At the same time, we also recorded whether each data comes from training set or testing set.

According to the data source website, it provides a set of 25,000 highly polar movie reviews for training and 25,000
for testing. So the data is divided into training and testing based on dataset label.



```python
train = df.loc[df['dataset'].isin(['train'])]
```


```python
test = df.loc[df['dataset'].isin(['test'])]
```


```python
train.to_csv('./train_dataset.csv',index=False)
test.to_csv('./test_dataset.csv',index=False)
train=pd.read_csv('./train_dataset.csv')
test=pd.read_csv('./test_dataset.csv')
```

The data for training and testing are shown below：


```python
train.head(4)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>review</th>
      <th>sentiment</th>
      <th>dataset</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>For a movie that gets no respect there sure ar...</td>
      <td>1</td>
      <td>train</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bizarre horror movie filled with famous faces ...</td>
      <td>1</td>
      <td>train</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A solid, if unremarkable film. Matthau, as Ein...</td>
      <td>1</td>
      <td>train</td>
    </tr>
    <tr>
      <th>3</th>
      <td>It's a strange feeling to sit alone in a theat...</td>
      <td>1</td>
      <td>train</td>
    </tr>
  </tbody>
</table>
</div>




```python
test.head(4)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>review</th>
      <th>sentiment</th>
      <th>dataset</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Based on an actual story, John Boorman shows t...</td>
      <td>1</td>
      <td>test</td>
    </tr>
    <tr>
      <th>1</th>
      <td>This is a gem. As a Film Four production - the...</td>
      <td>1</td>
      <td>test</td>
    </tr>
    <tr>
      <th>2</th>
      <td>I really like this show. It has drama, romance...</td>
      <td>1</td>
      <td>test</td>
    </tr>
    <tr>
      <th>3</th>
      <td>This is the best 3-D experience Disney has at ...</td>
      <td>1</td>
      <td>test</td>
    </tr>
  </tbody>
</table>
</div>



## 1.2 Preprocessing Data
In this part of Data Management, we preprocessed the data obtained in the previous part. Preprocessing mainly includes removing HTML tags, removing punctuations and lowercase all words, so as to turn IMDB comments into word list.

Define a 'review_to_wordlist' function to preprocess the data.


```python
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
```


```python
label = train['sentiment']
train_data = []
for i in range(len(train['review'])):
    train_data.append(' '.join(review_to_wordlist(train['review'][i])))
test_data = []
for i in range(len(test['review'])):
    test_data.append(' '.join(review_to_wordlist(test['review'][i])))

```

View the pre-processed data:


```python
print(train_data[0], '\n')
print(test_data[0])
```

    for a movie that gets no respect there sure are a lot of memorable quotes listed for this gem imagine a movie where joe piscopo is actually funny maureen stapleton is a scene stealer the moroni character is an absolute scream watch for alan the skipper hale jr as a police sgt 
    
    based on an actual story john boorman shows the struggle of an american doctor whose husband and son were murdered and she was continually plagued with her loss a holiday to burma with her sister seemed like a good idea to get away from it all but when her passport was stolen in rangoon she could not leave the country with her sister and was forced to stay back until she could get i d papers from the american embassy to fill in a day before she could fly out she took a trip into the countryside with a tour guide i tried finding something in those stone statues but nothing stirred in me i was stone myself suddenly all hell broke loose and she was caught in a political revolt just when it looked like she had escaped and safely boarded a train she saw her tour guide get beaten and shot in a split second she decided to jump from the moving train and try to rescue him with no thought of herself continually her life was in danger here is a woman who demonstrated spontaneous selfless charity risking her life to save another patricia arquette is beautiful and not just to look at she has a beautiful heart this is an unforgettable story we are taught that suffering is the one promise that life always keeps


## 1.3 Feature Extraction and Vectorization of text

Processing natural language text and extract useful information from the given word, a sentence using machine learning and deep learning techniques requires the string/text needs to be converted into a set of real numbers (a vector).

Word Embeddings or Word vectorization is a methodology in NLP to map words or phrases from vocabulary to a corresponding vector of real numbers which used to find word predictions, word similarities and semantics.

Several common methods for Word Embedding are one-hot encoding, TF-IDF and Word2vec.

In this project, TF-IDF is tried to approach word embedding.


### 1.3.1 TF-IDF Algorithm

TF-IDF (term frequency-Inverse Document Frequency) is a common weighting technique used for information retrieval and data mining. It is often used to mine keywords in the text, and the algorithm is simple and efficient, which is often used by the industry for the initial text data cleansing.

Tf-idf is actually: TF * IDF. The main idea is: if a certain word or phrase appears frequently in an article (high TF) and rarely in other articles (high IDF), then this word or phrase is considered to have a good ability to distinguish categories and is suitable for classification.

In this project, the specific steps of TFIDF are as follows：

The first step is to calculate Term Frequency(TF):
TF= The number of times a term appears in a piece of review

The first step is to calculate the inverse document frequency:
IDF=log(the number of documents in the corpus/The number of documents containing the term)

The third step is to calculate TF-IDF: TF-IDF= TF * IDF

It can be seen that TF-IDF is directly proportional to the frequency of occurrence of a word in the document and inversely proportional to the frequency of occurrence of the word in the entire language. Therefore, the algorithm for automatic keyword extraction is very clear, which is to calculate the TF-IDF value of each word in the review, and then arrange in descending order, taking the first few words.

### 1.3.1 Application of TF-IDF Algorithm

Follow the steps above, we implement Feature Extraction and Vectorization of text on training data and testing data respectively.The process is as follows, and the binary grammar model is adopted


```python
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
```

    Tf-idf processing finished.


That's it for the data processing, so let's move on to the Model Building

# 2. Model Building
In this section, we tried three ways to do the Movie Review Sentiment Classification.
## 2.1 Naive Bayes Classification

The first method we try is naive Bayesian classification.Naive bayesian classification (NBC) is based on applying Bayes' theorem with strong (naïve) independence assumptions between the features.First, the given training set is used to learn the joint probability distribution from input to output, and then, based on the learned model, the input X is used to find the output Y with the maximum posterior probability.

The codes of Naive Bayes Classification model building and AUC calculation are as follows. we apply 10-Fold cross validation to training data.


```python
from sklearn.naive_bayes import MultinomialNB as MNB

model_NB = MNB()
model_NB.fit(train_x, label)
MNB(alpha=1.0, class_prior=None, fit_prior=True)

from sklearn.model_selection import cross_val_score
import numpy as np

print ("10-Fold cross validation: ", np.mean(cross_val_score(model_NB, train_x, label, cv=10, scoring='roc_auc')))
```

    10-Fold cross validation:  0.9499691520000001


As we can see, AUC of the model is 0.9499691520000001.

## 2.2 Logistic Regression
Logistic regression is the representative of discriminative model and naive Bayes is the representative of generative model.
Now let's do binary logistic regression do the Movie Review Sentiment Classification.


```python
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import GridSearchCV
import warnings

warnings.filterwarnings("ignore")

# 
grid_values = {'C':[30]}  
# 
model_LR = GridSearchCV(LR(penalty = 'l2', dual = True, random_state = 0), grid_values, scoring = 'roc_auc', cv = 20)
model_LR.fit(train_x, label)
# 
GridSearchCV(cv=20, estimator=LR(C=1.0, class_weight=None, dual=True,
             fit_intercept=True, intercept_scaling=1, penalty='l2', random_state=0, tol=0.0001), iid=True, n_jobs=1,
        param_grid={'C': [30]}, pre_dispatch='2*n_jobs', refit=True,
        scoring='roc_auc', verbose=0)
#
print (model_LR.cv_results_['mean_test_score'])
print (model_LR.cv_results_['std_test_score'])
print (model_LR.cv_results_['params'])
```

    [0.96481306]
    [0.00511874]
    [{'C': 30}]


As we can see, AUC of the logistic regression model is 0.96481306,which is better than Naive Bayes Classification.

## 2.3 Random Forest
As the name implies, a random forest is established in a random way. There are many decision trees in the forest, and there is no correlation between each decision tree in the random forest. After the forest is obtained, when a new input sample is entered, each decision tree in the forest is judged separately to see which category the sample belongs to (for the classification algorithm), and then to see which category is selected the most, so as to predict which category the sample belongs to.

In this part, a random forest model with 100 trees is created.


```python
from sklearn.ensemble import RandomForestClassifier
# Create the model with 100 trees
RF = RandomForestClassifier(n_estimators=100)
# Fit on training data
RF.fit(train_x, label)
score=cross_val_score(model_NB, train_x, label, cv=10, scoring='roc_auc')
np.mean(score)
```




    0.9499691520000001



As we can see, AUC of random forest model is 0.9499691520000001,which is smaller than the previous two.

After completing the three model buildings, we conducted the outcome evaluation next.

# 3. Outcome Evaluation
In this part, the accurcy of the model is obtained by classifying review with testing data and comparing with the actual classification. Firstly,add a column ID to testing dataset to facilitate backtesting of the modeling results.

The calculation of accurcy is: accurcy= (The number of correct classification)/(The number of reviews in testing data)


```python
#add a column ID to testing dataset to facilitate backtesting of the modeling results.
list_c=list(test.index)
test['id']=list_c
test.head(4)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>review</th>
      <th>sentiment</th>
      <th>dataset</th>
      <th>id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Based on an actual story, John Boorman shows t...</td>
      <td>1</td>
      <td>test</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>This is a gem. As a Film Four production - the...</td>
      <td>1</td>
      <td>test</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>I really like this show. It has drama, romance...</td>
      <td>1</td>
      <td>test</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>This is the best 3-D experience Disney has at ...</td>
      <td>1</td>
      <td>test</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



## 3.1 Naive Bayes Classification
Predict in testing data using our fitted Naive Bayes Classification model above and merge the calssification result with the real sentiment classification. 


```python
test_predicted = np.array(model_NB.predict(test_x))
nb_output = []
nb_output = pd.DataFrame(data=test_predicted, columns=['sentiment_predict'])
nb_output['id'] = test['id']
nb_output = nb_output[['id', 'sentiment_predict']]
#merge the calssification result with the real sentiment classification. 
nb=pd.merge(test,nb_output)
```

Calculate accurcy as follow:


```python
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
```

    correct= 21447
    wrong= 3553
    accurcy= 0.85788


As we can see from the result: accurcy= 0.85788, which means the accurcy of the Naive Bayes Classification in the testing data is 85.788%

## 3.2 Logistic Regression
Predict in testing data using our fitted Logistic Regression model above and merge the calssification result with the real sentiment classification. 


```python
#Predict in testing data using our fitted Logistic Regression model
test_predicted = np.array(model_LR.predict(test_x))
lr_output = pd.DataFrame(data=test_predicted, columns=['sentiment_predict'])
lr_output['id'] = test['id']
lr_output = lr_output[['id', 'sentiment_predict']]
#merge the calssification result with the real sentiment classification. 
lr=pd.merge(test,lr_output)
```

Calculate accurcy as follow:


```python
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
```

    correct= 22262
    wrong= 2738
    accurcy= 0.89048


As we can see from the result: accurcy= 0.89048, which means the accurcy of the Naive Bayes Classification in the testing data is 89.048%

## 3.3 Random Forest
Predict in testing data using our fitted Random Forest model above and merge the calssification result with the real sentiment classification. 


```python
# Predict in testing data using our fitted Random Forest model above
rf_predictions = RF.predict(test_x)

rf_output = pd.DataFrame(data=rf_predictions, columns=['sentiment_predict'])
rf_output['id'] = test['id']
rf_output = rf_output[['id', 'sentiment_predict']]
#merge the calssification result with the real sentiment classification. 
rf=pd.merge(test,rf_output)
```

Calculate accurcy as follow:


```python
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
```

    correct= 21224
    wrong= 3776
    accurcy= 0.84896


As we can see from the result: accurcy= 0.84896, which means the accurcy of the Naive Bayes Classification in the testing data is 84.896%.

# 4. Conclusion

By comparing the three Classification models together, it can be seen that Logistic Regression is the best performance for training data and testing data of this project, Naive Bayes Classification comes second and Random Forest comes worst. However, there is little difference in the AUC and accuracy of the three models, and the overall performances are all good.

To make it more intuitive, the performance chart is  as follow:


```python
conclusion=[]
conclusion = pd.DataFrame({'model':['Naive Bayes Classification','Logistic Regression','Random Forest'],
                      'AUC':[0.9499691520000001,0.96481306,0.9499691520000001],
                      'Accuracy':[0.85788,0.89048,0.84896]})
conclusion
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>AUC</th>
      <th>Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Naive Bayes Classification</td>
      <td>0.949969</td>
      <td>0.85788</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Logistic Regression</td>
      <td>0.964813</td>
      <td>0.89048</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Random Forest</td>
      <td>0.949969</td>
      <td>0.84896</td>
    </tr>
  </tbody>
</table>
</div>



In this project, the Feature Extraction and Vectorization of text method is TF-IDF. The advantage of TF-IDF is that it is simple, fast, and easy to understand.However, the disadvantage is that sometimes the importance of a woed in the article is not comprehensive enough to be measured by word frequency, sometimes important words may not appear enough, and this kind of calculation cannot reflect the location information and the importance of the word in the context. 

If we want to represent the contextual structure of a word, the Word2VEc algorithm may be tried.
