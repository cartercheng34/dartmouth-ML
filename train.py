import pandas as pd
import numpy as np
import nltk
import heapq
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest

from scipy.sparse import csr_matrix
import scipy as sp
from sklearn import svm
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

from nltk.corpus import stopwords  
from nltk.tokenize import word_tokenize 
from nltk.tokenize import TweetTokenizer


# generate simple classifier whether this review is 5 star
def AwesomeConvert(x):
    if(x > 4.5):
        return 1
    else:
        return 0



def pre_process(text):
    
    # lowercase
    if(pd.isnull(text)):
        return '0'

    text=text.lower()
    
    #remove tags
    text=re.sub("","",text)
    
    # remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ",text)
    
    return text


# read training data

df = pd.read_csv('./datasets/Train.csv')

# convert overall score to 1 or 0
# df['overall'] = df['overall'].apply(AwesomeConvert)

print("preprocessing...")


"""
uncomment below if require upsampling or downsampling
"""
# df_majority = df[df.overall==1]
# df_minority = df[df.overall==0]

# df_majority_downsampled = resample(df_majority, 
#                                  replace=False,    
#                                  n_samples=31771,
#                                  random_state=123) 

# df_downsampled = pd.concat([df_majority_downsampled, df_minority])
# print(df_downsampled.overall.value_counts())

# df_minority_upsampled = resample(df_minority, 
#                                  replace=True,     
#                                  n_samples=79327,    
#                                  random_state=123) 
# df_upsampled = pd.concat([df_majority, df_minority_upsampled])

#split data

helpful = df['helpful'] #number
reviewT = df['reviewText'] #string
summary = df['summary'] #string
price = df['price'] #number
overall = df['overall']
categories = df['categories']
rootG = df['root-genre'] #string
songs = df['songs'] #number
rank = df['salesRank'] #number



# print(df['overall'].value_counts(), '\n')

# counting number of 4,3,2,1 star review respectively
Num_4 = df['overall'].value_counts()[4]
Num_3 = df['overall'].value_counts()[3]
Num_2 = df['overall'].value_counts()[2]
Num_1 = df['overall'].value_counts()[1]

# average <= 4, 3.14
average = (4*Num_4 + 3*Num_3 + 2*Num_2 + 1*Num_1) / (Num_4 + Num_3 + Num_2 + Num_1)

# 0.73
ratio = (4.5-average)/(5-average)

# convert overall score to 1 or 0
Y = df['overall'].apply(AwesomeConvert)


"""
uncomment below if require upsampling or downsampling
"""

#label 
# reviewT = df_downsampled.reviewText
# summary = df_downsampled.summary
# price = df_downsampled.price
# rank = df_downsampled.salesRank
# Y = df_downsampled.overall
# reviewT = df_upsampled.reviewText
# summary = df_upsampled.summary
# price = df_upsampled.price
# rank = df_upsampled.salesRank
# Y = df_upsampled.overall

#tokenize

stop_words = set(stopwords.words('english'))


cv=CountVectorizer() 

# remove specific char 

trimReview = reviewT.apply(pre_process)
trimSummary = summary.apply(pre_process)
trimGenre = rootG.apply(pre_process)

# generate TD-IDF scores for word representation
print("create TF-IDF vector...")
tfidf_vectorizer = TfidfVectorizer(use_idf=True, max_features=80000, stop_words='english', ngram_range=(1, 2), norm='l2')

tfidf_vectorizer_3 = TfidfVectorizer(use_idf=True, max_features=80000, ngram_range=(1, 2), norm='l2')
tfidf_vectorizer_4 = TfidfVectorizer(use_idf=True, max_features=20, ngram_range=(1, 2), norm='l1')


tfidf_vectorizer_vectors = tfidf_vectorizer.fit_transform(trimReview)
feature_names = tfidf_vectorizer.get_feature_names()

tfidf_vectorizer_vectors3 = tfidf_vectorizer_3.fit_transform(trimSummary)
# tfidf_vectorizer_vectors4 = tfidf_vectorizer_4.fit_transform(trimGenre)

# stacking vectors
X = sp.sparse.hstack(( tfidf_vectorizer_vectors, tfidf_vectorizer_vectors3 ))

selector = SelectKBest(k=40000)
X = selector.fit_transform(X,Y)
# X = sp.sparse.hstack(( X, tfidf_vectorizer_vectors4 ))


# X_new_1 = SelectKBest(k=5000).fit_transform(tfidf_vectorizer_vectors,Y)
# X_new_2 = SelectKBest(k=5000).fit_transform(tfidf_vectorizer_vectors3,Y)
# X = sp.sparse.hstack(( X_new_1, X_new_2 ))

# X = sp.sparse.hstack(( X, csr_matrix(price).T ))
# X = sp.sparse.hstack(( X, csr_matrix(rank).T ))



print("Model training...")

estimators = [
    ('NB', MultinomialNB()),
    ('LR', LogisticRegression(max_iter=400)),
    ('ADA', AdaBoostClassifier(n_estimators=100, random_state=10))
]

clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=20)

# other testing models

# clf = RandomForestClassifier(n_estimators=300, max_depth=8)
# clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
# clf = MultinomialNB()
# clf = LogisticRegression(max_iter=400)
# clf = AdaBoostClassifier(n_estimators=100, random_state=10)

clf.fit(X, Y)

# For training

# clf.fit(X_train, y_train)
# y_pred_1 = clf.predict(X_test)
# print("Accuracy: ",metrics.accuracy_score(y_test, y_pred_1))
# print("F1: ",metrics.f1_score(y_test, y_pred_1, average = 'weighted'))
# print("Precision: ",metrics.precision_score(y_test, y_pred_1))
# print("Recall: ",metrics.recall_score(y_test, y_pred_1))




# generating prediction for testing data

print("predicting test data")

test_raw = pd.read_csv('./datasets/Test.csv')



test_reviewT = test_raw['reviewText'] #string
test_summary = test_raw['summary'] #string
test_price = test_raw['price'] #number
test_rank = test_raw['salesRank'] #number
test_rootG = test_raw['root-genre']

test_trimReview = test_reviewT.apply(pre_process)
test_trimSummary = test_summary.apply(pre_process)
# test_trimGenre = test_rootG.apply(pre_process)

tfidf_vectorizer_vectors = tfidf_vectorizer.transform(test_trimReview)

tfidf_vectorizer_vectors3 = tfidf_vectorizer_3.transform(test_trimSummary)

# tfidf_vectorizer_vectors4 = tfidf_vectorizer_4.transform(test_trimGenre)

test_X = sp.sparse.hstack(( tfidf_vectorizer_vectors, tfidf_vectorizer_vectors3 ))

test_X = selector.transform(test_X)

# test_X = sp.sparse.hstack(( test_X, tfidf_vectorizer_vectors4 ))


y_pred_test = clf.predict(test_X)

test_raw['Awesome'] = y_pred_test

# compare the ratio of 5-star prediction to 

avg = test_raw[['amazon-id', 'Awesome']].groupby('amazon-id').mean() > ratio


result = test_raw['amazon-id'].to_frame().merge(avg, on='amazon-id')
result['Awesome'] = result['Awesome'].apply(lambda x: 1 if x == True else 0)

result = result.drop_duplicates()

output = pd.DataFrame({'amazon-id': result['amazon-id'].astype(str), 'Awesome': result['Awesome']})
output.to_csv('./Product_Predictions.csv')


