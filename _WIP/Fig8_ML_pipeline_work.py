#%% [markdown]
# # ML Pipeline Preparation for disaster message data analysis

#%% [markdown]
# ### Importing all necessary modules
#%%
import pandas as pd
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
import numpy as np
import os
from sqlalchemy import create_engine
os.chdir('/Users/tamasdinh/Dropbox/Data-Science_suli/3_Udacity-Data-Scientist/0_Projects/4_Figure8-disaster-response')

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

import re
import time
import datetime

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score

#%% [markdown]
# ### Setting up tokenizer function
#%%
def tokenize(text):
    clean_text = re.sub(r'[^a-zA-Z]', ' ', text)
    clean_text = word_tokenize(clean_text)
    clean_text = [WordNetLemmatizer().lemmatize(word.lower()) for word in clean_text if word not in stopwords.words('english')]
    return clean_text

#%% [markdown]
# ### Loading data from previously prepared message database
#%%
# load data from database
engine = create_engine('sqlite:///./data/Fig8messages.db')
df = pd.read_sql_table('Messages', engine)
df.head()

#%% [markdown]
# ### Analyzing target labels
#%%
df_counts = df.iloc[:, 5:].mean().reset_index().set_index('index')
df_counts['counts'] = df.iloc[:,5:].sum().values
plt.figure(figsize = (12, 8))
plt.ylim([0, 1])
df_counts.iloc[:, 0].plot.bar()

#%%
min_count_target = df_counts.counts[df_counts.counts > 0].min()
print(f'Minimum count in target label: {min_count_target}')
df_counts.round(2)

#%% [markdown]
# ### Analyzing word count
#%%
from collections import Counter
words = []
for item in df.message:
    msg = tokenize(item)
    words += msg
word_count = pd.Series(Counter(words)).sort_values(ascending = False)
word_count

#%%
plt.figure(figsize = (12, 8))
word_count.plot(kind = 'line')
word_count.describe()

#%%
plt.figure(figsize = (12, 8))
word_count[word_count > min_count_target].plot(kind = 'line')
print(f'Number of word features remaining when taking minimum count of target variables ({min_count_target}): {word_count[word_count > min_count_target].shape[0]}')

#%% [markdown]
# ### Preparing training and test sets
# 'Child alone' has no occurences; for modelling reasons I'll drop that target variable (in a flexible, programmatic way so that if someone wnats to use the app with another dataset he/she can do so)

#%%
category_counts_series = df.loc[:,df.dtypes != 'object'].sum(axis = 0)
zero_cats = [x for x in category_counts_series[category_counts_series == 0].index]
print('The following categories have no records to them:\n {}'.format(zero_cats))
print('-------- Removing {} category label(s) ---------'.format(len(zero_cats)))
df.drop(zero_cats, axis = 1, inplace = True)
X = df[['message']].values.ravel()
y = df.drop(['message', 'genre', 'original', 'id', 'index'], axis = 1).values

#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
print('train-test split ready')

#%%
count_vect = CountVectorizer(tokenizer = tokenize)
X_train = count_vect.fit_transform(X_train)
print('count vectorizer ready')

#%%
filter = (X_train.sum(axis = 0) > min_count_target).A1
X_train = X_train[:,filter]

#%%
tfidf = TfidfTransformer()
X_train = tfidf.fit_transform(X_train)
print('td idf transform ready')

#%%
log_model = RandomForestClassifier(verbose = 3)
print('model fit started...')
log_model.fit(X_train, y_train[:, 1])
print('model fit ready')

#%%
X_test = count_vect.transform(X_test)

#%%
X_test = X_test[:,filter]
X_test = tfidf.transform(X_test)

#%%
y_preds = log_model.predict(X_test)
print('predictions ready')

#%%
hello = y_test[:,0]

#%%
from pprint import pprint
pprint(classification_report(y_preds, y_test[:,1]))
pprint(accuracy_score(y_preds, y_test[:,1]))

#%%
y_test[:,1].mean()

#%%
class TfIdfFilter(BaseEstimator, TransformerMixin):
    
    def __init__(self, tfidf_limit = 80):
        self.filter = 0
        self.tdfidf_limit = tfidf_limit
    
    def fit(self, X, y = None):
        bow = pd.Series(X.sum(axis = 0))
        self.filter = bow[bow > np.percentile(bow, self.tdfidf_limit)].index

    def transform(self, X):
        X = pd.DataFrame(X).iloc[self.filter]
        return X

#%% [markdown]
# ### 3. Build a machine learning pipeline
# - You'll find the [MultiOutputClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) helpful for predicting multiple target variables.

#%%
pipeline = Pipeline([
    ('countvect', CountVectorizer(tokenizer = tokenize)),
    ('tf-idf', TfidfTransformer()),
#    ('tf-idf-filter', TfIdfFilter()),
    ('multi-output-classification', LogisticRegression())
])

#%% [markdown]
# ### 4. Train pipeline
# - Split data into train and test sets
# - Train pipeline

#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
print('Fitting pipeline model on training data\n------------------------------')
start = time.time()
model = pipeline
model.fit_transform(X_train, y_train[:, 0])
end = time.time()
print('\nElapsed time: {:.0f} seconds'.format(end-start))
print('\nGenerating predictions on test data\n------------------------------')
start = time.time()
y_preds = model.predict(X_test)
end = time.time()
print('Elapsed time: {:.0f} seconds'.format(end-start))

#%% [markdown]
# ### 5. Test your model
# Report the f1 score, precision and recall for each output category of the dataset. You can do this by iterating through the columns and calling sklearn's `classification_report` on each.

#%%
heclassification_report)

#%%
def classification_report(y_preds, y_test):
    print('Generating classification metrics\n------------------------------')
    start = time.time()
    for i in range(y_preds.shape[1]):
        print(category_counts_series.index[i], '\n-------------------------------')
        print(classification_report(y_test[:, i], y_preds[:, i]))
    end = time.time()
    print('Elapsed time: {:.0f} seconds'.format(end-start))

#%%
y_train[:, 0].shape

#%%
y_test.shape

#%%
precision_score(y_preds, y_test[:,0], average = 'micro')

#%%
pd.DataFrame(y_test, columns = count_vect.vocabulary_)

#%%


#%%
