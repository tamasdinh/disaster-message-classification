#%% [markdown]
# # ML Pipeline Preparation for disaster message data analysis

#%% [markdown]
# ### Importing all necessary modules
#%%
import pandas as pd
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sqlalchemy import create_engine
import pickle
import os
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
from datetime import datetime

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, make_scorer

#%% [markdown]
# ### Loading data from previously prepared message database
#%%
engine = create_engine('sqlite:///./data/Fig8messages.db')
df = pd.read_sql_table('Messages', engine)
df.head()

#%% [markdown]
# ### Preparing training and test sets
#%%
category_counts_series = df.drop(['message', 'genre', 'original', 'id', 'index'], axis = 1).sum(axis = 0)
zero_cats = [x for x in category_counts_series[category_counts_series == 0].index]
print('The following categories have no records to them:\n {}'.format(zero_cats))
print('-------- Removing {} category label(s) ---------'.format(len(zero_cats)))
df.drop(zero_cats, axis = 1, inplace = True)
category_counts_series.drop(zero_cats, inplace = True)
X = df[['message']].values.ravel()
y = df.drop(['message', 'genre', 'original', 'id', 'index'], axis = 1).values

#%% [markdown]
# ### Setting up tokenizer function
#%%
def tokenize(text):
    clean_text = re.sub(r'[^a-zA-Z]', ' ', text)
    clean_text = word_tokenize(clean_text)
    clean_text = [WordNetLemmatizer().lemmatize(word.lower()) for word in clean_text if word not in stopwords.words('english')]
    return clean_text

#%% [markdown]
# ### Setting up custom Transformer for word count filtering
#%%
class WordCountFilter(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y):
        min_count_target = y.sum(axis = 0).min()
        self.filter = (X.sum(axis = 0) > min_count_target).A1
        return self

    def transform(self, X):
        return X[:,self.filter]

#%% [markdown]
# ### Setting up metric collection
#%%
df_metrics = []
def metrics_collection(y_preds, y_test):
    prec_list = []
    recall_list = []
    f1_score_list = []
    for i in range(y_preds.shape[1]):
        prec_list.append(precision_score(y_test[:, i], y_preds[:, i]))
        recall_list.append(recall_score(y_test[:, i], y_preds[:, i]))
        f1_score_list.append(f1_score(y_test[:, i], y_preds[:, i]))
    df = pd.DataFrame({'F1-score': f1_score_list, 'Precision': prec_list, 'Recall': recall_list}, index = category_counts_series.index)
    df = df.loc[df_naive.index]
    return df

#%% [markdown]
# ### Calculating naive benchmark
#%%
df_naive = df.iloc[:,5:].mean(axis = 0).reset_index().set_index('index')
df_naive.columns = ['Precision']
df_naive['F1-score'] = (1 + 1**2) * (df_naive.Precision * 1) / (1**2 * (df_naive.Precision + 1))
df_naive = df_naive[['F1-score', 'Precision']].sort_values(by = 'F1-score', ascending = False)
df_metrics.append(df_naive)

#%% [markdown]
# ### Setting up machine learning pipeline
#%%
pipeline = Pipeline([
    ('countvect', CountVectorizer(tokenizer = tokenize)),
    ('wordcount-filter', WordCountFilter()),
    ('tf-idf', TfidfTransformer()),
    ('multi-output-classification', MultiOutputClassifier(GradientBoostingClassifier(n_estimators = 300, verbose = 1, n_iter_no_change = 50)))
])

#%% [markdown]
# ### Training pipeline
#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
print('Fitting pipeline model on training data\n------------------------------')
start = time.time()
model = pipeline
model.fit(X_train, y_train)
end = time.time()
print('Total time elapsed: {:0f} minutes ------'.format((end-start)/60))

#%% [markdown]
# ### Generating predictions on test data
#%%
print('\nGenerating predictions on test data\n------------------------------')
start = time.time()
y_preds = model.predict(X_test)
end = time.time()
print('Elapsed time: {:.0f} seconds'.format(end-start))

#%% [markdown]
# ### Testing model
#%%
df_metrics_untuned = metrics_collection(y_preds, y_test)
df_metrics.append(df_metrics_untuned)
df_metrics_untuned

#%% [markdown]
# ### Plotting results
#%%
def metrics_plotting(dataframes):
    colors = ['#4284D8','#ECB26F', '#59C3B1', '#F96161'] # blue, yellow, green, light red
    categories = ['Naive benchmark', 'Untuned model', 'Tuned model']
    patches = []
    for i, category in enumerate(categories):
        patches.append(mpatches.Patch(color = colors[i], label = category))
    plt.figure(figsize = (24, 8))
    titles = ['F1-score', 'Precision score', 'Recall score']
    for i in range(3):
        if i == 2:
            plt.legend(handles = patches, bbox_to_anchor = (0.5, 1.2, 0, 0), \
               loc = 'upper center', borderaxespad = 0., ncol = len(patches), fontsize = 'large')
        ax = plt.subplot(1, 3, i + 1)
        ax.set_title(titles[i], fontsize = 14)
        ax.set_ylim([0, 1])
        ax.set_xlim([0, df_naive.shape[0]])
        ax.set_xticks(range(df_naive.shape[0]))
        ax.hlines([0.2, 0.4, 0.6, 0.8], 0, df_naive.shape[0], color = 'lightblue', linestyle = '--', linewidth = 1)
        ax.set_xticklabels(df_naive.index)
        for tick in ax.get_xticklabels():
            tick.set_rotation(90)
        for w in range(len(dataframes)):
            try:
                ax.bar(range(df_naive.shape[0]), dataframes[w].iloc[:,i], alpha = 0.25, color = colors[w])
            except:
                pass
    plt.show()

#%%
metrics_plotting(df_metrics)

#%% [markdown]
# ### Tuning model with GridSearchCV
#%%
parameters = {
    'multi-output-classification__estimator__max_depth': [1, 3],
    'multi-output-classification__estimator__learning_rate': [0.1, 0.2, 0.5],
    'multi-output=classification__estimator__n_estimators': [300, 500]
}
scorer = make_scorer(roc_auc_score)
cv = GridSearchCV(pipeline, param_grid = parameters, scoring = scorer, cv = 3, return_training_score = True)

#%%
from sklearn.metrics import roc_auc_score
for i in range(y_preds.shape[1]):
    print(f'{df_naive.index[i]} ---', roc_auc_score(y_test[:,i], y_preds[:,i]))

#%%
start = time.time()
print('Starting grid search optimization at {} -------'.format(time.time()))
cv.fit(X_train, y_train)
end = time.time()
print('Total time elapsed: {:0f} minutes ------'.format((end-start)/60))

#%%
y_preds = cv.best_estimator_.predict(X_test)
df_metrics_tuned = metrics_collection(y_preds, y_test)
df_metrics.append(df_metrics_tuned)
metrics_plotting(df_metrics)

#%%
grid_search_res = pd.DataFrame(cv.cv_results_).iloc[:,[4, 5, 10, 11, 12, 16, 17]].set_index('rank_test_score').sort_index(ascending = True)
grid_search_res.columns = [x.split('__')[-1] for x in grid_search_res.columns]
grid_search_res

#%%
grid_search_res.to_csv(f'./models/Grid_search_results_{time.strftime("%Y%m%d_%H%M%S")}.csv')

#%%
with open(f'./models/optim_model_{time.strftime("%Y%m%d_%H%M%S")}.pkl', 'wb') as pkl:
    pickle.dump(cv.best_estimator_, pkl)
