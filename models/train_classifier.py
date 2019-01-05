import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re
import time
from sqlalchemy import create_engine
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, make_scorer


def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Messages', engine)

    df_naive = df.iloc[:,5:].mean(axis = 0).reset_index().set_index('index')
    df_naive.columns = ['Precision']
    df_naive['F1-score'] = (1 + 1**2) * (df_naive.Precision * 1) / (1**2 * (df_naive.Precision + 1))
    df_naive = df_naive[['F1-score', 'Precision']].sort_values(by = 'F1-score', ascending = False)
    
    category_counts_series = df.drop(['message', 'genre', 'original', 'id', 'index'], axis = 1).sum(axis = 0)
    zero_cats = [x for x in category_counts_series[category_counts_series == 0].index]
    print('The following categories have no records to them:\n {}'.format(zero_cats))
    print('-------- Removing {} category label(s) ---------'.format(len(zero_cats)))
    df.drop(zero_cats, axis = 1, inplace = True)
    df_naive.drop(zero_cats, inplace = True)
    category_counts_series.drop(zero_cats, inplace = True)
    
    X = df[['message']].values.ravel()
    y = df.drop(['message', 'genre', 'original', 'id', 'index'], axis = 1)
    y = y[df_naive.index].values

    return X, y, [x for x in df_naive.index], df_naive


def tokenize(text):
    clean_text = re.sub(r'[^a-zA-Z]', ' ', text)
    clean_text = word_tokenize(clean_text)
    clean_text = [WordNetLemmatizer().lemmatize(word.lower()) for word in clean_text if word not in stopwords.words('english')]
    return clean_text


class WordCountFilter(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y):
        min_count_target = y.sum(axis = 0).min()
        self.filter = (X.sum(axis = 0) > min_count_target).A1
        return self

    def transform(self, X):
        return X[:,self.filter]


def build_model():
    pipeline = Pipeline([
        ('countvect', CountVectorizer(tokenizer = tokenize)),
        ('wordcount-filter', WordCountFilter()),
        ('tf-idf', TfidfTransformer()),
        ('multi-output-classification', MultiOutputClassifier(GradientBoostingClassifier(n_estimators = 300, n_iter_no_change = 50)))
    ])
    parameters = {
        'multi-output-classification__estimator__max_depth': [3], #[1, 3],
        'multi-output-classification__estimator__learning_rate': [0.1], #[0.1, 0.2, 0.5],
        'multi-output-classification__estimator__n_estimators': [300] #[300, 500]
    }
    scorer = make_scorer(roc_auc_score)
    cv = GridSearchCV(pipeline, param_grid = parameters, scoring = scorer, cv = 2)
    return cv


def metrics_collection(y_preds, y_test, category_names):
    prec_list = []
    recall_list = []
    f1_score_list = []
    for i in range(y_preds.shape[1]):
        prec_list.append(precision_score(y_test[:, i], y_preds[:, i]))
        recall_list.append(recall_score(y_test[:, i], y_preds[:, i]))
        f1_score_list.append(f1_score(y_test[:, i], y_preds[:, i]))
    df = pd.DataFrame({'F1-score': f1_score_list, 'Precision': prec_list, 'Recall': recall_list}, index = category_names)
    df = df.loc[category_names]
    return df


def metrics_plotting(dataframes, model_filepath, category_names):
    colors = ['#4284D8','#ECB26F'] # blue, yellow
    categories = ['Naive benchmark', 'Tuned model']
    patches = []
    for i, category in enumerate(categories):
        patches.append(mpatches.Patch(color = colors[i], label = category))
    plt.figure(figsize = (24, 8))
    titles = ['F1-score', 'Precision score', 'Recall score']
    for i in range(len(titles)):
        if i == 2:
            plt.legend(handles = patches, bbox_to_anchor = (0.5, 1.2, 0, 0), \
               loc = 'upper center', borderaxespad = 0., ncol = len(patches), fontsize = 'large')
        ax = plt.subplot(1, 3, i + 1)
        ax.set_title(titles[i], fontsize = 14)
        ax.set_ylim([0, 1])
        ax.set_xlim([0, len(category_names)])
        ax.set_xticks(range(len(category_names)))
        ax.hlines([0.2, 0.4, 0.6, 0.8], 0, len(category_names), color = 'lightblue', linestyle = '--', linewidth = 1)
        ax.set_xticklabels(category_names)
        for tick in ax.get_xticklabels():
            tick.set_rotation(90)
        for w in range(len(dataframes)):
            try:
                ax.bar(range(len(category_names)), dataframes[w].iloc[:,i], alpha = 0.5, color = colors[w])
            except:
                pass
    plt.savefig(f'{model_filepath.replace(model_filepath.split(".")[-1], "png")}', bbox_inches = 'tight')
    plt.show()


def evaluate_model(model, X_test, Y_test, model_filepath, category_names, naive_benchmark):
    y_preds = model.predict(X_test)
    metrics_df = metrics_collection(y_preds, Y_test, category_names)
    print(metrics_df)

    metrics_list = []
    metrics_list.append(naive_benchmark)
    metrics_list.append(metrics_df)
    metrics_plotting(metrics_list, model_filepath, category_names)


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as pkl:
        pickle.dump(model.best_estimator_, pkl)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names, naive_benchmark = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, model_filepath, category_names, naive_benchmark)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')
        print('You can see a comparison of model performance against the naive benchmark in the .png file saved with your model.')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()