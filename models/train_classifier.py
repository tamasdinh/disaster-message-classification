import sys
import pandas as pd
import numpy as np
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

    category_counts_series = df.drop(['message', 'genre', 'original', 'id', 'index'], axis = 1).sum(axis = 0)
    zero_cats = [x for x in category_counts_series[category_counts_series == 0].index]
    print('The following categories have no records to them:\n {}'.format(zero_cats))
    print('-------- Removing {} category label(s) ---------'.format(len(zero_cats)))
    df.drop(zero_cats, axis = 1, inplace = True)
    category_counts_series.drop(zero_cats, inplace = True)
    X = df[['message']].values.ravel()
    y = df.drop(['message', 'genre', 'original', 'id', 'index'], axis = 1).values
    return X, y, [x for x in category_counts_series.index]


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
        ('multi-output-classification', MultiOutputClassifier(GradientBoostingClassifier(n_estimators = 300, verbose = 1, n_iter_no_change = 50)))
    ])
    parameters = {
        'multi-output-classification__estimator__max_depth': [3], #[1, 3],
        'multi-output-classification__estimator__learning_rate': [0.1] #[0.1, 0.2, 0.5],
        'multi-output-classification__estimator__n_estimators': [300] #[300, 500]
    }
    scorer = make_scorer(roc_auc_score)
    cv = GridSearchCV(pipeline, param_grid = parameters, scoring = scorer, cv = 3)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    def metrics_collection(y_preds, y_test):
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
    y_preds = model.predict(X_test)
    metrics_df = metrics_collection(y_preds, Y_test)
    print(metrics_df)


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as pkl:
        pickle.dump(model, pkl)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()