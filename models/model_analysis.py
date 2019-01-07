#%%
import os
os.chdir('/Users/tamasdinh/Dropbox/Data-Science_suli/3_Udacity-Data-Scientist/0_Projects/4_Figure8-disaster-response/models')
import evaluation_helper as eh
from train_classifier import *

database_filepath = '../data/Fig8messages.db'
model_filepath = 'test_model.pkl'

#%%
X, Y, category_names, naive_benchmark = load_data(database_filepath)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

with open(model_filepath, 'rb') as m:
    model = pickle.load(m)

#%%
model.estimator.named_steps['multi-output-classification'].estimator.feature_importances_


#%%
def timeit(func):
    import time
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print('1 Elapsed time: {:1f} minutes'.format((end-start)/60))
        return result
    return wrapper


#%%
@timeit
def pred(model, X_test):
    return model.predict(X_test)

#%%
y_preds = pred(model, X_test)


#%%
learning_rate = 0.1

def learning(w1_init, w2_init, a, b, learning_rate):
    iter = 0
    w1 = w1_init
    w2 = w2_init
    while (w1 * a + w2 * b - 10) < 0:
        w1 += learning_rate
        w2 += learning_rate
        iter += 1
    print(iter)

learning(3, 4, 1, 1, 0.1)

