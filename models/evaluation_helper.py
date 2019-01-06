import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def naive_benchmark(df, target_cols):
    '''
    Calculates naive benchmark for binary classification cases. Especially useful for binary multioutput classification.
    Inputs: df - dataframe with all data (predictors and targets)
            target_cols - column names of target variables
    Output: dataframe with target variables as index, F1-score, precision, recall as columns
    '''
    df_naive = df.loc[:, target_cols].mean(axis = 0).reset_index().set_index('index')
    df_naive.columns = ['Precision']
    df_naive['F1-score'] = (1 + 1**2) * (df_naive.Precision * 1) / (1**2 * (df_naive.Precision + 1))
    df_naive = df_naive[['F1-score', 'Precision']].sort_values(by = 'F1-score', ascending = False)
    return df_naive


def metrics_collection(y_preds, y_test, category_names):
    '''
    Calculates and structures key classification performance metrics on a given model with multioutput option.
    Inputs: predictions on test set targets / true test set target classes / name(s) of target variable(s)
    Returns: dataframe - index: category names, F1-score, Precision score, Recall score
    '''
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
    '''
    Produces 3 subplots on key performance metrics.
    Inputs: dataframes - list of dataframes with index: category names
            model_filepath - path to save image of chart (takes the same name as the model)
            category_names - names of target variables
    Returns: saves chart in png format to disk, shows chart in pop-up window
    '''
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
