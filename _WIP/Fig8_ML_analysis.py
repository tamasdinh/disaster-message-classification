#%% [markdown]
# ### Importing all necessary modules
#%%
import pandas as pd
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
import numpy as np
from sqlalchemy import create_engine
import re
import os
os.chdir('/Users/tamasdinh/Dropbox/Data-Science_suli/3_Udacity-Data-Scientist/0_Projects/4_Figure8-disaster-response')

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


#%% [markdown]
# ### Loading data from previously prepared message database
#%%
engine = create_engine('sqlite:///./data/Fig8messages.db')
df = pd.read_sql_table('Messages', engine)
df.head()

#%% [markdown]
# ### Setting up tokenizer function
#%%
def tokenize(text):
    clean_text = re.sub(r'[^a-zA-Z]', ' ', text)
    clean_text = word_tokenize(clean_text)
    clean_text = [WordNetLemmatizer().lemmatize(word.lower()) for word in clean_text if word not in stopwords.words('english')]
    return clean_text

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