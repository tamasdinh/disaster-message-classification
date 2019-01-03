#%%
import pandas as pd
from sqlalchemy import create_engine

#%%
messages = pd.read_csv('./data/disaster_messages.csv')
print(f'Dimensions: {messages.shape}')
print(f'Number of duplicated id values: {messages.id.duplicated().sum()}')
print(f'Number of duplicated rows: {messages.duplicated().sum()}')
print(f'Unique values in genre: {messages.genre.unique()}')
messages.head()

#%%
categories = pd.read_csv('./data/disaster_categories.csv')
print(f'Dimensions: {categories.shape}')
print(f'Number of duplicated id values: {categories.id.duplicated().sum()}')
print(f'Number of duplicated rows: {categories.duplicated().sum()}')
categories.head()

#%%
for df in [messages, categories]:
    df.drop_duplicates(inplace = True)
    print(f'Number of duplicated rows: {df.duplicated().sum()}')
    print(f'Number of duplicated id values: {df.id.duplicated().sum()}')

#%%
df = pd.merge(messages, categories, how = 'left', on = 'id')
print(df.shape)
df.head()

#%%
targets = df.categories.str.split(';', expand = True).applymap(lambda x: x.split('-')[1])
targets.columns = [x.split('-')[0] for x in df.categories[0].split(';')]
print(targets.shape)
targets.head()

#%%
df = pd.concat([df.drop('categories', axis = 1), targets], axis = 1)
print(df.shape)
df.head()

#%%
df.duplicated().sum()

#%%
for column in df.columns:
    try:
        df[column] = df[column].astype('int8')
    except:
        pass
df.dtypes

#%%
engine = create_engine('sqlite:///Fig8messages.db')
df.to_sql('Fig8Messages', engine, if_exists = 'replace')

#%%
pd.read_sql('SELECT * FROM Fig8Messages LIMIT 10', engine)
