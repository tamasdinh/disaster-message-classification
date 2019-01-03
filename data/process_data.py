import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    for df in [messages, categories]:
        df.drop_duplicates(inplace = True)
    df = pd.merge(messages, categories, how = 'left', on = 'id')
    return df

def clean_data(df):
    targets = df.categories.str.split(';', expand = True).applymap(lambda x: x.split('-')[1])
    targets.columns = [x.split('-')[0] for x in df.categories[0].split(';')]
    df = pd.concat([df.drop('categories', axis = 1), targets], axis = 1)
    for column in df.columns:
        try:
            df[column] = df[column].astype('int8')
        except:
            pass
    return df

def save_data(df, database_filepath, table_name):
    engine = create_engine(f'sqlite:///{database_filepath}')
    df.to_sql(table_name, engine, if_exists = 'replace')


def main():
    if len(sys.argv) == 5:
        messages_filepath, categories_filepath, database_filepath, table_name = sys.argv[1:]
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)
        print('Cleaning data...')
        df = clean_data(df)
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath, table_name)
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument and the table name as the fourth argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db Messages')

if __name__ == '__main__':
    main()