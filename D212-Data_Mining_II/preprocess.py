import pandas as pd
import numpy as np

filepath = 'medical_clean.csv'

continuous_columns = [
    'population',
    'children',
    'age',
    'income',
    'vitd_levels',
    'doc_visits',
    'full_meals_eaten',
    'length_of_stay',
    'total_charges',
    'additional_charges'
]



def preprocess_df(df=None, keep_continuous=True):
    '''
    
    '''

    ## read in df if not provdided
    if df is None:
        df = pd.read_csv(filepath)

    df.columns = df.columns.str.lower()
    print(f'dataframe shape: {df.shape}')

    df.rename(columns={'totalcharge': 'total_charges',
                       'initial_days': 'length_of_stay',
                       'caseorder': 'case_order'}, inplace=True)

    df.set_index('case_order', inplace=True)
    if keep_continuous:
        df = df[continuous_columns]

    return df



