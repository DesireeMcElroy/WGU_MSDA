# imports
import pandas as pd
import numpy as np

# pre-processing
from sklearn.preprocessing import StandardScaler



def get_df(filepath='medical_raw_df.csv'):
    '''
    This function takes in filepath as an argument and returns a readable dataframe
    with dataframe info
    
    
    Args:
    filepath: default 'medical_raw_df.csv'
    '''
    
    df = pd.read_csv(filepath, index_col=[0])
    # lowercase columns
    df.columns = map(str.lower, df.columns)
    # verify no nulls
    assert df.isnull().sum().sum() == 0
    # verify no duplicates
    assert df.duplicated().sum() == 0
    
    
    print(df.info())
    
    return df




def clean_df(df):
    '''
    This function takes the medical dataframe and cleans it by correcting time zone entries,
    zip code, unifying the boolean value entries to 1,0, rounding continuous vars reassigning
    and data types. Lastly, it removes unnescessary columns.
    
    Args:
    df: the medical dataset in a pandas dataframe
    '''

    # changing datatypes
    # change columns to boolean data type
    to_bool = ['readmis',
               'soft_drink',
               'highblood',
               'stroke',
               'overweight',
               'arthritis',
               'diabetes',
               'hyperlipidemia',
               'backpain',
               'anxiety',
               'allergic_rhinitis',
               'reflux_esophagitis',
               'asthma']
    
    for col in to_bool:
        df[col] = df[col].replace({'Yes':1, 'No':0}).astype(bool)
    
    # round entries in columns to only have two decimal places
    round_num = ['vitd_levels',
                 'totalcharge',
                 'additional_charges']
    for col in round_num:
        df[col] = round(df[col], 2)
    
    # change columns to integer data type
    to_int = ['population',
              'children',
              'age',
              'income',
              'initial_days']
    for col in to_int:
        df[col] = df[col].astype('int32')
    
    # change columns to categorical data type
    to_cat = ['marital',
              'gender',
              'initial_admin',
              'services',
              'item1',
              'item2',
              'item3',
              'item4',
              'item5', 
              'item6',
              'item7',
              'item8',
              'timezone',
              'state',
              'complication_risk']
    for col in to_cat:
        df[col] = df[col].astype('category')
          
    # make columns more readable  
    columns = {'caseorder':'case_order',
              'uid':'unique_id',
              'readmis':'readmission',
              'vitd_supp':'vitd_supplement',
              'highblood':'high_blood',
              'services':'services_received',
              'totalcharge':'daily_charges',
              'initial_days':'hospital_stay_days'}
    
    df[['item1','item2', 'item3', 'item4']] = df[['item1','item2', 'item3', 'item4']].astype('int32')
    df[['item5','item6', 'item7', 'item8']] = df[['item5','item6', 'item7', 'item8']].astype('int32')
    df.rename(columns=columns, inplace=True)
    df = df.reset_index()
    ## remove unnecessary columns not used for modeling
    remove_cols = ['CaseOrder',
                   'customer_id',
                   'interaction',
                   'unique_id',
                   'city',
                   'state',
                   'county',
                   'zip',
                   'lat',
                   'lng',
                   'income',
                   'job',
                   'timezone'
                    ,'item1'
                    ,'item2'
                    ,'item3'
                    ,'item4'
                    ,'item5'
                    ,'item6' 
                    ,'item7'
                    ,'item8']
    df.drop(columns=remove_cols, inplace=True)
    
    return df


def dummify_categorical(df, cat_vars)
    '''
    This function
    '''
    print(f'Categorical variables to be one hot encoded {cat_vars}')
    print('---------')
    # Create dummies for the specified columns
    dummy_df = pd.get_dummies(df[cat_vars], drop_first=True)

    print(f'New dummy columns {list(dummy_df.columns)}')

    # concatenate the original dataframe with the dummies
    df = pd.concat([df, dummy_df], axis=1)

    # # drop the original columns 
    df.drop(columns = cat_vars, inplace=True)
    
    return df



def scale_vars(df, cont_vars, num_vars):
    '''
    This function
    '''
    # assign scaler
    scaler = StandardScaler()

    # run scale function on columns to scale
    df[cont_vars] = scaler.fit_transform(df[cont_vars])
    
    # run scale function on columns to scale
    df[num_vars] = scaler.fit_transform(df[num_vars])
    
    return df





def transform_bool(df, bool_vars):
    '''
    This function
    '''
    # replace True with 1's and False with 0's
    df[bool_vars] = df[bool_vars].replace(True, 1)
    df[bool_vars] = df[bool_vars].replace(False, 0)

    df[bool_vars].head()

return df



def split_data(df, test_size=.2)
    '''

    '''
    # split data into train, validate and test sets
    train, validate = train_test_split(df,
                                            test_size=test_size, 
                                            random_state=314,
                                            shuffle=True)


    train.to_csv('train_df.csv')
    validate.to_csv('validate_df.csv')

    print(f'Train shape ---> {train.shape}')
    print(f'Validate shape ---> {validate.shape}')


    return train, validate







def model_preprocessing():

    



    return preprocessed_df
