# imports
import pandas as pd
import numpy as np



def wrangle_df(filepath='medical_raw_df.csv'):
    
    filepath = filepath
    df = pd.read_csv(filepath, index_col=[0])
    
    # lowercase columns
    df.columns = map(str.lower, df.columns)
    # verify no nulls
    assert df.isnull().sum().sum() == 0
    # verify no duplicates
    assert df.duplicated().sum() == 0
    
    # change timezone column entries before changing data type
    tz_dict = {
        "America/Puerto_Rico" : "US - Puerto Rico",
        "America/New_York": "US - Eastern",
        "America/Detroit" : "US - Eastern",
        "America/Indiana/Indianapolis" : "US - Eastern",
        "America/Indiana/Vevay" : "US - Eastern",
        "America/Indiana/Vincennes" : "US - Eastern",
        "America/Kentucky/Louisville" : "US - Eastern",
        "America/Toronto" : "US - Eastern",
        "America/Indiana/Marengo" : "US - Eastern",
        "America/Indiana/Winamac" : "US - Eastern",
        "America/Chicago" : "US - Central", 
        "America/Menominee" : "US - Central",
        "America/Indiana/Knox" : "US - Central",
        "America/Indiana/Tell_City" : "US - Central",
        "America/North_Dakota/Beulah" : "US - Central",
        "America/North_Dakota/New_Salem" : "US - Central",
        "America/Denver" : "US - Mountain",
        "America/Boise" : "US - Mountain",
        "America/Phoenix" : "US - Arizona",
        "America/Los_Angeles" : "US - Pacific",
        "America/Nome" : "US - Alaskan",
        "America/Anchorage" : "US - Alaskan",
        "America/Sitka" : "US - Alaskan",
        "America/Yakutat" : "US - Alaskan",
        "America/Adak" : "US - Aleutian",
        "Pacific/Honolulu" : 'US - Hawaiian'
        }
    df.timezone.replace(tz_dict, inplace=True)
    
    # convert zip column to str, then fill 0s in entries
    df.zip = df.zip.astype('str').str.zfill(5)
    
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
                   # 'state',
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





def model_preprocessing():

    



    return preprocessed_df
