# imports
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

# pre-processing and evaluation
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score, roc_auc_score, roc_curve, auc
from sklearn.inspection import permutation_importance



def acquire_df(filepath = 'medical_raw_df.csv'):
    filepath = filepath
    df = pd.read_csv(filepath, index_col=[0])
    # lowercase columns
    df.columns = map(str.lower, df.columns)
    # verify no nulls
    assert df.isnull().sum().sum() == 0
    # verify no duplicates
    assert df.duplicated().sum() == 0
    
    return df


def clean_df(df):
    df.columns = map(str.lower, df.columns)
    
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
    columns = {'uid':'unique_id',
              'readmis':'readmission',
              'vitd_supp':'vitd_supplement',
              'highblood':'high_blood',
              'services':'services_received',
              'totalcharge':'daily_charges',
              'initial_days':'hospital_stay_days'}

    df[['item1','item2', 'item3', 'item4']] = df[['item1','item2', 'item3', 'item4']].astype('int32')
    df[['item5','item6', 'item7', 'item8']] = df[['item5','item6', 'item7', 'item8']].astype('int32')
    df.rename(columns=columns, inplace=True)
    # df = df.set_index('caseorder')
    ## remove unnecessary columns not used for modeling
    remove_cols = ['customer_id',
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
    
    
def dummify(df, cat_vars, bool_vars): 
    '''
    This function takes in the medical dataframe and cat_vars
    and creates dummy columns
    df: medical dataframe
    cat_vars: columns to dummify
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

    # replace True with 1's and False with 0's
    df[bool_vars] = df[bool_vars].replace(True, 1)
    df[bool_vars] = df[bool_vars].replace(False, 0)
    df[bool_vars].head()
    
    return df


def split_data(df, test_size=.2):
    '''
    This function takes in the medical dataset and a test_size
    and splits the data according to the test_size. It also separates
    the y target from train and validate
    
    arguments:
    df: medical dataframe
    test_size: default .2 other any float
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
    
    global trainY
    global valY
    global trainX
    global valX

    # correct diabetes to 1,0
    train['diabetes'] = train['diabetes'].replace({'True':1, 'False':0}).astype('int32')
    validate['diabetes'] = validate['diabetes'].replace({'True':1, 'False':0}).astype('int32')

    ## separate y target
    trainY = train['diabetes']
    valY = validate['diabetes']

    trainX = train.drop(columns=['diabetes'])
    valX = validate.drop(columns=['diabetes'])
    
    return train, validate




def naivebayes_model_training(nb_algo, trainX, trainY, valX, valY, columns=None):
    '''
    This function takes in the split data sets and the specific Naive Bayes
    algorithm, trains a model, prints out the metrics scores as well
    as the confusion matrix.

    Arguments:
    nb_algo: Method of NB algorithm
    trainX: raining features
    trainY: raining label
    valX: validation features
    valY: validation label
    columns: subset of columns from trainX to use, optional
    '''
    
    # Ensure column filtering works correctly
    if columns is not None:
        trainX = trainX[columns]
        valX = valX[columns]
        print(f'Using reduced column set: {columns}')
        
    print(f'Algorithm {nb_algo.__class__.__name__}')
    
    # Train the model
    nb_algo.fit(trainX, trainY)

    # Predictions
    trainY_pred = nb_algo.predict(trainX)
    valY_pred = nb_algo.predict(valX)

    # Print classification report for validation set
    print("-------------")
    print('Classification report for Validation')
    print(classification_report(valY, valY_pred))
    print("-------------")

    # Calculate and print confusion matrix details
    tn, fp, fn, tp = confusion_matrix(valY, valY_pred).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)

    print(f"Val Accuracy is {accuracy:.3f}")
    print(f"Val Recall is {recall:.3f}")
    print(f"Val Precision is {precision:.3f}")
    print("-------------")

    # Print F1 scores for train and validation sets
    print(f'F-1 Score of {nb_algo.__class__.__name__} classifier on train set: {f1_score(trainY, trainY_pred, average="weighted"):.3f}')
    print(f'F-1 Score of {nb_algo.__class__.__name__} classifier on validate set: {f1_score(valY, valY_pred, average="weighted"):.3f}')
    print('--------------')

    # Confusion matrix and heatmap
    val_cm = confusion_matrix(valY, valY_pred)
    labels = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    labels = np.asarray(labels).reshape(2, 2)
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(val_cm / np.sum(val_cm), annot=True, fmt='.0%', cmap=cmap,
                xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'], linewidth=.5)
    plt.xlabel('Predicted Diabetes')
    plt.ylabel('Actual Diabetes')
    plt.show()
    
    
    
    
    
    
    
    
def main():
    # filepath 
    filepath = 'medical_raw_df.csv'
    
    # acquire
    df = acquire_df(filepath)
    
    # clean
    df_clean = clean_df(df)
    
    # variables
    cat_vars = ['gender', 'marital', 'area', 'initial_admin', 'complication_risk', 'services_received']
    bool_vars = ['readmission', 'high_blood', 'stroke', 'arthritis', 'overweight', 'hyperlipidemia', 
                 'backpain', 'anxiety', 'allergic_rhinitis', 'reflux_esophagitis', 'asthma', 'soft_drink']
    
    # dummify cat_vars
    df_prepped = dummify(df_clean, cat_vars, bool_vars)
    
    # Step 4: Split the data into training and validation sets
    train, validate = split_data(df_prepped)
    
    # Step 5: Train the Naive Bayes model
    nb_algo = MultinomialNB()  # Example of using Multinomial Naive Bayes
    naivebayes_model_training(nb_algo, trainX, trainY, valX, valY)
    
if __name__ == "__main__":
    main()
