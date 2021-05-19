import pandas as pd 
import numpy as np 
import os 
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

def max_value(df, variable, top):
    return np.where(df[variable] > top, top, df[variable])

def preprocess_and_save_data(filename, output_dir):
    df = pd.read_csv(filename)
    df.dropna(subset=['RainTomorrow'], inplace=True)

    #extract Date variable into Year, Month and Day 
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day 
    df.drop('Date', axis=1, inplace = True)


    # categorical = [col for col in X.columns if X[col].dtypes == 'O']
    numerical = [col for col in df.columns if df[col].dtypes != 'O']
    
    #fill missing values for numerical and categorical variables
    for col in numerical: 
        df.iloc[:][col].fillna(df[col].median(), inplace=True)
    df.loc[:]['WindGustDir'].fillna(df['WindGustDir'].mode()[0], inplace=True)
    df.loc[:]['WindDir9am'].fillna(df['WindDir9am'].mode()[0], inplace=True)
    df.loc[:]['WindDir3pm'].fillna(df['WindDir3pm'].mode()[0], inplace=True)
    df.loc[:]['RainToday'].fillna(df['RainToday'].mode()[0], inplace=True)
    
    # use interquantile range to remove outliers of numerical variables
    df.loc[:]['Rainfall'] = max_value(df, 'Rainfall', 3.2)
    df.loc[:]['Evaporation'] = max_value(df, 'Evaporation', 21.8)
    df.loc[:]['WindSpeed9am'] = max_value(df, 'WindSpeed9am', 55)
    df.loc[:]['WindSpeed3pm'] = max_value(df, 'WindSpeed3pm', 57)
    
    X = df.drop(['RainTomorrow'], axis = 1)
    y = df['RainTomorrow']
   
    # binary encode RainToday variable
    encoder = ce.BinaryEncoder(cols=['RainToday'])
    X = encoder.fit_transform(X)
   
    # encode target variable
    le = LabelEncoder()
    y_transformed = le.fit_transform(y)
    
    # dummy categorical variables 
    X_dummy = pd.get_dummies(X,  columns=["Location", "WindGustDir", "WindDir9am", "WindDir3pm"])
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_dummy)
    print('Saving feature vectors to {}'.format(os.path.join(output_dir, 'feature_vectors.npy')))
    np.save(os.path.join(output_dir, 'feature_vectors.npy'), X_scaled)
    print('Saving label vector to {}'.format(os.path.join(output_dir, 'labels.npy')))
    np.save(os.path.join(output_dir, 'labels.npy'), y_transformed)

if __name__ == '__main__':
    OUTPUT_DIR = 'data'
    data_file = 'data/weatherAUS.csv'
    preprocess_and_save_data(data_file, OUTPUT_DIR)

