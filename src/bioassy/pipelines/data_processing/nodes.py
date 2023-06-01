import pandas as pd
import numpy as np
import sklearn
import imblearn
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

def preparing_list_of_train_test_data(df_1284mr,df_1284r,df_1608mr,df_1608r,df_362r,df_373_439r,df_373r,df_439mr,df_439r,df_456r,df_604_644r,df_604r,df_644mr,df_644r,df_687_721r,df_687r,df_688r,df_721mr,df_721r,df_746_1284r,df_746r,df_1284mr_t,df_1284r_t,df_1608mr_t,df_1608r_t,df_362r_t,df_373_439r_t,df_373r_t,df_439mr_t,df_439r_t,df_456r_t,df_604_644r_t,df_604r_t,df_644mr_t,df_644r_t,df_687_721r_t,df_687r_t,df_688r_t,df_721mr_t,df_721r_t,df_746_1284r_t,df_746r_t):
    """Preparing list of dataframe
    args: train dataframes and test dataframes
    Return: list_df_train,list_df_test,df_list_names"""
    df_list_train = [df_1284mr,df_1284r,df_1608mr,df_1608r,df_362r,df_373_439r,df_373r,df_439mr,df_439r,df_456r,df_604_644r,df_604r,df_644mr,df_644r,df_687_721r,df_687r,df_688r,df_721mr,df_721r,df_746_1284r,df_746r]
    list_df_test = [df_1284mr_t,df_1284r_t,df_1608mr_t,df_1608r_t,df_362r_t,df_373_439r_t,df_373r_t,df_439mr_t,df_439r_t,df_456r_t,df_604_644r_t,df_604r_t,df_644mr_t,df_644r_t,df_687_721r_t,df_687r_t,df_688r_t,df_721mr_t,df_721r_t,df_746_1284r_t,df_746r_t]
    df_list_names = ['df_1284mr','df_1284r','df_1608mr','df_1608r','df_362r','df_373_439r','df_373r','df_439mr','df_439r','df_456r','df_604_644r','df_604r','df_644mr','df_644r','df_687_721r','df_687r','df_688r','df_721mr','df_721r','df_746_1284r','df_746r']
    return df_list_train,list_df_test,df_list_names

def concatenation_of_file(df_list_train,list_df_test):
    """ Concate train and test files
    args: df_list_train,list_df_test
    Return: list_df_concat"""
    list_df_concat = []
    for i in range(len(df_list_train)):
        for j in range(len(list_df_test)):
            if i==j:
                df = pd.concat([df_list_train[i],list_df_test[j]],axis=0)
                list_df_concat.append(df)
    return list_df_concat

def checking_missing_value(list_df_concat,df_list_names):
    """Checking missing value
    args:list_df_concat,df_list_names
    Return: print"""
    for i in range(len(list_df_concat)):
        for j in range(len(df_list_names)):
            if i==j:
                a=print(df_list_names[j])
                b=print(list_df_concat[i].isnull().sum())
                c=print('------------------------')
                d=print()
    return a,b,c,d

def data_splitting(list_df_concat):
    """Data splitting in dependent and independent variables
    args: list_df_concat
    Return: list_X,list_y"""
    list_X = []
    list_y = []

    for j in list_df_concat:
        X = j.drop('Outcome',axis=1)
        list_X.append(X)
        y = j.Outcome
        list_y.append(y)
    return list_X,list_y

def treating_missing_value(list_X):
    """ Treating missing value
    args: list_X
    Return: list_X1"""
    list_X1 = []

    for i in list_X:
        X = i.fillna(method='ffill',axis=1).fillna(method='bfill',axis=1)
        list_X1.append(X)
    return list_X1

def checking_imbalance_data(list_X1,list_y,df_list_names):
        """Checking imbalance data
        args: list_X,list_y,df_list_names
        Return: print"""
        for i in range(len(list_y)):
            for j in range(len(df_list_names)):
                if i==j:
                    p=print(df_list_names[j])
                    q=print(list_y[i].value_counts())
                    r=print('------------------------')
                    s=print()
        return p,q,r,s

def treating_imbalance_data(list_X1,list_y):
    """Treating imbalance data by using SMOTE
    args: list_X,list_y
    Return: list_X_resampled,list_y_resampled"""
    list_X_resampled = []
    list_y_resampled = []

    for i in range(len(list_X1)):
            for j in range(len(list_y)):
                if i == j:
                    from imblearn.over_sampling import SMOTE
                    sm = SMOTE()
                    X_resampled,y_resampled = sm.fit_resample(list_X1[i],list_y[j])
                    list_X_resampled.append(X_resampled)
                    list_y_resampled.append(y_resampled)
    return list_X_resampled,list_y_resampled

def scaling_data(list_X_resampled,list_y_resampled):
    """Scaling of data
    Args: list_X_resampled,list_y_resampled
    Return: list_Xscal"""
    list_Xscal = []

    for i in list_X_resampled:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scal = scaler.fit_transform(i)
        list_Xscal.append(X_scal)
    return list_Xscal

def application_pca(list_Xscal):
    """ Applying pca
    args: list_Xscal
    Return: list_Xn_comp,list_pca,list_PC_components"""
    list_Xn_comp = []
    list_pca = []
    list_PC_components = []
    for i in list_Xscal:
        pca = PCA(n_components = min(i.shape[0],int(len(pd.DataFrame(i).columns))))
        list_pca.append(pca)
        print(pca)
        X_n = pca.fit_transform(i)
        list_Xn_comp.append(X_n)
        PC_components = np.arange(pca.n_components_)
        list_PC_components.append(PC_components)
    return list_Xn_comp,list_pca,list_PC_components

def plot_scree_plot(list_Xn_comp,list_pca,list_PC_components):
    """Showing scree plot
    args: list_Xn_comp,list_pca,list_PC_components
    Return"""
    
    for j in range(len(list_Xn_comp)):
            for i in range(len(list_PC_components)):
                for k in range(len(list_pca)):
                    if i==k:

                        fig = sns.set(style='whitegrid', font_scale=1.2)
                        fig, ax = plt.subplots(figsize=(50, 7),num=21,clear=True)
                        fig = sns.barplot(x=list_PC_components[i], y=list_pca[k].explained_variance_ratio_, color='b')
                        fig = sns.lineplot(x=list_PC_components[i], y=np.cumsum(list_pca[k].explained_variance_ratio_), color='black', linestyle='-', linewidth=2, marker='o', markersize=8)
        
 
                        plt.axhline(y=0.9,color='r',linestyle='--')                    
                        plt.title('Scree Plot')
                        plt.xlabel('N-th Principal Component')
                        plt.ylabel('Variance Explained')
                        plt.ylim(0, 1)
                        plt.show()
    return