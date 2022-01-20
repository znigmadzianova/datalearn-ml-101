import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
sns.set()

from scipy import stats
from scipy.stats import norm

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

from termcolor import colored

def target_calculations(target):
    class_counts = target.value_counts()
    class_graph = class_counts.plot(x=target, kind='bar')
    print(colored('Class Value Counts: \n', attrs=['bold']), class_counts, '\n')
    print(colored('Class Counts Graph: \n', attrs=['bold']), class_graph)
    
def attributes_graph(df, target_name):
    val = int((len(df.columns) - 1) / 3)
    columns_list = df.columns.to_list()
    first_list = columns_list[:val]
    second_list = columns_list[val:val*2]
    third_list = columns_list[val*2:-1]
    first_list.insert(0, target_name)
    second_list.insert(0,target_name)
    third_list.insert(0,target_name)
    total_list = [first_list, second_list, third_list]
    for i in total_list:
        print(colored('Attributes: \n', attrs=['bold']), i)
        sns.pairplot(df[i], kind='scatter', hue=target_name, palette="Blues_d", plot_kws=dict(s=80, edgecolor='white'))
        plt.show()
        

def correlation_calculations(df, target_name):
    correlation_info = df.corr()
    print(colored('Correlation Visualization: \n', attrs=['bold']))
    correlation_info[target_name].sort_values().plot(kind='bar', figsize=(20,8))
    plt.show()
    print('\n')
    correlation_list = correlation_info.abs()[target_name].sort_values(ascending=False)
    print(colored('Correlation List: \n', attrs=['bold']), correlation_list)
    
    
def standartize_values(df):
    st_sc = StandardScaler()
    df_sc = st_sc.fit_transform(df)
    df_sc = pd.DataFrame(df_sc)
    return df_sc

def get_confusion_df(y_true, y_pred, class_names):
    conf_matrix = confusion_matrix(y_true, y_pred, labels=None)
    df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)
    return df_cm
    
def confusion_matrix_plot(y_true, y_pred, class_names):
    df_cm = get_confusion_df(y_true, y_pred, class_names)
    sns.heatmap(df_cm, annot=True, fmt="d", cmap = 'Blues', annot_kws = {'size': 15})
    plt.ylabel("Actual Class")
    plt.xlabel("Predicted Class")    
    plt.show()
    
def misclassification_plot(y_true, y_pred, class_names):
    df_cm = get_confusion_df(y_true, y_pred, class_names)
    for col in df_cm.columns:
        df_cm.at[col, col] = 0
    ax = df_cm.plot(kind="bar", title="Misclassified Classes")
    ax.set_xlabel("Classes")
    ax.set_ylabel("Number of Incorrectly Predicted Class")    
    plt.show()