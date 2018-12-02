# Homework 5: Option 1 - predicting a person's “Empathy”
# Author: Fangfang Fu
# Netid: ffu3
# Date: 12/1/2018
# showTree function is cited from hw2

import pandas as pd
import numpy as np
from statistics import mode
from sklearn.model_selection import train_test_split
import sys

def showTree(dt, dictionary):
    left   = dt.tree_.children_left
    right  = dt.tree_.children_right
    thresh = dt.tree_.threshold
    feats  = [ dictionary[i] for i in dt.tree_.feature ]
    value  = dt.tree_.value
    def showTree_(node, s, depth):
        for i in range(depth-1):
            sys.stdout.write('|    ')
        if depth > 0:
            sys.stdout.write('-')
            sys.stdout.write(s)
            sys.stdout.write('-> ')
        if thresh[node] == -2: # leaf
            print("class {0}\t({1} for class 0, {2} for class 1)".format(np.argmax(value[node]), value[node][0,0], value[node][0,1]))
        else: # internal node
            print(feats[node]+"?")
            # print '%s?' % feats[node]
            showTree_(left[node], 'N', depth+1)
            showTree_(right[node], 'Y', depth+1)
    showTree_(0, '', 0)

def processData(filename):
    # get Yall
    df = pd.read_csv(filename, sep = ",", usecols=[94])
    # fill blank with mode of its column
    headers = list(df)
    df[headers]=df[headers].fillna(df.mode().iloc[0])
    # process the data to -1 and 1
    df.Empathy[df.Empathy < 4] = -1
    df.Empathy[df.Empathy >= 4] = 1
    Yall = df['Empathy'].values

    # get Xall
    df = pd.read_csv(filename, sep = ",", usecols=list(range(0,94)) + list(range(95,150)))
    # fill blank with mode of its column
    headers = list(df)
    df[headers]=df[headers].fillna(df.mode().iloc[0])
    # process the data to 0 and 1
    mapping = {'never smoked':1, 'tried smoking':2, 'former smoker':3, 'current smoker':4,   # Smoking
            'never':1, 'social drinker':4, 'drink a lot':4,   # Alcohol
            'i am always on time':1, 'i am often early':2, 'i am often running late':4,   # Punctuality
            'never':4, 'only to avoid hurting someone':2, 'sometimes':2, 'everytime it suits me':5,   # Lying
            'no time at all':1,'less than an hour a day':2, 'few hours a day':4, 'most of the day':5,   # Internet usage
            'female':1, 'male':4,   # Gender
            'right handed':1, 'left handed':4,   # Left - right handed
             'currently a primary school pupil':0,'primary school':1,'secondary school':2,'college/bachelor degree':4,'masters degree':5,'doctorate degree':6, # Education
            'no':1, 'yes':4,   # Only child
            'village':1, 'city':4,   # Village - town
            'block of flats':1, 'house/bungalow':4}    # House - block of flats
    df2 = df.applymap(lambda s: mapping.get(s) if s in mapping else s)
    df2['Age'] = np.where(df2['Age'] < 23, 1, 4)     # Age min =15, max = 30
    df2['Height'] = np.where(df2['Height'] <= 170, 1, 4)     # Height min = 62, max = 203, mode = 170
    df2['Weight'] = np.where(df2['Weight'] <= 60, 1, 4)     # Weight min = 41, max = 165, mode = 60
    df2['Number of siblings'] = np.where(df2['Number of siblings'] == 0, 1, 4)      # Number of siblings min = 0, max = 10, mode = 1
    Xall = np.where(df2 < 4, 0, 1) 
    return Xall,Yall,headers


class SurveyData:
    # preprocess the data
    Xall,Yall,headers = processData("data/responses.csv")       

    # Split the data - Train: 60%, Develop: 20%, Test: 20%
    Xtr,Xte,Ytr,Yte = train_test_split(Xall, Yall, test_size=0.2, random_state=0)
    Xtr,Xde,Ytr,Yde = train_test_split(Xtr, Ytr, test_size=0.25, random_state=0)

def perceptronProcessData(filename):
    # get Yall
    df = pd.read_csv(filename, sep = ",", usecols=[94])
    # fill blank with mode of its column
    headers = list(df)
    df[headers]=df[headers].fillna(df.mode().iloc[0])
    # process the data to -1 and 1
    df.Empathy[df.Empathy < 4] = -1
    df.Empathy[df.Empathy >= 4] = 1
    Yall = df['Empathy'].values

    # get Xall
    df = pd.read_csv(filename, sep = ",", usecols=list(range(0,94)) + list(range(95,150)))
    # fill blank with mode of its column
    headers = list(df)
    df[headers]=df[headers].fillna(df.mode().iloc[0])
    # process the data to 0 and 1
    mapping = {'never smoked':1, 'tried smoking':2, 'former smoker':3, 'current smoker':4,   # Smoking
            'never':1, 'social drinker':3, 'drink a lot':5,   # Alcohol
            'i am always on time':2, 'i am often early':1, 'i am often running late':3,   # Punctuality
            'never':4, 'only to avoid hurting someone':2, 'sometimes':2, 'everytime it suits me':5,   # Lying
            'no time at all':1,'less than an hour a day':2, 'few hours a day':3, 'most of the day':4,   # Internet usage
            'female':0, 'male':1,   # Gender
            'right handed':0, 'left handed':1,   # Left - right handed
            'currently a primary school pupil':0, 'primary school':1, 'secondary school':2, 'college/bachelor degree':3, 'masters degree':4, 'doctorate degree':5, # Education
            'no':0, 'yes':1,   # Only child
            'village':0, 'city':1,   # Village - town
            'block of flats':0, 'house/bungalow':1}    # House - block of flats
    df2 = df.applymap(lambda s: mapping.get(s) if s in mapping else s)
    Xall = df2.values
    return Xall,Yall,headers

class PerceptronSurveyData:
    # preprocess the data
    Xall,Yall,headers = perceptronProcessData("data/responses.csv")       

    # Split the data - Train: 60%, Develop: 20%, Test: 20%
    Xtr,Xte,Ytr,Yte = train_test_split(Xall, Yall, test_size=0.2, random_state=0)
    Xtr,Xde,Ytr,Yde = train_test_split(Xtr, Ytr, test_size=0.25, random_state=0)
