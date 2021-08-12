
# -*- coding: utf-8 -*-
"""
Mode Choice Predictor 1.02 
"""
import pickle
import numpy as np
def load():
    with open("data/mini.pkl", 'rb') as open_file:
        mini = pickle.load(open_file)
    with open("data/macro.pkl", 'rb') as open_file:
        macro = pickle.load(open_file)
    with open("data/micro.pkl", 'rb') as open_file:
        micro = pickle.load(open_file)
    return [mini,macro,micro]
   
    
    
def predict(features,clf,printresult):
    features = np.array(features)
    if features.ndim == 1:
        predi=clf.predict_proba(features.reshape((1,-1)))
        if printresult: print_predi(predi[0])
        return predi[0]

    elif features.ndim == 2:
        predi=clf.predict_proba(features)

        if printresult: print_predia(predi)
        return predi
    else:
        raise ValueError("Features can be a 1 or 2 dimensional array")



def predict_mini(features,printresult=False):
    """
    features can be a single set of features
    or an array of feature sets
    features need to be formatted like this:
    [trip length,area,trip reason]
    
    trip length:
        1: < 0.5km
        2: < 1 km
        3: < 2.5 km
        4: < 5 km
        5: < 10 km
        6: < 20 km
        7: < 50 km
        8: > 50 km
    area:
        1: big city (>1 mio)
        2: city
        3: suburban
        4: rural
    trip reason:
        10: commute to work
        20: business
        30: school/education
        40: bring/take somebody
        50: shopping
        60: private activity
        70: other leisure
        80: private meeting
        90: other reason

    """
    [mini,macro,micro]=load()
    return predict(features,mini,printresult)      
    
def predict_macro(features,printresult=False):
    """
    features can be a single set of features
    or an array of feature sets
    features need to be formatted like this:
    [trip length,area,age,economic situation,job,distance to public transport]
    
    trip length:
        1: < 0.5km
        2: < 1 km
        3: < 2.5 km
        4: < 5 km
        5: < 10 km
        6: < 20 km
        7: < 50 km
        8: > 50 km
    area:
        1: big city (>1 mio)
        2: city
        3: suburban
        4: rural
    age:
        1: 6-14
        2: 15-19
        3: 20-24
        4: 25-34
        5: 35-44
        6: 45-54
        7: 55-64
        8: > 65
    economic situation:
        1: very bad
        2: bad
        3: average
        4: good
        5: very good
    job:
        1: pupil
        2: employed
        3: retired
        4: other
    distance to public transport:
        1:    <= 5  minutes by foot
        2:  6 - 15  minutes by foot
        3: 16 - 30  minutes by foot
        4: 31 - 60  minutes by foot
        5: 61 - 120 minutes by foot
        6:    > 120 minutes by foot

    """
    [mini,macro,micro]=load()
    return predict(features,macro,printresult)
    
def predict_micro(features,printresult=False):
    """
    features can be a single set of features
    or an array of feature sets
    features need to be formatted like this:
    [trip length,area,age,economic situation,job,distance to public transport,
     trip reason, car availability, gender]
    
    trip length:
        1: < 0.5km
        2: < 1 km
        3: < 2.5 km
        4: < 5 km
        5: < 10 km
        6: < 20 km
        7: < 50 km
        8: > 50 km
    area:
        1: big city (>1 mio)
        2: city
        3: suburban
        4: rural
    age:
        1: 6-14
        2: 15-19
        3: 20-24
        4: 25-34
        5: 35-44
        6: 45-54
        7: 55-64
        8: > 65
    economic situation:
        1: very bad
        2: bad
        3: average
        4: good
        5: very good
    job:
        1: pupil
        2: employed
        3: retired
        4: other
    distance to public transport:
        1:    <= 5  minutes by foot
        2:  6 - 15  minutes by foot
        3: 16 - 30  minutes by foot
        4: 31 - 60  minutes by foot
        5: 61 - 120 minutes by foot
        6:    > 120 minutes by foot
    trip reason:
        10: commute to work
        20: business
        30: school/education
        40: bring/take somebody
        50: shopping
        60: private activity
        70: other leisure
        80: private meeting
        90: other reason
    car availability:
        1: always
        2: sometimes
        3: never
    gender:
        1: male
        2: female

    """
    [mini,macro,micro]=load()
    return predict(features,micro,printresult)



def print_predi(predi):
    print("Mode choice:")
    print("foot:      ",round(predi[0]*100,2),"%")
    print("bike:      ",round(predi[1]*100,2),"%")
    print("driver:    ",round(predi[2]*100,2),"%")
    print("passenger: ",round(predi[3]*100,2),"%")
    print("public:    ",round(predi[4]*100,2),"%")
def print_predia(predi):
    print("Mode choices: (foot%/bike%/passenger%/driver%/public%)")
    for pred in predi:
        print(round(pred[0]*100,1),"/",round(pred[1]*100,1),"/",
        round(pred[2]*100,1),"/",round(pred[3]*100,1),"/",
        round(pred[4]*100,1))
 