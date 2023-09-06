#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 22:33:38 2022

@author: yvette
"""

# Method Description:
# In hw3, I used a hybrid recommend system of item-based CF and model-based CF.
# In the competition, I threw away the item-based CF and only used model-based CF which is more accurate
# To improve the rmse, I substracted more features from business.json and user.json and dropped the features substracted from the other describing json files because I found they may mislead the system
# I used the MinMaxScaler to process the features 
# To improve the model accuracy, I combined the xgboost and catboost and generated the average
# what's more, I adjusted the final result written to csv by round the results

# Error distribution:
    #>=0 and <1: 102577
    #>=1 and <2: 32532
    #>=2 and <3: 6121
    #>=3 and <4: 814
    #>=4: 0

# RMSE: 0.9760618070883263

# Execution Time:  532.0816044807434


from pyspark import SparkContext, SparkConf
import sys
import time 
import json
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import catboost as cbt
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

def trans_date_2_float(date):
    s = 0
    for item in date.split('-'):
        s += float(item)
    return s

def trans_2_ascii(string):
    s = 0
    for character in string:
        s += ord(character)
    return s

def trans_2_str(x, arg_1, arg_2):
    if arg_1 in x.keys():
        if x[arg_1] is not None:
            if arg_2 in x[arg_1].keys():
                if x[arg_1][arg_2] == 'True':
                    return 1
                if x[arg_1][arg_2] == 'False':
                    return 0
            else:
                return -1
        else:
            return -1
    else:
        return -1

def check_time_weekends(x):
    if 'hours' in x.keys():
        if x['hours'] is not None:
            s = 0
            if 'Saturday' in x['hours'].keys():
                s += 1
            if 'Sunday' in x['hours'].keys():
                s += 1
            return s
        else:
            return -1
    else:
        return -1

def check_time_weekdays(x):
    if 'hours' in x.keys():
        if x['hours'] is not None:
            s = 0
            if 'Monday' in x['hours'].keys():
                s += 1
            if 'Tuesday' in x['hours'].keys():
                s += 1
            if 'Wednesday' in x['hours'].keys():
                s += 1
            if 'Thursday' in x['hours'].keys():
                s += 1
            if 'Friday' in x['hours'].keys():
                s += 1
            return s
        else:
            return -1
    else:
        return -1

def trans_2_num(x, arg_1, arg_2):
    if arg_1 in x.keys():
        if x[arg_1] is not None:
            if arg_2 in x[arg_1].keys():
                return x[arg_1][arg_2]
            else:
                return -1
        else:
            return -1
    else:
        return -1

def create_dict(l):
    d ={}
    for i in l:
        d[i] = {}
    return d

def inner_add(a, b):
    x = a[0] + b[0]
    y = a[1] + b[1]
    return (x, y)

def get_dict_len(dic):
    try:
        return len(list(dic.items()))
    except:
        return 0

def get_str_len(string):
    
    try:
        s = 0
        for item in string.split(','):
            for i in item:
                s+=ord(i)
        return s   
    except:
        return 0


def input_1_feature(b_dict, f_list, keyname):
    for f in f_list:
        try:
            b_dict[f[0]][keyname] = f[1]
        except:
            pass 
    return b_dict

def input_2_feature(b_dict, f_list, keyname_1, keyname_2):
    for f in f_list:
        try:
            b_dict[f[0]][keyname_1] = f[1]
            b_dict[f[0]][keyname_2] = f[2]
        except:
            pass 
    return b_dict

def append_train_data(train):
    temp = []
    for item in train:
        r = {}
        r.update(item[2])
        r.update(item[3])
        r["real_rates"] = item[4]
        temp.append(r)
    return temp

def append_test_data(test):
    temp = []
    for item in test:
        r = {}
        r.update(item[2])
        r.update(item[3])
        temp.append(r)
    return temp

def create_new_frame_1(x, user_dict, busi_dict):
    a = x[0]
    b = x[1]
    c = user_dict[x[0]]
    d = busi_dict[x[1]]
    e = float(x[2])
    return (a, b, c, d, e)

def create_new_frame_2(x, user_dict, busi_dict):
    a = x[0]
    b = x[1]
    c = user_dict[x[0]]
    d = busi_dict[x[1]]
    return (a, b, c, d)

def zip_2_one(list_1,list_2):
    new_list = []
    for i, j in zip(list_1,list_2):
        new_list.append((i,j))
    return new_list

def write_csv(file,pre_list):
    with open(file, 'w') as f:
        f.write("user_id,business_id,prediction")
        for item in pre_list:
            if item[1] > 5:
                f.write('\n'+ str(item[0][0]) +"," + str(item[0][1]) + "," + str(5.0))
            elif item[1] < 1:
                f.write('\n'+ str(item[0][0]) +"," + str(item[0][1]) + "," + str(1.0))
            else:
                f.write('\n'+ str(item[0][0]) +"," + str(item[0][1]) + "," + str(item[1]))
        f.close()
    
if __name__ == "__main__":
    #time start
    begin = time.time()
    
    #import files local
    folder_file = "/Users/yvette/Desktop/data/"
    test_file = "/Users/yvette/Desktop/data/yelp_val.csv"
    output_file = "/Users/yvette/Desktop/output.csv"
    
    #import files sys
    #folder_file = sys.argv[1]
    #test_file = sys.argv[2]
    #output_file = sys.argv[3]
    
    #path name
    train_file = folder_file + 'yelp_train.csv'
    user_file = folder_file + 'user.json'
    busi_file = folder_file + 'business.json'
    
    #create spark
    sc = SparkContext.getOrCreate()
    sc.setLogLevel("ERROR")
    
    #read train rdd
    train_textfile = sc.textFile(train_file, 30)
    train_title = train_textfile.first()
    train_rdd = train_textfile.filter(lambda x: x != train_title).map(lambda x: (x.split(",")[0], x.split(",")[1], x.split(",")[2]))
    
    #read test rdd
    test_textfile = sc.textFile(test_file, 30)
    test_title = test_textfile.first()
    test_rdd = test_textfile.filter(lambda x: x != test_title).map(lambda x: (x.split(",")[0], x.split(",")[1]))
    #print(test_rdd.take(10))
    
    #train list
    user_train_list = train_rdd.map(lambda x: x[0]).distinct().collect()
    busi_train_list = train_rdd.map(lambda x: x[1]).distinct().collect()
    
    #test list
    user_test_list = test_rdd.map(lambda x: x[0]).distinct().collect()
    busi_test_list = test_rdd.map(lambda x: x[1]).distinct().collect()
    
    #user list
    user_list = list(set(user_train_list).union(set(user_test_list)))
    user_dict = create_dict(user_list)
    #print(user_list[1])
    
    #busi list
    busi_list = list(set(busi_train_list).union(set(busi_test_list)))
    busi_dict = create_dict(busi_list)
    
    ### collect feature ####
    # business
    busi_rdd_feature= sc.textFile(busi_file, 30).map(lambda x: json.loads(x))
    
    busi_rdd_1 = busi_rdd_feature.map(lambda x: (x["business_id"], x["stars"], x["review_count"]))
    busi_dict = input_2_feature(busi_dict, busi_rdd_1.collect(), "busi_stars", "busi_review_count")
    
    busi_rdd_2 = busi_rdd_feature.map(lambda x: (x["business_id"], x["latitude"], x["longitude"]))
    busi_dict = input_2_feature(busi_dict, busi_rdd_2.collect(), "busi_latitude", "busi_longitude")
    
    busi_rdd_3 = busi_rdd_feature.map(lambda x: (x["business_id"], trans_2_ascii(x['neighborhood']), x["is_open"]))
    busi_dict = input_2_feature(busi_dict, busi_rdd_3.collect(), "busi_neighbor","busi_open")
    
    busi_rdd_4 = busi_rdd_feature.map(lambda x: (x["business_id"], get_dict_len(x['attributes']), get_str_len(x['categories'])))
    busi_dict = input_2_feature(busi_dict, busi_rdd_4.collect(), "busi_attributes", "busi_categories")
    
    busi_rdd_5 = busi_rdd_feature.map(lambda x: (x["business_id"], trans_2_ascii(x['city']), trans_2_ascii(x['state'])))
    busi_dict = input_2_feature(busi_dict, busi_rdd_5.collect(), "busi_city", "busi_state")
    
    busi_rdd_6 = busi_rdd_feature.map(lambda x: (x["business_id"], trans_2_ascii(x['address'])))
    busi_dict = input_1_feature(busi_dict, busi_rdd_6.collect(), "busi_address")
    
    busi_rdd_7 = busi_rdd_feature.map(lambda x: (x["business_id"], trans_2_str(x,'attributes','GoodForKids'), trans_2_str(x,'attributes','RestaurantsGoodForGroups')))
    busi_dict = input_2_feature(busi_dict, busi_rdd_7.collect(), "busi_attributes_1", "busi_categories_2")
    
    busi_rdd_8 = busi_rdd_feature.map(lambda x: (x["business_id"], trans_2_str(x,'attributes','RestaurantsTableService'), trans_2_str(x,'attributes','RestaurantsReservations')))
    busi_dict = input_2_feature(busi_dict, busi_rdd_8.collect(), "busi_attributes_3", "busi_categories_4")
    
    busi_rdd_9 = busi_rdd_feature.map(lambda x: (x["business_id"], trans_2_str(x,'attributes','RestaurantsDelivery'), trans_2_str(x,'attributes','RestaurantsTakeOut')))
    busi_dict = input_2_feature(busi_dict, busi_rdd_9.collect(), "busi_attributes_5", "busi_categories_6")
    
    busi_rdd_10 = busi_rdd_feature.map(lambda x: (x["business_id"], trans_2_num(x,'attributes','RestaurantsPriceRange2'), get_dict_len(x['hours'])))
    busi_dict = input_2_feature(busi_dict, busi_rdd_10.collect(), "busi_attributes_7", "busi_attributes_8")
    
    #print("Duration_busi:", time.time() - begin)
    
    # uesr
    user_rdd_feature = sc.textFile(user_file, 30).map(lambda x: json.loads(x))
    
    user_rdd_1 = user_rdd_feature.map(lambda x: (x["user_id"], x["average_stars"], x["review_count"]))
    user_dict = input_2_feature(user_dict, user_rdd_1.collect(), "average_stars", "review_count")
    
    user_rdd_2 = user_rdd_feature.map(lambda x: (x["user_id"], x["useful"], x["funny"]))
    user_dict = input_2_feature(user_dict, user_rdd_2.collect(), "useful", "funny")
    
    user_rdd_3 = user_rdd_feature.map(lambda x: (x["user_id"], x["compliment_note"], x["compliment_plain"]))
    user_dict = input_2_feature(user_dict, user_rdd_3.collect(), "compliment_note", "compliment_plain")

    user_rdd_4 = user_rdd_feature.map(lambda x: (x["user_id"], x["compliment_hot"], x["compliment_more"]))
    user_dict = input_2_feature(user_dict, user_rdd_4.collect(), "compliment_hot", "compliment_more")
    
    user_rdd_5 = user_rdd_feature.map(lambda x: (x["user_id"], x["compliment_cute"], x["compliment_list"]))
    user_dict = input_2_feature(user_dict, user_rdd_5.collect(), "compliment_cute", "compliment_list")
    
    user_rdd_6 = user_rdd_feature.map(lambda x: (x["user_id"], x["compliment_cool"], x["compliment_funny"]))
    user_dict = input_2_feature(user_dict, user_rdd_6.collect(), "compliment_cool", "compliment_funny")
    
    user_rdd_7 = user_rdd_feature.map(lambda x: (x["user_id"], x["compliment_writer"], x["compliment_photos"]))
    user_dict = input_2_feature(user_dict, user_rdd_7.collect(), "compliment_writer", "compliment_photos")
    
    user_rdd_8 = user_rdd_feature.map(lambda x: (x["user_id"], get_str_len(x["friends"]), get_str_len(x["elite"])))
    user_dict = input_2_feature(user_dict, user_rdd_8.collect(), "friends", "elite")
    
    user_rdd_9 = user_rdd_feature.map(lambda x: (x["user_id"], x["cool"], x["fans"]))
    user_dict = input_2_feature(user_dict, user_rdd_9.collect(), "cool", "fans")
    
    user_rdd_10 = user_rdd_feature.map(lambda x: (x["user_id"], trans_date_2_float(x["yelping_since"])))
    user_dict = input_1_feature(user_dict, user_rdd_10.collect(), "user_date")
    
    #user_rdd_feature.unpersist()

    print("Duration_feature:", time.time() - begin)
    ### end feature ####
    
    #make train data
    train_1 = train_rdd.map(lambda x: (create_new_frame_1(x, user_dict, busi_dict))).collect()
    train_2 = train_rdd.map(lambda x: (create_new_frame_2(x, user_dict, busi_dict))).collect()
    test = test_rdd.map(lambda x: (create_new_frame_2(x, user_dict, busi_dict))).collect()
    
    train_data_1 = pd.DataFrame().from_dict(append_train_data(train_1))
    index_1 = train_data_1.index
    column_1 = train_data_1.columns
    train_data_2 = pd.DataFrame().from_dict(append_test_data(train_2))
    index_2 = train_data_2.index
    column_2 = train_data_2.columns
    test_data = pd.DataFrame().from_dict(append_test_data(test))
    test_index = test_data.index
    test_column = test_data.columns
    
    train_data_no_label_scaled = pd.DataFrame(MinMaxScaler().fit_transform(train_data_2), index= index_2, columns=column_2)
    train_data_no_label_scaled["real_rates"] = train_data_1["real_rates"]
    train_data = train_data_no_label_scaled.fillna(-1)
    test_data = pd.DataFrame(MinMaxScaler().fit_transform(test_data), index=test_index, columns=test_column).fillna(-1)

    ## train xgboost regressor
    xgbr = xgb.XGBRegressor(objective='reg:linear', n_estimators=325, learning_rate=0.07, \
                            max_depth=8, subsample=1.0, random_state=555, min_child_weight=4, \
                                reg_alpha=0.5, reg_lambda=0.5, colsample_bytree=0.7)
    #xgbr.fit(train_data.loc[:, train_data.columns != 'real_rates'], train_data["real_rates"])
    #print("train Duration : ", time.time() - begin)
    xgbr.fit(train_data.loc[:, train_data.columns != 'real_rates'], train_data["real_rates"])
    
    cbtr = cbt.CatBoostRegressor(n_estimators=525, depth=10, learning_rate=0.1, loss_function='RMSE', random_state=555, silent=True)
    #xgbr.fit(train_data.loc[:, train_data.columns != 'real_rates'], train_data["real_rates"])
    #print("train Duration : ", time.time() - begin)
    cbtr.fit(train_data.loc[:, train_data.columns != 'real_rates'], train_data["real_rates"])
    
    ## predict test_data
    predictions_1 = xgbr.predict(test_data)
    predictions_2 = cbtr.predict(test_data)
    predictions = []
    for (i_1, i_2) in zip(predictions_1, predictions_2):
        predictions.append((i_1 +i_2)/2)
    #predictions = xgbr.predict(test_data)

    #write to csv
    write_csv(output_file, zip_2_one(test, predictions))  
    print("Duration : ", time.time() - begin)