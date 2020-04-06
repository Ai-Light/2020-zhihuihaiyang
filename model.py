#!/usr/bin/env python
# coding: utf-8

import gc
import pandas as pd
import numpy as np
import os
import time
import lightgbm as lgb
from copy import deepcopy
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
import warnings
from glob import glob
from scipy.sparse import csr_matrix


start_t = time.time()
print('ww_900_start')
pd.set_option('display.max_columns', 100)
warnings.filterwarnings('ignore')

def group_feature(df, key, target, aggs,flag):   
    agg_dict = {}
    for ag in aggs:
        agg_dict['{}_{}_{}'.format(target,ag,flag)] = ag
    print(agg_dict)
    t = df.groupby(key)[target].agg(agg_dict).reset_index()
    return t


def haversine_dist(lat1,lng1,lat2,lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    radius = 6371  # Earth's radius taken from google
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat/2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng/2) ** 2
    h = 2 * radius * np.arcsin(np.sqrt(d))
    return h


def extract_feature(df, train, flag):
#     # speed split    
#     date_nunique = df.groupby(['ship'])['speed_cat'].nunique().to_dict()
#     train['speed_cat_nunique'] = train['ship'].map(date_nunique) 
    '''
    统计feature
    '''
    if (flag == 'on_night') or (flag == 'on_day'): 
        t = group_feature(df, 'ship','speed',['max','mean','median','std','skew'],flag)
        train = pd.merge(train, t, on='ship', how='left')
        # return train
    
    
    if flag == "0":
        t = group_feature(df, 'ship','direction',['max','median','mean','std','skew'],flag)
        train = pd.merge(train, t, on='ship', how='left')  
    elif flag == "1":
        t = group_feature(df, 'ship','speed',['max','mean','median','std','skew'],flag)
        train = pd.merge(train, t, on='ship', how='left')
        t = group_feature(df, 'ship','direction',['max','median','mean','std','skew'],flag)
        train = pd.merge(train, t, on='ship', how='left') 
        hour_nunique = df.groupby('ship')['speed'].nunique().to_dict()
        train['speed_nunique_{}'.format(flag)] = train['ship'].map(hour_nunique)   
        hour_nunique = df.groupby('ship')['direction'].nunique().to_dict()
        train['direction_nunique_{}'.format(flag)] = train['ship'].map(hour_nunique)  

    t = group_feature(df, 'ship','x',['max','min','mean','median','std','skew'],flag)
    train = pd.merge(train, t, on='ship', how='left')
    t = group_feature(df, 'ship','y',['max','min','mean','median','std','skew'],flag)
    train = pd.merge(train, t, on='ship', how='left')
    t = group_feature(df, 'ship','base_dis_diff',['max','min','mean','std','skew'],flag)
    train = pd.merge(train, t, on='ship', how='left')

       
    train['x_max_x_min_{}'.format(flag)] = train['x_max_{}'.format(flag)] - train['x_min_{}'.format(flag)]
    train['y_max_y_min_{}'.format(flag)] = train['y_max_{}'.format(flag)] - train['y_min_{}'.format(flag)]
    train['y_max_x_min_{}'.format(flag)] = train['y_max_{}'.format(flag)] - train['x_min_{}'.format(flag)]
    train['x_max_y_min_{}'.format(flag)] = train['x_max_{}'.format(flag)] - train['y_min_{}'.format(flag)]
    train['slope_{}'.format(flag)] = train['y_max_y_min_{}'.format(flag)] / np.where(train['x_max_x_min_{}'.format(flag)]==0, 0.001, train['x_max_x_min_{}'.format(flag)])
    train['area_{}'.format(flag)] = train['x_max_x_min_{}'.format(flag)] * train['y_max_y_min_{}'.format(flag)]

    # train['dis_lng_{}'.format(flag)] = list(map(haversine_dist,train['x_max_{}'.format(flag)],train['y_max_{}'.format(flag)],train['x_min_{}'.format(flag)],train['y_min_{}'.format(flag)]))       
    
    mode_hour = df.groupby('ship')['hour'].agg(lambda x:x.value_counts().index[0]).to_dict()
    train['mode_hour_{}'.format(flag)] = train['ship'].map(mode_hour)
    train['slope_median_{}'.format(flag)] = train['y_median_{}'.format(flag)] / np.where(train['x_median_{}'.format(flag)]==0, 0.001, train['x_median_{}'.format(flag)])

    return train

def get_data(files, is_sort=True, sort_column="time"):
    datas = [pd.read_csv(f) for f in files]
    if is_sort:
        dfs = [df.sort_values(by=sort_column, ascending=True, na_position='last') for df in datas]
    df = pd.concat(datas, axis=0, ignore_index=True)
    return df


def extract_dt(df):
    df['time'] = pd.to_datetime(df['time'], format='%m%d %H:%M:%S')
    df['date'] = df['time'].dt.date
    df['hour'] = df['time'].dt.hour

    df['x_dis_diff'] = (df['x'] - 6165599).abs()
    df['y_dis_diff'] = (df['y'] - 5202660).abs()
    df['base_dis_diff'] = ((df['x_dis_diff']**2)+(df['y_dis_diff']**2))**0.5    
    del df['x_dis_diff'],df['y_dis_diff']    
    
    df["x"] = df["x"] / 1e6
    df["y"] = df["y"] / 1e6    
    df['day_nig'] = 0
    df.loc[(df['hour'] > 5) & (df['hour'] < 20),'day_nig'] = 1
    return df

train_files = glob("tcdata/hy_round2_train_20200225/*.csv")
test_files = glob("tcdata/hy_round2_testB_20200312/*.csv")
train_files = sorted(train_files)
test_files = sorted(test_files)


def get_data(files, is_sort=True, sort_column="time"):
    datas = [pd.read_csv(f) for f in files]
    if is_sort:
        dfs = [df.sort_values(by=sort_column, ascending=True, na_position='last') for df in datas]
    df = pd.concat(datas, axis=0, ignore_index=True)
    return df

train = get_data(train_files)
train.columns = ['ship','x','y','speed','direction','time','type']
test = get_data(test_files)
test.columns = ['ship','x','y','speed','direction','time']

train = extract_dt(train)
test = extract_dt(test)
train_label = train.drop_duplicates(['ship'],keep = 'first')
test_label = test.drop_duplicates(['ship'],keep = 'first')
train_label['type'] = train_label['type'].map({'围网':0,'刺网':1,'拖网':2})

num = train_label.shape[0]
data_label = train_label.append(test_label)
data =train.append(test)

data_1 = data[data['speed']==0]
data_2 = data[data['speed']!=0]
data_label = extract_feature(data_1, data_label,"0")
data_label = extract_feature(data_2, data_label,"1")

data_1 = data[data['day_nig'] == 0]
data_2 = data[data['day_nig'] == 1]
data_label = extract_feature(data_1, data_label,"on_night")
data_label = extract_feature(data_2, data_label,"on_day")

if os.path.isfile('nmf_testb.csv'):
    nmf_fea = pd.read_csv('nmf_testb.csv')
    data_label = data_label.merge(nmf_fea,on='ship',how = 'left')
    del nmf_fea
else:
    for j in range(1,4):
        print('********* {} *******'.format(j))
        for i in ['speed','x','y']:
            data[i + '_str'] = data[i].astype(str)
            from nmf_list import nmf_list
            nmf = nmf_list(data,'ship',i + '_str',8,2)
            nmf_a = nmf.run(j)
            data_label = data_label.merge(nmf_a,on = 'ship',how = 'left')



first = "0"
second = "1"
data_label['direction_median_ratio'] = data_label['direction_median_{}'.format(second)] / data_label['direction_median_{}'.format(first)]
data_label['slope_ratio'] = data_label['slope_{}'.format(second)] / data_label['slope_{}'.format(first)] 
data_label['slope_mean_ratio'] = data_label['slope_median_{}'.format(second)] / data_label['slope_median_{}'.format(first)]

first = "on_night"
second = "on_day"
data_label['speed_median_ratio'] = data_label['speed_median_{}'.format(second)] / data_label['speed_median_{}'.format(first)] 
data_label['speed_std_ratio'] = data_label['speed_std_{}'.format(second)] / data_label['speed_std_{}'.format(first)] 
# data_label['lat_lng_ratio'] = data_label['dis_lng_{}'.format(second)] / data_label['dis_lng_{}'.format(first)] 
'''
count feature
'''
flag = 'all'
for cc in ['direction','speed']:
    t = group_feature(data_label,cc, 'ship',['count'],flag +cc+ 'x')
    data_label = pd.merge(data_label, t, on=cc, how='left')  

for i in ["0","1"]:
    if i == "1":
        for j in [
#                 'slope_speed_cat_nunique_{}'.format(i),
#                   'slope_mean_speed_cat_nunique_{}'.format(i),
                  'speed_nunique_{}'.format(i),
                  'direction_nunique_{}'.format(i)
                 ]:
            
            t = group_feature(data_label,j, 'ship',['count'],j+"_count")
            data_label = pd.merge(data_label, t, on=j, how='left') 
    for j in [
           'slope_median_{}'.format(i),
#               'x_max_x_min_{}'.format(i),
#               'y_max_y_min_{}'.format(i)
             ]:
#         t = group_feature(data_label,j, 'ship',['count'],j+"_count")
#         data_label = pd.merge(data_label, t, on=j, how='left') 
        t = group_feature(data_label,j, 'speed',['min','max','median','std','skew'],j+"_tongji")
        data_label = pd.merge(data_label, t, on=j, how='left')
        # t = group_feature(data_label,j, 'direction',['min','max','median','std','skew'],j+"_tongji")
        # data_label = pd.merge(data_label, t, on=j, how='left')

def cut_bins(raw_data, col_name=None, q=49):
    features, bins = pd.qcut(raw_data[col_name], q=q, retbins=True, duplicates="drop")
    labels = list(range(len(bins) - 1))
    features, bins = pd.qcut(raw_data[col_name], labels=labels, q=q, retbins=True, duplicates="drop")
    return features, bins, labels


MAX_CATE = 199
data["x_cate"], x_bins, x_labels = cut_bins(data, col_name="x", q=MAX_CATE)
data["y_cate"], y_bins, y_labels = cut_bins(data, col_name="y", q=MAX_CATE)
# data["x_sub_y_cate"], x_sub_y_bins, x_sub_y_labels = cut_bins(data, col_name="x_sub_y", q=MAX_CATE)
data["distance_cate"], dist_bins, dist_labels = cut_bins(data, col_name="base_dis_diff", q=MAX_CATE)

data["speed_cate"], speed_bins, speed_labels = cut_bins(data, col_name="speed", q=MAX_CATE)

MAX_CATE = 120
data["direct_cate"], speed_bins, speed_labels = cut_bins(data, col_name="direction", q=MAX_CATE)

if os.path.isfile('emb_testb.csv'):
    w2v_fea = pd.read_csv('emb_testb.csv')
    data_label = data_label.merge(w2v_fea, on='ship', how='left')
    del w2v_fea
else:
    from gensim.models import Word2Vec
    import gc
    def emb(df, f1, f2):
        emb_size = 23
        print('====================================== {} {} ======================================'.format(f1, f2))
        tmp = df.groupby(f1, as_index=False)[f2].agg({'{}_{}_list'.format(f1, f2): list})
        sentences = tmp['{}_{}_list'.format(f1, f2)].values.tolist()
        del tmp['{}_{}_list'.format(f1, f2)]
        for i in range(len(sentences)):
            sentences[i] = [str(x) for x in sentences[i]]
        model = Word2Vec(sentences, size=emb_size, window=5, min_count=3, sg=0, hs=1, seed=2222)
        emb_matrix = []
        for seq in sentences:
            vec = []
            for w in seq:
                if w in model:
                    vec.append(model[w])
            if len(vec) > 0:
                emb_matrix.append(np.mean(vec, axis=0))
            else:
                emb_matrix.append([0] * emb_size)
        emb_matrix = np.array(emb_matrix)
        for i in range(emb_size):
            tmp['{}_{}_emb_{}'.format(f1, f2, i)] = emb_matrix[:, i]
        del model, emb_matrix, sentences
        return tmp


    emb_cols = [
        ['ship', 'x_cate'],
        ['ship', 'y_cate'],
        ['ship', 'speed_cate'],
        ['ship', 'distance_cate'],
        # ['ship', 'direct_cate'],
    ]
    for f1, f2 in emb_cols:
        data_label = data_label.merge(emb(data, f1, f2), on=f1, how='left')

    gc.collect()

    # emb_list = ['ship']
    # for i in data_label.columns:
    #     if '_emb_' in i:
    #         emb_list.append(i)

    # data_label[emb_list].to_csv('emb_testb.csv',index=False)


print('feature done')

train_label = data_label[:num]
test_label = data_label[num:]
features = [x for x in train_label.columns if x not in ['ship','type','time','x','y','diff_time','date','day_nig','direction','speed','hour',
                                                       'speed_many','dire_diff','direction_str','speed_str','dis','x_speed','y_speed'] ]
target = 'type'
# print(len(features), ','.join(features))


from feature_selector import FeatureSelector
fs = FeatureSelector(data = train_label[features], labels = train_label[target])
fs.identify_zero_importance(task = 'classification', eval_metric = 'multiclass',
                            n_iterations = 10, early_stopping = True)
fs.identify_low_importance(cumulative_importance = 0.97)
low_importance_features = fs.ops['low_importance']
print('====low_importance_features=====')
print(low_importance_features)
for i in low_importance_features:
    features.remove(i)



print('feature number',len(features))
gc.collect()



def macro_f1(y_hat, data):
    y_true = data.get_label()
    y_hat = y_hat.reshape(-1, y_true.shape[0])
    y_hat = np.argmax(y_hat, axis=0)
    f1_multi = precision_recall_fscore_support(y_true, y_hat, labels=[0, 1, 2])[2]
    f1_macro =  f1_score(y_true, y_hat, average ="macro")
    assert np.mean(f1_multi) ==  f1_macro
    return 'f1', f1_macro, True


def f1_single(y_hat, data, index=0):
    y_true = data.get_label()
    y_hat = y_hat.reshape(-1, y_true.shape[0])
    y_hat = np.argmax(y_hat, axis=0)
    f1_multi = precision_recall_fscore_support(y_true, y_hat, labels=[0, 1, 2])[2]
    f1_s = round(f1_multi[index], 4)
    return 'f1_{}'.format(index), f1_s, True


train_X = train_label[features]
test_X = test_label[features]
print(train_X.shape, test_X.shape)
train_y = train_label[target]


params = {
        'task':'train', 
        'num_leaves': 63,
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'None', # [f1_0, f1_1, f1_2],
        'min_data_in_leaf': 10,
        'learning_rate': 0.01,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.95,
        'early_stopping_rounds': 2000,
#         'lambda_l1': 0.1,
#         'lambda_l2': 0.1,
        "first_metric_only": True,
        'bagging_freq': 3, 
        'max_bin': 255,
        'random_state': 42,
        'verbose' : -1
    }

    
models = []
test_preds = []
val_preds = []
oof_seed = np.zeros((len(train_label), 3))
seed = [2222,2018778]
for j in seed:
    print("+++++++++++++++++ seed {} ++++++++++++".format(str(j)))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=j)
    oof = np.zeros((len(train_label), 3))
    for i, (trn_idx, val_idx) in enumerate(skf.split(train_X, train_y)):
        print("-" * 81)
        print("[!] fold {}".format(i))
        lgb_params = deepcopy(params)
        # print(lgb_params)
        trn_X = csr_matrix(train_X)[trn_idx]
        trn_y = train_y.iloc[trn_idx]
        val_X = csr_matrix(train_X)[val_idx]
        val_y = train_y.iloc[val_idx]
        dtrain = lgb.Dataset(trn_X, trn_y) 
        dval = lgb.Dataset(val_X, val_y) 
        model = lgb.train(lgb_params, 
               dtrain, 
               num_boost_round=400000,
               valid_sets=[dval], 
               feval=lambda preds, train_data: [
                   macro_f1(preds, train_data),
                   f1_single(preds, train_data, index=0),
                   f1_single(preds, train_data, index=1),
                   f1_single(preds, train_data, index=2)],
               verbose_eval=-1)
        models.append(model)
        # print(model.best_iteration)
        val_pred = model.predict(val_X, iteration=model.best_iteration)
        oof[val_idx] = val_pred
        val_y = train_y.iloc[val_idx]
        val_pred = np.argmax(val_pred, axis=1)
        print(str(i), 'val f1', metrics.f1_score(val_y, val_pred, average='macro'))
        test_preds.append(model.predict(test_X, iteration=model.best_iteration))
        print("[!] fold {} finish\n".format(i))
        del dtrain, dval
        gc.collect()
    val_pred = np.argmax(oof, axis=1)
    print(str(j), 'every_flod val f1', metrics.f1_score(train_y, val_pred, average='macro'))
    oof_seed += oof/len(seed)

oof1 = np.argmax(oof_seed, axis=1)
print('oof f1', metrics.f1_score(oof1,train_y, average='macro'))
val_score = np.round(metrics.f1_score(oof1, train_y, average='macro'),6)

def ensemble_predictions(predictions, weights=None, type_="linear"):
    if not weights:
        print("[!] AVE_WGT")
        weights = [1./ len(predictions) for _ in range(len(predictions))]
    assert len(predictions) == len(weights)
    if np.sum(weights) != 1.0:
        weights = [w / np.sum(weights) for w in weights]
    print("[!] weights = {}".format(weights))
    assert np.isclose(np.sum(weights), 1.0)
    if type_ == "linear":
        res = np.average(predictions, weights=weights, axis=0)
    elif type_ == "harmonic":
        res = np.average([1 / p for p in predictions], weights=weights, axis=0)
        return 1 / res
    elif type_ == "geometric":
        numerator = np.average(
            [np.log(p) for p in predictions], weights=weights, axis=0
        )
        res = np.exp(numerator / sum(weights))
        return res
    elif type_ == "rank":
        from scipy.stats import rankdata
        res = np.average([rankdata(p) for p in predictions], weights=weights, axis=0)
        return res / (len(res) + 1)
    return res

def merge(prob, number=-1, index=0):
    from copy import deepcopy
    new_prob = deepcopy(prob)
    top = np.argsort(prob[:, index])[::-1][: number]
    print(top[: 4])
    for i in range(len(new_prob)):
        pad_value = np.array([0, 0, 0])
        pad_value[index] = 1
        if i in top:
            new_prob[i, ] = pad_value
        else:
            new_prob[i, index] = 0.
    return new_prob


test_pred_prob = ensemble_predictions(test_preds)
test_pred = test_pred_prob.argmax(axis=1)

test_pro = test_label[['ship']]
test_pro['pro_1'] = test_pred_prob[:,0]
test_pro['pro_2'] = test_pred_prob[:,1]
test_pro['pro_3'] = test_pred_prob[:,2]
pred_pro = merge(test_pro[['pro_1', 'pro_2', 'pro_3']].values, 900,0)
test_pred = pred_pro.argmax(axis=1)


test_data = test_label[['ship']]
test_data["label"] = test_pred
test_data["label"] = test_data["label"].map({0:'围网',1:'刺网',2:'拖网'})
# test_data['label'][:100] = '刺网'
test_data[["ship", "label"]].to_csv("result.csv", index=False, header=None)
print(test_data["label"].value_counts())
print('runtime:', time.time() - start_t)