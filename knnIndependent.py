#!/usr/bin/env python
# coding: utf-8



import data
import models
import soundfile as sf
import torch
from sklearn.model_selection import GridSearchCV
import pickle as pkl


import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import os


# In[ ]:





# In[3]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = data.read_config("experiments/no_unfreezing.cfg"); _,_,_=data.get_SLU_datasets(config)
model = models.Model(config).eval()
model.load_state_dict(torch.load("experiments/no_unfreezing/training/model_state.pth", map_location=device)) # load trained model
import pdb
signal, _ = sf.read("test.wav")
signal = torch.tensor(signal, device=device).float().unsqueeze(0)

model.decode_intents(signal)


# In[6]:


signal.shape


# In[7]:


slu_path='/home/ec2-user/fluent_speech_commands_dataset/'


# In[8]:


#! pip install textgrid
train = pd.read_csv(os.path.join(slu_path, 'data/train_data.csv'), index_col = 0)
val = pd.read_csv(os.path.join(slu_path,'data/valid_data.csv'), index_col = 0)
test= pd.read_csv(os.path.join(slu_path,'data/test_data.csv'), index_col = 0)



acts_to_id = {act: i for act, i in zip(sorted(train['action'].unique()), range(len(train['action'].unique())))}
id_to_acts = {val:key for key, val in acts_to_id.items()}
locs_to_id = {act: i for act, i in zip(sorted(train['location'].unique()), range(len(train['location'].unique())))}
id_to_locs = {val:key for key, val in locs_to_id.items()}

obj_to_id = {act: i for act, i in zip(sorted(train['object'].unique()), range(len(train['object'].unique())))}
id_to_obj = {val:key for key, val in obj_to_id.items()}

unique_labels = [(a,o,l) for a,o,l in train[['action', 'object', 'location']].drop_duplicates().reset_index(drop = True).to_numpy()]
lbl_to_id = {act: i for act, i in zip(unique_labels, range(len(unique_labels)))}
id_to_lbl = {val:key for key, val in lbl_to_id.items()}




def vectorize_intents(input_df):
    acts_vect = []
    obj_vect = []
    locs_vect  = []
    lbls_vect = []
    
    for act, obj, loc in zip(list(input_df['action']), list(input_df['object']), list(input_df['location'])):
        acts_vect.append(acts_to_id[act])
        obj_vect.append(obj_to_id[obj])
        locs_vect.append(locs_to_id[loc])
        lbls_vect.append(lbl_to_id[(act, obj, loc)])
    input_df['action_ft'] = acts_vect
    input_df['location_ft'] = locs_vect
    input_df['object_ft'] = obj_vect
    input_df['label'] = lbls_vect
    return input_df




train = vectorize_intents(train)
test = vectorize_intents(test)




def get_x_y(input_df):
    x_s = []
    y_s = []
    for fpath, label in zip(list(input_df['path']), list(input_df['label'])):
        signal, _ = sf.read("test.wav")
        signal = torch.tensor(signal, device=device).float().unsqueeze(0)
        encoded = model.pretrained_model.compute_features(signal)
        x_s.append(encoded.cpu().numpy()[0])
        y_s.append(label)
    return x_s, y_s


# In[23]:
if 'x_tr.pkl' not in os.listdir('.'):
    

    x_tr, y_tr = get_x_y(train)
    x_te, y_te = get_x_y(test)
    with open('x_tr.pkl', 'wb') as f:
        pkl.dump(x_tr, f)
    with open('y_tr.pkl', 'wb') as f:
        pkl.dump(y_tr, f)
    with open('x_te.pkl', 'wb') as f:
        pkl.dump(x_te, f)
    with open('y_te.pkl', 'wb') as f:
        pkl.dump(y_te,f)

else:
    with open('x_tr.pkl', 'rb') as f:
        x_tr = pkl.load(f)
    with open('y_tr.pkl', 'rb') as f:
        y_tr = pkl.load(f)
    with open('x_te.pkl', 'rb') as f:
        x_te = pkl.load(f)
    with open('y_te.pkl', 'rb') as f:
        y_te = pkl.load(f)


# In[21]:

def flatten_data(dlist):
    return [j.flatten() for j in dlist]

def get_avg(dlist):
    print(dlist[0].shape)
    return [j.mean(axis=1) for j in dlist]
x_tr_flat = flatten_data(x_tr)
#! pip install scikit-learn
x_te_flat = flatten_data(x_te)

x_tr_avg = get_avg(x_tr)
x_te_avg = get_avg(x_te)
print(x_tr_flat[0].shape, x_te_flat[0].shape)
print(x_tr_avg[0].shape)
# In[22]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import pickle as pkl


# In[ ]:
y_tr_obj = train['object_ft']
y_tr_loc = train['location_ft']
y_tr_act = train['action_ft']

y_te_obj = test['object_ft']
y_te_loc = test['location_ft']
y_te_act = test['action_ft']

if 'knn_obj_flat.pkl' not in os.listdir('.'):
    for tr_set,nm in zip([x_tr_flat, x_tr_avg], ['flat', 'avg']):
        knn_obj = KNeighborsClassifier(n_jobs = -1, n_neighbors=5).fit(tr_set, y_tr_obj)
        knn_loc = KNeighborsClassifier(n_jobs = -1, n_neighbors = 5).fit(tr_set, y_tr_loc)
        knn_act = KNeighborsClassifier(n_jobs = -1, n_neighbors = 5).fit(tr_set, y_tr_act)
        for fname, modl in zip(['knn_obj_{}.pkl'.format(nm), 'knn_loc_{}.pkl'.format(nm), 'knn_act_{}.pkl'.format(nm)], [knn_obj, knn_loc, knn_act]):
            with open(fname, 'wb') as f:
                pkl.dump(modl, f)
    
for nm in ['flat', 'avg']:
    with open('knn_obj_{}.pkl'.format(nm), 'rb') as f:
        knn_obj = pkl.load(f)
    with open('knn_loc_{}.pkl'.format(nm), 'rb') as f:
        knn_loc = pkl.load(f)
    with open('knn_act_{}.pkl'.format(nm), 'rb') as f:
        knn_act = pkl.load(f)
    if nm == 'flat':
        tst_set = x_te_flat
    else:   
        tst_set = x_te_avg
    pred_obj = knn_obj.predict(tst_set)
    print("{} accuracy, independent knn training, object: {}".format(nm, accuracy_score(y_te_obj, pred_obj)))
    pred_loc = knn_obj.predict(tst_set)
    print("{} accuracy, independent knn training, location: {}".format(nm, accuracy_score(y_te_loc, pred_loc)))
    pred_act = knn_obj.predict(tst_set)
    print("{} accuracy, independent knn training, action: {}".format(nm, accuracy_score(y_te_act, pred_act)))
    pred_lbls = [(int(a), int(o), int(l)) for a, o, l in zip(pred_act, pred_obj, pred_loc)]
    pred_lbls_vect = [lbl_to_id.get(lbl, -1) for lbl in pred_lbls]
    wrong_lbls = [j for j in pred_lbls_vect if j == -1]
    print(len(wrong_lbls)/len(pred_lbls_vect))
    print("{} accuracy, independent knn training".format(nm))
    print("{}".format(accuracy_score(y_te, pred_lbls_vect)))

