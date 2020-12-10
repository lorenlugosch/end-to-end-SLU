#!/usr/bin/env python
# coding: utf-8

# In[1]:


import data
import models
import soundfile as sf
import torch
from sklearn.model_selection import GridSearchCV
import pickle as pkl
# In[2]:


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



# In[7]:


slu_path='/home/ec2-user/fluent_speech_commands_dataset/'


# In[8]:


#! pip install textgrid
train = pd.read_csv(os.path.join(slu_path, 'data/train_data.csv'), index_col = 0)
val = pd.read_csv(os.path.join(slu_path,'data/valid_data.csv'), index_col = 0)
test= pd.read_csv(os.path.join(slu_path,'data/test_data.csv'), index_col = 0)



# In[9]:




# In[10]:


acts_to_id = {act: i for act, i in zip(sorted(train['action'].unique()), range(len(train['action'].unique())))}
id_to_acts = {val:key for key, val in acts_to_id.items()}
locs_to_id = {act: i for act, i in zip(sorted(train['location'].unique()), range(len(train['location'].unique())))}
id_to_locs = {val:key for key, val in locs_to_id.items()}

obj_to_id = {act: i for act, i in zip(sorted(train['object'].unique()), range(len(train['object'].unique())))}
id_to_obj = {val:key for key, val in obj_to_id.items()}

unique_labels = [(a,o,l) for a,o,l in train[['action', 'object', 'location']].drop_duplicates().reset_index(drop = True).to_numpy()]
lbl_to_id = {act: i for act, i in zip(unique_labels, range(len(unique_labels)))}
id_to_lbl = {val:key for key, val in lbl_to_id.items()}


# In[11]:


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


# In[12]:


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
if 'grid_search_flattened_knn.pkl' not in os.listdir('.'):
    print("Training kNN")
    knn = KNeighborsClassifier(n_jobs = -1)
    gs1 = GridSearchCV(knn, {'n_neighbors':[1,3,5,7,9,11,13,15,25]})
    gs2 = GridSearchCV(knn, {'n_neighbors':[1,3,5,7,9,11,13,15,25]})

    gs1_fit = gs1.fit(x_tr_flat, y_tr)
    gs2_fit = gs2.fit(x_tr_avg, y_tr)
    

    with open('grid_search_flattened_knn.pkl', 'wb') as f:
        pkl.dump(gs1_fit, f)

    with open('grid_search_averaged_knn.pkl', 'wb') as f:
        pkl.dump(gs2_fit, f)

else:
    with open('grid_search_flattened_knn.pkl', 'rb') as f:
        gs1_fit = pkl.load(f)

    with open('grid_search_averaged_knn.pkl', 'rb') as f:
        gs2_fit = pkl.load(f)
print("Accuracy on flattened data, knn: {}".format(accuracy_score(y_te, gs1_fit.predict(x_te_flat))))
print("Accuracy on averaged data, knn: {}".format(accuracy_score(y_te, gs2_fit.predict(x_te_avg))))

print("Grid search, flattened")
print(gs1_fit.get_params())

print("Grid search, averaged")
print(gs2_fit.get_params())

