import numpy as np 

import matplotlib
matplotlib.use("TkAgg") # Use TKAgg to show figures
import matplotlib.pyplot as plt 
from sklearn.manifold import TSNE 
import torch

print(matplotlib.get_backend())

seq = '08'
f_ind = 100
fname = '{:07d}'.format(f_ind)

load_path = seq_path = '/data2/zixuan.chen/data/validation_predictions/sequences/'+seq+'/'
load_folder = '/data2/zixuan.chen/data/validation_predictions/sequences/08/scans/' 
load_path = load_folder + fname + '.npy'

a = np.load(load_path, allow_pickle=True)


seq = a[0]
f_name = a[1]
# _ids = a[2]
# _sem_labels = a[3]
# _n_pts = a[4]
# _coors = a[5]
_feats = a[6]
# frame_preds = a[7]
ins_preds = a[8]
# velo_file = a[9]
batch_labels = a[10]
batch_ins = a[11]

# ----------------------
ins = ins_preds
sem = batch_labels
valid = ins !=0

max_pt = 30 
_ids = []
_sem_labels = []
_n_pts = []
# _coors = []
# _feats = []
_ins_labels = []

ids, n_ids = np.unique(ins[valid], return_counts=True) 
for ii in range(len(ids)):
    if n_ids[ii] <= max_pt:
        continue
    pt_idx = np.where(ins==ids[ii])[0]
    ins_label = torch.tensor(batch_ins[pt_idx])
    # coors = torch.tensor(pt_coors[pt_idx])
    sem_label = np.unique(sem[pt_idx])
    # features = torch.tensor(feat[pt_idx]).type(torch.float32)
    n_pt = n_ids[ii]
    _ids.extend([ids[ii]])     
    _sem_labels.extend(sem_label)  
    _n_pts.extend([n_pt])       
    # _coors.extend([coors])      
    # _feats.extend([features])
    _ins_labels.extend([ins_label])

ins_labels = np.concatenate(_ins_labels, axis=0)
feats = np.concatenate(_feats, axis=0)

c_ids = np.unique(ins_labels, return_counts=True)[0]
# ------------------------------ 

X, y = feats, ins_labels
n_samples, n_features = X.shape

'''t-SNE'''
tsne = TSNE(n_components=2, init='pca', random_state=501)
X_tsne = tsne.fit_transform(X)

print("Org data dimension is {}.Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))

''''''
# x_min, x_max = X_tsne.min(0), X_tsne.max(0)
# X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
# plt.figure() # figsize=(8, 8)
# for i in range(10):
#     plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i][0]), color=plt.cm.Set1(y[i][0]), 
#              fontdict={'weight': 'bold', 'size': 9})
# plt.xticks([])
# plt.yticks([])


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
 
    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i][0]),
                 color=plt.cm.Set1(np.argwhere(c_ids==y[i][0])[0][0]),## color=plt.cm.Set1(np.random.randint(10))
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig

plt.rcParams['figure.figsize'] = (20, 20) ## 显示的大小
fig = plot_embedding( X_tsne, y, 't-sne-seq-{}-frame-{}.jpg'.format(seq, fname))
plt.savefig('t-sne-seq-{}-frame-{}.jpg'.format(seq, fname))
# plt.show()
