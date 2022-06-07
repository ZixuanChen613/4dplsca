import numpy as np 
import os
import matplotlib
matplotlib.use("TkAgg") # Use TKAgg to show figures
import matplotlib.pyplot as plt 
from sklearn.manifold import TSNE 
import torch

print(matplotlib.get_backend())

seq = '11'
# f_ind = 0
# fname = '{:07d}'.format(f_ind)

load_folder = '/_data/tsne/sequences/' + seq + '/'
path_list = os.listdir(load_folder)
path_list.sort(key=lambda x: int(x[3:-4]))

ins_sems = []
ins_ids = []
ins_feats = []


for i in range(0, len(path_list)):  # len(path_list)
    path = os.path.join(load_folder, path_list[i])
    tsne_np = np.load(path, allow_pickle=True)
    if len(tsne_np[2]) == 0:
        print(i)
        continue
    # print(i)
    # if i==647:
    #     aaa = 1
    ins_sem = tsne_np[0]
    ins_id = tsne_np[1]
    ins_feat = tsne_np[2][0]
    ins_sems.append(ins_sem)
    ins_ids.append(ins_id)
    ins_feats.append(ins_feat)


ins_labels = np.concatenate(ins_ids, axis=0)
feats = np.concatenate(ins_feats, axis=0)

ids, n_ids = np.unique(ins_labels, return_counts=True)
limit_n = 100
for i in range(len(ids)):
    if n_ids[i] < limit_n:
        not_valid_ind = np.argwhere(ids[i] == ins_labels)
        ins_labels = np.delete(ins_labels, not_valid_ind, axis=0)
        feats = np.delete(feats, not_valid_ind, axis=0)

c_ids = np.unique(ins_labels, return_counts=True)[0]
# ------------------------------ 

X, y = feats, ins_labels
n_samples, n_features = X.shape

'''t-SNE'''
tsne = TSNE(n_components=2, init='pca', random_state=501)
X_tsne = tsne.fit_transform(X)

print("Org data dimension is {}.Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))

''''''



def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
 
    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(np.argwhere(c_ids==y[i])[0][0]),
                # color=plt.cm.Set1(np.random.randint(20)),
                fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig

plt.rcParams['figure.figsize'] = (20, 20) ## 显示的大小
fig = plot_embedding( X_tsne, y, 't-sne-seq-{}-ca-limit_{}.jpg'.format(seq, str(limit_n)))
plt.savefig('t-sne-seq-{}-ca-limit_{}.jpg'.format(seq, str(limit_n)))
# plt.show()
