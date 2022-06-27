import os
import numpy as np
import yaml

data_cfg = '/data1/zixuan.chen/data/kitti/semantic-kitti.yaml'
with open(data_cfg, 'r') as stream:
    doc = yaml.safe_load(stream)
    learning_map_doc = doc['learning_map']
    inv_learning_map_doc = doc['learning_map_inv']

inv_learning_map = np.zeros((np.max([k for k in inv_learning_map_doc.keys()]) + 1), dtype=np.int32)
for k, v in inv_learning_map_doc.items():
    inv_learning_map[k] = v


seq = 8
folder = '/_data/zixuan/data_0620/single_frame/validation_predictions/sequences/'+'08'+'/'+'scans'

path_list = os.listdir(folder)
path_list.sort(key=lambda x: int(x[3:-4]))
save_path = '/_data/zixuan/data_0620/single_frame/pls_evaluation'

for i in range(0, len(path_list)):  # len(path_list)
    path = os.path.join(folder, path_list[i])
    data = np.load(path, allow_pickle=True)
    label_sem_class = data[7]
    ins_preds = data[8]
    # seq = 8

    #clean instances which have too few points
    for ins_id in np.unique(ins_preds):              
        if ins_id == 0:
            continue
        valid_ind = np.argwhere(ins_preds == ins_id)[:, 0]     
        ins_preds[valid_ind] = ins_id+20                       
        if valid_ind.shape[0] < 25:
            ins_preds[valid_ind] = 0             
                                                                
    for sem_id in np.unique(label_sem_class):                 
        if sem_id < 1 or sem_id > 8:                           
            valid_ind = np.argwhere((label_sem_class == sem_id) & (ins_preds == 0))[:, 0]    
            ins_preds[valid_ind] = sem_id
            
    ins_preds = ins_preds.astype(np.int32)
    new_preds = np.left_shift(ins_preds, 16)

    sem_pred = label_sem_class.astype(np.int32)
    inv_sem_labels = inv_learning_map[sem_pred]
    new_preds = np.bitwise_or(new_preds, inv_sem_labels)

    if not os.path.exists(os.path.join(save_path, 'sequences', '{0:02d}'.format(seq), 'predictions')):
        os.makedirs(os.path.join(save_path, 'sequences', '{0:02d}'.format(seq), 'predictions'))

    new_preds.tofile('{}/{}/{:02d}/predictions/{:06d}.label'.format(save_path, 'sequences', seq, i))
    print(i)
