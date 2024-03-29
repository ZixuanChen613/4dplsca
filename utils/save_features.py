# import pdb
# pdb.set_trace()
import numpy as np
import os
import torch
from shutil import copyfile
import cont_assoc.utils.predict as pred
#Save point features for contrastive approach
def save_features(test_loader, batch, s_ind, f_ind, frame_points, pt_features, proj_mask, ins_preds, frame_preds, save_preds):


    _ids = []
    _sem_labels = []
    _n_pts = []
    _coors = []
    _feats = []
    fname = '{:07d}'.format(f_ind) # x['pcd_fname'][i][-10:-4]
    seq = test_loader.dataset.sequences[s_ind]  # x['pcd_fname'][i][-22:-20]
    pt_coors = frame_points[:, :3]    # (123398, 3)
    feat = pt_features       # (123389, 256)
    frame_preds = frame_preds.astype(np.int64)
    ins_preds = ins_preds.astype(np.int32)

    if save_preds:
        frame_preds = pred.majority_voting(frame_preds, ins_preds)
        sem = frame_preds
        ins = ins_preds
        valid = ins != 0
        seq_path = '/_data/zixuan/data_0727/single_frame/validation_predictions/sequences/'+seq+'/'
        max_pt = 25

    else:
        sem = batch.val_labels[0].astype(np.uint8)              # (123389, 1)   0-19
        ins = batch.ins_labels.cpu().numpy().astype(np.int32)  # (123389, 1)  instance labels
        valid = np.where((proj_mask==True) & (ins!=0))[0]  # (119195,)   valid instance flag ; (119195, 1) True
        seq_path = '/_data/zixuan/data_0727/single_frame/instance_features/sequences/'+seq+'/'
        max_pt = 10
        sem = pred.majority_voting(sem, ins)  ####???


    ids, n_ids = np.unique(ins[valid], return_counts=True) # ins[valid]: (4322, 1)
    for ii in range(len(ids)):
        if n_ids[ii] <= max_pt:
            continue
        pt_idx = np.where(ins==ids[ii])[0]      # number is n_ids
        coors = torch.tensor(pt_coors[pt_idx])
        sem_label = np.unique(sem[pt_idx])
        features = torch.tensor(feat[pt_idx]).type(torch.float32)
        n_pt = n_ids[ii]
        _ids.extend([ids[ii]])      # [65791, 13631498, 13697034, 13762570, 13828106, 13893642, 13959178, 14024714, 14090250, 14155786, 14286858, 14352394, 33030154]
        _sem_labels.extend(sem_label)  # [8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        _n_pts.extend([n_pt])       # [88, 108, 63, 103, 469, 1737, 1103, 189, 31, 75, 20, 60, 271]
        _coors.extend([coors])      # len(ids) x torch.Size([n, 3])
        _feats.extend([features])   # len(ids) x torch.Size([n, 128])

    filename = seq_path + 'scans/' + fname
    if not os.path.exists(seq_path):
        os.makedirs(seq_path+'scans/')
        orig_seq_path = test_loader.dataset.path + '/sequences/' + seq + '/'
        #copy txt files
        copyfile(orig_seq_path + 'poses.txt', seq_path + 'poses.txt')
        copyfile(orig_seq_path + 'calib.txt', seq_path + 'calib.txt')
        #create empty file
        f = open(seq_path + 'empty.txt','w')
        f.close()
    
    if not save_preds:
        #dont save if no instances: len(ids) == 0
        if len(_ids) == 0:
            f = open(seq_path + 'empty.txt', 'a')
            f.write(fname+'\n')
            f.close()
            return

    dataset_seq_path = os.path.join(test_loader.dataset.path, 'sequences', test_loader.dataset.sequences[s_ind])
    velo_file = os.path.join(dataset_seq_path, 'velodyne', test_loader.dataset.frames[s_ind][f_ind] + '.bin')

    batch_labels = batch.val_labels[0].astype(np.uint8).reshape((-1, 1))
    batch_ins = batch.ins_labels.cpu().numpy().astype(np.int32).reshape((-1, 1))

    if save_preds:
        np_instances = np.array([seq, fname, _ids, _sem_labels, _n_pts, _coors, _feats, 
        frame_preds, ins_preds, velo_file, batch_labels, batch_ins], dtype=object)
    else:
        np_instances = np.array([seq, fname, _ids, _sem_labels, _n_pts, _coors, _feats], dtype=object)

    np.save(filename, np_instances, allow_pickle=True)

