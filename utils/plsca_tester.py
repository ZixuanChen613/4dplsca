#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Class handling the test of any model
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 11/06/2018
#


# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/

# import pdb
# pdb.set_trace()

# Basic libs
import os
import torch
import torch.nn as nn
import numpy as np
from os import makedirs, listdir
from os.path import exists, join
import time
import json
from sklearn.neighbors import KDTree

# PLY reader
from utils.ply import read_ply, write_ply

# Metrics
from utils.metrics import IoU_from_confusions, fast_confusion
from sklearn.metrics import confusion_matrix
from utils.kalman_filter import KalmanBoxTracker
from utils.tracking_utils import *
from scipy.optimize import linear_sum_assignment
#from utils.visualizer import show_ModelNet_models
from utils.save_features import save_features
import cont_assoc.utils.predict as pred
import cont_assoc.utils.contrastive as cont

from easydict import EasyDict as edict
import yaml
# ----------------------------------------------------------------------------------------------------------------------
#
#           Tester Class
#       \******************/
#


def associate_overlapped_ins(overlaps, overlap_history, label_inst, new_instances, prev_instances):
    for (new_id, prev_id) in overlaps.items():
        ins_points = torch.where((label_inst == new_id))
        if not new_id in new_instances or prev_id not in prev_instances:
            continue
        overlap_history[new_id] = prev_id#add tracking id
        label_inst[ins_points[0]] = prev_id
        prev_instances[prev_id]['bbox_proj'] = new_instances[new_id]['bbox_proj']
        # prev_instances[prev_id]['mean'] = new_instances[new_id]['mean']
        prev_instances[prev_id]['center'] = new_instances[new_id]['center']
        prev_instances[prev_id]['life'] += 1
        prev_instances[prev_id]['tracker'].update(new_instances[new_id]['kalman_bbox'], prev_id)
        prev_instances[prev_id]['kalman_bbox'] = torch.from_numpy(prev_instances[prev_id]['tracker'].get_state()).float()
        prev_instances[prev_id]['bbox'] = kalman_box_to_eight_point(prev_instances[prev_id]['kalman_bbox'])

        del new_instances[new_id]

    return overlap_history, label_inst, new_instances, prev_instances

def associate_not_overlapped_ins(associations, overlaps, overlap_history, label_inst, new_instances, prev_instances):
    for prev_id, new_id in associations:
        if new_id in overlaps:
            continue
        
        ins_points = torch.where((label_inst == new_id))
        label_inst[ins_points[0]] = prev_id
        overlap_history[new_id] = prev_id
        prev_instances[prev_id]['bbox_proj'] = new_instances[new_id]['bbox_proj']
        # prev_instances[prev_id]['mean'] = new_instances[new_id]['mean']
        prev_instances[prev_id]['center'] = new_instances[new_id]['center']
        prev_instances[prev_id]['life'] += 1
        prev_instances[prev_id]['tracker'].update(new_instances[new_id]['kalman_bbox'], prev_id)
        prev_instances[prev_id]['kalman_bbox'] = torch.from_numpy(prev_instances[prev_id]['tracker'].get_state()).float()
        prev_instances[prev_id]['bbox'] = kalman_box_to_eight_point(prev_instances[prev_id]['kalman_bbox'])

        del new_instances[new_id]
        
    return overlap_history, label_inst, new_instances, prev_instances





def associate_instances(previous_instances, current_instances, overlaps,  pose, association_weights):
    pose = pose.cpu().float()
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    p_n = len(previous_instances.keys())
    c_n = len(current_instances.keys())

    association_costs = torch.zeros(p_n, c_n)
    prev_ids = []
    current_ids = []

    current_instances_prev = {}
    for i, (k, v) in enumerate(previous_instances.items()):
        #v['kalman_bbox'][0:3] += pose[:3, 3]
        #v['kalman_bbox'][0:3] = torch.matmul(v['kalman_bbox'][0:3],pose[:3, :3])
        #v['bbox'][0:3] = v['kalman_bbox'][0:3] - v['kalman_bbox'][4:]/2
        #v['bbox'][3:] = v['kalman_bbox'][0:3] + v['kalman_bbox'][4:] / 2
        pass

    for i, (k, v) in enumerate(previous_instances.items()):
        prev_ids.append(k)
        for j, (k1, v1) in enumerate(current_instances.items()):
            if v1['class'] ==  v['class'] and k1 not in overlaps:
                #cost_3d = 1 - IoU(v1['bbox'], v['bbox'])
                #if k1 in current_instances_prev:
                #    cost_3d = min (cost_3d, 1 - IoU(current_instances_prev[k1]['bbox'], v['bbox']))
                #if cost_3d > 0.75:
                #    cost_3d = 1e8
                #if v1['bbox_proj'] is not None:
                #    cost_2d = 1 - IoU(v1['bbox_proj'], v['bbox_proj'])
                #    if k1 in current_instances_prev:
                #        cost_2d = min(cost_2d, 1 - IoU(current_instances_prev[k1]['bbox_proj'], v['bbox_proj']))

                #    if cost_2d > 0.75:
                #        cost_2d = 1e8
                #else:
                #    cost_2d = 0

                cost_center = euclidean_dist(v1['kalman_bbox'], v['kalman_bbox'])
                if k1 in current_instances_prev:
                    cost_center = min(cost_center, euclidean_dist(current_instances_prev[k1]['kalman_bbox'],v['kalman_bbox']))
                if cost_center > 5:
                    cost_center = 1e8

                #feature_cost = 1 - cos(v1['mean'], v['mean'])
                #if k1 in current_instances_prev:
                #    feature_cost = min(feature_cost, 1 - cos(current_instances_prev[k1]['mean'], v['mean']))
                #if feature_cost > 0.5:
                #    feature_cost = 1e8
                costs = torch.tensor([0, 0, cost_center, 0])

                for idx, a_w in enumerate(association_weights):
                    association_costs[i, j] += a_w * costs[idx]
            else:
                association_costs[i, j] = 1e8

            if i == 0:
                current_ids.append(k1)

    idxes_1, idxes_2 = linear_sum_assignment(association_costs.cpu().detach())

    associations = []

    for i1, i2 in zip(idxes_1, idxes_2):
        # max_cost = torch.sum((previous_instances[prev_ids[i1]]['var'][0,-3:]/2)**2)
        if association_costs[i1][i2] < 1e8:
            associations.append((prev_ids[i1], current_ids[i2]))

    return association_costs, associations

def associate_instances_overlapping_frames(previous_ins_label, current_ins_label):

    previous_instance_ids, c_p = np.unique(previous_ins_label, return_counts=True)
    current_instance_ids, c_c = np.unique(current_ins_label, return_counts=True)

    previous_instance_ids = [x for i,x in enumerate(previous_instance_ids) if c_p[i] > 25] #
    current_instance_ids = [x for i, x in enumerate(current_instance_ids) if c_c[i] > 50] #

    p_n = len(previous_instance_ids) -1
    c_n = len(current_instance_ids) -1

    prev_ids = []
    current_ids = []

    association_costs = torch.zeros(p_n, c_n)
    for i, p_id in enumerate(previous_instance_ids[1:]):
        prev_ids.append(p_id)
        for j, c_id in enumerate(current_instance_ids[1:]):
            intersection = np.sum( (previous_ins_label==p_id) & (current_ins_label == c_id) )

            union =  np.sum(previous_ins_label==p_id) + np.sum(current_ins_label == c_id) - intersection
            iou = intersection/union
            cost = 1 - iou
            association_costs[i, j] = cost if cost < 0.50 else 1e8
            if i == 0:
                current_ids.append(c_id)

    idxes_1, idxes_2 = linear_sum_assignment(association_costs.cpu().detach())
    associations = []
    association_costs_matched = []
    for i1, i2 in zip(idxes_1, idxes_2):
        if association_costs[i1][i2] < 1e8:
            associations.append((prev_ids[i1], current_ids[i2]))
            association_costs_matched.append(association_costs[i1][i2])

    return association_costs_matched,  associations


class ModelTester:

    # Initialization methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, pls_net, ca_net, pls_chkp_path=None, ca_chkp_path=None, on_gpu=True):

        ############
        # Parameters
        ############
        self.pls_net = pls_net
        self.ca_net = ca_net

        self.instances = {} #store instance ids and mean, cov fors sequantial prediction
        self.next_ins_id = 1 #next ins id for new instance

        # Choose to train on CPU or GPU
        if on_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        self.pls_net.to(self.device)
        self.ca_net.to(self.device)

        ##########################
        # Load previous checkpoint
        ##########################

        pls_checkpoint = torch.load(pls_chkp_path)
        ca_checkpoint = torch.load(ca_chkp_path)

        self.pls_net.load_state_dict(pls_checkpoint['model_state_dict'])
        self.ca_net.load_state_dict(ca_checkpoint['state_dict'])       # map_location='cpu'

        self.epoch = pls_checkpoint['epoch']
        self.pls_net.eval()
        self.ca_net.eval()
        self.last_ins_id = 0  ## ?????????????
        print("Model and training state restored.")



        return

    # Test main methods
    # ------------------------------------------------------------------------------------------------------------------


    def get_ins_feat(self, pt_coors, ins_pred, raw_features):
        #Group points into instances
        pt_raw_feat = [raw_features]
        pt_coordinates = [pt_coors]
        ins_pred = [ins_pred]

        coordinates, features, n_instances, ins_ids, ins_pred = cont.group_instances(pt_coordinates, pt_raw_feat, ins_pred)

        #Discard scans without instances
        features = [x for x in features if len(x)!=0]
        coordinates = [x for x in coordinates if len(x)!=0]

        if len(features)==0:#don't run tracking head if no ins
            # return [], [], [], ins_pred
            return [], [], [], ins_pred, {}

        #Get per-instance feature
        tracking_input = {'pt_features':features,'pt_coors':coordinates}

        ins_feat = self.ca_net(tracking_input)

        if len(coordinates) != len(ins_ids):
            #scans without instances
            new_feats, new_coors, new_coors_T = cont.fix_batches(ins_ids, features, coordinates, coordinates_T)
            tracking_input = {'pt_features':new_feats,'pt_coors':new_coors, 'pt_coors_T':new_coors_T}

        return ins_feat, n_instances, ins_ids, ins_pred, tracking_input

    def track(self, ins_pred, ins_feat, n_instances, ins_ids, tr_input, poses):
        #Separate instances of different scans
        points = tr_input['pt_coors']
        features = tr_input['pt_features']
        ins_feat = torch.split(ins_feat, n_instances)
        poses = [[p] for p in poses]

        #Instance IDs association
        ins_pred = self.ca_net.AssocModule.associate(ins_pred, ins_feat,
                                                            points, features,
                                                            poses, ins_ids)

        self.last_ins_id = self.ca_net.AssocModule.get_last_id()
        self.ca_net.AssocModule.update_last_id(self.last_ins_id)

        return ins_pred




    def panoptic_4d_test(self, test_loader, pls_cfg, ag_cfg, num_votes=100, debug=True):
        """
        Test method for slam segmentation models
        """

        ############
        # Initialize
        ############

        # Choose validation smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        test_smooth = 0.5
        last_min = -0.5
        softmax = torch.nn.Softmax(1)

        ############### get inv_learning_map #############
        data_cfg = '/data1/zixuan.chen/data/kitti/semantic-kitti.yaml'
        with open(data_cfg, 'r') as stream:
            doc = yaml.safe_load(stream)
            learning_map_doc = doc['learning_map']
            inv_learning_map_doc = doc['learning_map_inv']

        inv_learning_map = np.zeros((np.max([k for k in inv_learning_map_doc.keys()]) + 1), dtype=np.int32)
        for k, v in inv_learning_map_doc.items():
            inv_learning_map[k] = v

        # Number of classes including ignored labels
        nc_tot = test_loader.dataset.num_classes
        nc_model = self.pls_net.C
        # Test saving path
        test_path = None
        report_path = None

        if pls_cfg.dataset_task == '4d_panoptic':

            #assoc_saving = [asc_type for idx, asc_type in enumerate(config.association_types) if config.association_weights[idx] > 0]
            #assoc_saving.append(str(config.n_test_frames))
            #assoc_saving = '_'.join(assoc_saving)
            assoc_saving = pls_cfg.sampling
            pls_cfg.assoc_saving = pls_cfg.sampling+'_'+ pls_cfg.decay_sampling
            if hasattr(pls_cfg, 'stride'):
                pls_cfg.assoc_saving = pls_cfg.sampling + '_' + pls_cfg.decay_sampling+ '_str' + str(pls_cfg.stride) +'_'
            if hasattr(pls_cfg, 'big_gpu') and pls_cfg.big_gpu:
                pls_cfg.assoc_saving = pls_cfg.assoc_saving + 'bigpug_'


        if pls_cfg.saving:
            test_path = join('/_data/zixuan/data_v2/', 'test_plsca', pls_cfg.saving_path.split('/')[-1]+ '_'+pls_cfg.assoc_saving+str(pls_cfg.n_test_frames))
            if not exists(test_path):
                makedirs(test_path)
            report_path = join(test_path, 'reports')
            if not exists(report_path):
                makedirs(report_path)

        if test_loader.dataset.set in ['validation']:
            for folder in ['val_predictions', 'val_probs']:
                if not exists(join(test_path, folder)):
                    makedirs(join(test_path, folder))
        else: # test
            for folder in ['test_predictions', 'test_probs']:
                if not exists(join(test_path, folder)):
                    makedirs(join(test_path, folder))

        # Init validation container
        all_f_preds = []
        all_f_labels = []
        if test_loader.dataset.set == 'validation':
            for i, seq_frames in enumerate(test_loader.dataset.frames):
                all_f_preds.append([np.zeros((0,), dtype=np.int32) for _ in seq_frames])
                all_f_labels.append([np.zeros((0,), dtype=np.int32) for _ in seq_frames])

        #####################
        # 4Dplsca Network predictions
        #####################

        predictions = []
        targets = []
        test_epoch = 0

        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)

        processed = 0 #number of frames that processed

        # Start test loop
        while True:
            print('Initialize workers')
            for i, batch in enumerate(test_loader):
                # New time
                t = t[-1:]
                t += [time.time()]

                if i == 0:
                    print('Done in {:.1f}s'.format(t[1] - t[0]))

                flag = True
                if pls_cfg.n_test_frames > 1:
                    lengths = batch.lengths[0].cpu().numpy()
                    for b_i, length in enumerate(lengths):
                        f_inds = batch.frame_inds.cpu().numpy()
                        f_ind = f_inds[b_i, 1]
                        if f_ind % pls_cfg.n_test_frames != pls_cfg.n_test_frames-1:
                             flag = False

                if processed == test_loader.dataset.all_inds.shape[0]:
                    return
                #if not flag:
                #    continue
                #else:
                processed +=1

                if 'cuda' in self.device.type:
                    batch.to(self.device)
                
                if f_ind == 0:
                    prev_instances = {}
                    overlap_history = {}

                with torch.no_grad():
                    # f_ind = 1
                    outputs, centers_output, var_output, embedding = self.pls_net(batch, pls_cfg)   # # (153815, 19); (N, 1); (N, 260 = 256+4); (N, 256)
                    #ins_preds = torch.zeros(outputs.shape[0])

                    probs = softmax(outputs).cpu().detach().numpy()                        # (153815, 19)

                    for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                        if label_value in test_loader.dataset.ignored_labels:
                            probs = np.insert(probs, l_ind, 0, axis=1)
                    preds = test_loader.dataset.label_values[np.argmax(probs, axis=1)]
                    preds = torch.from_numpy(preds)
                    preds.to(outputs.device)
                    sequence = test_loader.dataset.sequences[batch.frame_inds[0][0]]
                    pose = test_loader.dataset.poses[batch.frame_inds[0][0]][batch.frame_inds[0][1]]
                    if sequence not in self.instances:
                        self.instances[sequence] = {}
                    #ins_preds = net.ins_pred(preds, centers_output, var_output, embedding, batch.points)
                    ins_preds, new_instances, ins_id = self.pls_net.ins_pred_in_time(pls_cfg, preds, centers_output, var_output, embedding, self.instances[sequence],
                                                     self.next_ins_id, batch.points, batch.times.unsqueeze(1), pose)

                    self.next_ins_id = ins_id#update next available ins id
                    for ins_id, instance in new_instances.items(): #add new instances to history
                        self.instances[sequence][ins_id] = instance

                    dont_track_ids = []
                    for ins_id in self.instances[sequence].keys():
                        if self.instances[sequence][ins_id]['life'] == 0:
                            dont_track_ids.append(ins_id)
                        self.instances[sequence][ins_id]['life'] -= 1

                    for ins_id in dont_track_ids:
                        del self.instances[sequence][ins_id]

                # Get probs and labels
                embedding = embedding.cpu().detach().numpy()             # 153815
                stk_probs = softmax(outputs).cpu().detach().numpy()      # 153815,19
                ins_preds = ins_preds.cpu().detach().numpy()             # 153815
                centers_output = centers_output.cpu().detach().numpy()   # 153815, 1
                lengths = batch.lengths[0].cpu().numpy()
                f_inds = batch.frame_inds.cpu().numpy()                  # 0,1
                r_inds_list = batch.reproj_inds                          # 119331
                r_mask_list = batch.reproj_masks                         # 123433
                f_inc_r_inds_list = batch.f_inc_reproj_inds              # 119160
                f_inc_r_mask_list = batch.f_inc_reproj_masks             # 123389

                labels_list = batch.val_labels                           # 123433
                torch.cuda.synchronize(self.device)

                t += [time.time()]

                # Get predictions and labels per instance
                # ***************************************

                i0 = 0
                for b_i, length in enumerate(lengths):                 # 153815

                    # Get prediction
                    pt_features = embedding[i0:i0 + length]            # 153815
                    probs = stk_probs[i0:i0 + length]                  # 153815, 19
                    c_probs = centers_output[i0:i0 + length]           # 153815
                    ins_probs = ins_preds[i0:i0 + length]
                    proj_inds = r_inds_list[b_i]                       # 119331
                    proj_mask = r_mask_list[b_i]                       # 123433
                    frame_labels = labels_list[b_i]                    # 123433
                    s_ind = f_inds[b_i, 0]
                    f_ind = f_inds[b_i, 1]

                    # Project predictions on the frame points
                    proj_probs = probs[proj_inds]                      # 119331
                    proj_c_probs = c_probs[proj_inds]                  # 119331
                    proj_ins_probs = ins_probs[proj_inds]              # 119331
                    proj_pt_features = pt_features[proj_inds]          # 119331

                    # Safe check if only one point:
                    if proj_probs.ndim < 2:
                        proj_c_probs = np.expand_dims(proj_c_probs, 0)
                        proj_probs = np.expand_dims(proj_probs, 0)
                        proj_ins_probs = np.expand_dims(proj_ins_probs, 0)

                    # Save probs in a binary file (uint8 format for lighter weight)
                    seq_name = test_loader.dataset.sequences[s_ind]
                    if test_loader.dataset.set in ['validation']:
                        folder = 'val_probs'
                        pred_folder = 'val_predictions'
                    else:
                        folder = 'test_probs'
                        pred_folder = 'test_predictions'

                    filename = '{:s}_{:07d}.npy'.format(seq_name, f_ind)
                    filepath = join(test_path, folder, filename)
                    filename_i = '{:s}_{:07d}_i.npy'.format(seq_name, f_ind)
                    filename_c = '{:s}_{:07d}_c.npy'.format(seq_name, f_ind)
                    # filename_e = '{:s}_{:07d}_e.npy'.format(seq_name, f_ind)
                    # filename_m = '{:s}_{:07d}_m.npy'.format(seq_name, f_ind)
                    filepath_i = join(test_path, folder, filename_i)
                    filepath_c = join(test_path, folder, filename_c)
                    # filepath_e = join(test_path, folder, filename_e)
                    # filepath_m = join(test_path, folder, filename_m)

                    frame_probs_uint8 = np.zeros((proj_mask.shape[0], nc_model), dtype=np.uint8)  # 123433
                    frame_c_probs = np.zeros((proj_mask.shape[0], 1))
                    ins_preds = np.zeros((proj_mask.shape[0]))
                    pt_features = np.zeros((proj_mask.shape[0], 256))

                    frame_probs = frame_probs_uint8[proj_mask, :].astype(np.float32) / 255
                    frame_probs = test_smooth * frame_probs + (1 - test_smooth) * proj_probs
                    frame_probs_uint8[proj_mask, :] = (frame_probs * 255).astype(np.uint8)


                    ins_preds[proj_mask] = proj_ins_probs
                    frame_c_probs[proj_mask] = proj_c_probs
                    pt_features[proj_mask] = proj_pt_features

                    ############# 

                    frame_probs_uint8_f = frame_probs_uint8.copy()
                    for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                            if label_value in test_loader.dataset.ignored_labels:
                                frame_probs_uint8_f = np.insert(frame_probs_uint8_f, l_ind, 0, axis=1)

                    # Predicted labels
                    frame_preds = test_loader.dataset.label_values[np.argmax(frame_probs_uint8_f,
                                                                            axis=1)].astype(np.int32)
                    ##############


                    seq_path = join(test_loader.dataset.path, 'sequences', test_loader.dataset.sequences[s_ind])
                    velo_file = join(seq_path, 'velodyne', test_loader.dataset.frames[s_ind][f_ind] + '.bin')
                    frame_points = np.fromfile(velo_file, dtype=np.float32)                         # 123433
                    frame_points = frame_points.reshape((-1, 4))                                    # 123433

                    #np.save(filepath, frame_probs_uint8)
                    #print ('Saving {}'.format(filepath_i))
                    np.save(filepath_i, ins_preds)                  # 123433
                    np.save(filepath_c, frame_c_probs)              # 123433
                    # np.save(filepath_e, pt_features)                # 123433
                    # np.save(filepath_m, proj_mask)

                    # ins_features = {}
                    # for ins_id in np.unique(ins_preds):
                    #     if int(ins_id) in self.instances[sequence]:
                    #         pt_idx = np.where(ins_preds==ins_id)[0]
                    #         ins_features[int(ins_id)] = torch.tensor(pt_features[pt_idx]).type(torch.float32)

                    # filename_f = '{:s}_{:07d}_f.npy'.format(seq_name, f_ind)
                    # filepath_f = join(test_path, folder, filename_f)

                    #np.save(filepath_f, ins_features)

                    #load current frame
                    ins_path = filepath_i
                    # fet_path = os.path.join(prediction_path, '{0:02d}_{1:07d}_f.npy'.format(sequence, idx))

                    label_sem_class = frame_preds                             # (123389,)
                    label_inst = np.load(ins_path)                                  # (123389,)   between [0, 13]
                    frame_points = np.fromfile(velo_file, dtype=np.float32)
                    points = frame_points.reshape((-1, 4))                          # (123389, 3)
                    hpoints = np.hstack((points[:, :3], np.ones_like(points[:, :1])))  # (123389, 4)
                    new_points = np.sum(np.expand_dims(hpoints, 2) * pose.T, axis=1)   # (123389, 4) 
                    points = new_points[:, :3]                                      # (123389, 3)

                    things = (label_sem_class < 9) & (label_sem_class > 0)          # np.where(things==True)[0] ---> 2056
                    ins_ids = np.unique(label_inst * things)                        # things instance index

                    features = {}
                    for ins_id in ins_ids:
                        features[ins_id] = torch.from_numpy(np.zeros((1,1)))  
                                               #  ????????????????????????????
                    # if os.path.exists(fet_path):
                    #     features = np.load(fet_path, allow_pickle=True).tolist()
                    # else:
                    #     features = {}
                    #     for ins_id in ins_ids:
                    #         features[ins_id] = torch.from_numpy(np.zeros((1,1)))

                    projections = do_range_projection(points)           # (2, 123389)
                    points = torch.from_numpy(points)
                    new_instances = {}

                    label_inst = torch.from_numpy(label_inst.astype(np.int32))

                    # get instances from current frames to track
                    for ins_id in ins_ids:                                        # ins_id = 43 ---> 53
                        if ins_id == 0:
                            continue
                        if int(ins_id) not in features:                           # if new id comes in the new frame
                            ids = np.where(label_inst == ins_id)
                            label_inst[ids] = 0                                   # set label = 0 ????
                            continue

                        # pt_embedings = features[int(ins_id)]
                        ids = np.where(label_inst == ins_id)                         
                        if ids[0].shape[0] < 25:                       
                            label_inst[ids] = 0                                  # n_ins < 25 ----> remove
                            continue

                        (values, counts) = np.unique(label_sem_class[ids], return_counts=True)   # 1, 306
                        inst_class = values[np.argmax(counts)]                                   # 1

                        new_ids = remove_outliers(points[ids])                           # (306, )
                        new_ids = ids[0][new_ids]                                        # (262, )  remove outliers

                        bbox, kalman_bbox = get_bbox_from_points(points[ids])            # 17.1, -15.2, -1.2, 18.6, -11.3, -0.07
                        tracker = KalmanBoxTracker(kalman_bbox, ins_id)
                        center = get_median_center_from_points(points[ids])              # 17.2 -12.8 -0.7
                        bbox_proj = get_2d_bbox(projections[:, new_ids])             # remove outliers to get 2D bbox  [1206, 14, 1252, 27]
                        new_instances[ins_id] = {'life': 5, 'bbox': bbox, 'bbox_proj': bbox_proj, 'center' : center, 'n_point':ids[0].shape[0],
                                                'tracker': tracker, 'kalman_bbox': kalman_bbox, 'class' : inst_class}
                    new_instances_prev = {}
                    overlaps = {}
                    overlap_scores = {}


                    # if multi frame prediction
                    times = []
                    times.append(time.time()) # loading time
                    if pls_cfg.n_test_frames > 1 and f_ind > 0:
                        for fi in range(len(f_inc_r_inds_list[b_i])):
                            proj_inds = f_inc_r_inds_list[b_i][fi]                     # 119160
                            proj_mask = f_inc_r_mask_list[b_i][fi]                     # 123389
                            proj_ins_probs = ins_probs[proj_inds]                      # 119160
                            proj_probs = probs[proj_inds]                              # 119160, 19
                            if proj_probs.ndim < 2:
                                proj_ins_probs = np.expand_dims(proj_ins_probs, 0)
                                proj_probs = np.expand_dims(proj_probs, 0)

                            frame_probs_uint8_p = np.zeros((proj_mask.shape[0], nc_model), dtype=np.uint8)
                            frame_probs = frame_probs_uint8_p[proj_mask, :].astype(np.float32) / 255
                            frame_probs = test_smooth * frame_probs + (1 - test_smooth) * proj_probs
                            frame_probs_uint8_p[proj_mask, :] = (frame_probs * 255).astype(np.uint8)

                            ins_preds = np.zeros((proj_mask.shape[0]))
                            ins_preds[proj_mask] = proj_ins_probs                      # 123389

                            for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                                if label_value in test_loader.dataset.ignored_labels:
                                    frame_probs_uint8_p = np.insert(frame_probs_uint8_p, l_ind, 0, axis=1)

                                # Predicted labels
                            frame_preds = test_loader.dataset.label_values[np.argmax(frame_probs_uint8_p,       # 123389
                                                                                     axis=1)].astype(np.int32)

                            filename_i = '{:s}_{:07d}_{}_i.npy'.format(seq_name, f_ind - fi - 1, f_ind)
                            filepath_i = join(test_path, folder, filename_i)
                            # filename_m = '{:s}_{:07d}_{}_m.npy'.format(seq_name, f_ind - fi - 1, f_ind)
                            # filepath_m = join(test_path, folder, filename_m)
                            #('Saving {}'.format(filepath_i))
                            np.save(filepath_i, ins_preds)
                            # np.save(filepath_m, proj_mask)
                            
                            filename_p = '{:s}_{:07d}_{}.npy'.format(seq_name, f_ind-fi-1, f_ind)
                            filepath_p = join(test_path, folder, filename_p)
                            #print('Saving {}'.format(filepath_p))
                            np.save(filepath_p, frame_preds)

                            # ins_features = {}
                            # for ins_id in np.unique(ins_preds):
                            #     if int(ins_id) in self.instances[sequence]:
                            #         pt_idx = np.where(ins_preds==ins_id)[0]
                            #         ins_features[int(ins_id)] = torch.tensor(pt_features[pt_idx]).type(torch.float32)

                            # filename_f = '{:s}_{:07d}_{}_f.npy'.format(seq_name, f_ind-fi-1, f_ind)
                            # filepath_f = join(test_path, folder, filename_f)
                            #np.save(filepath_f, ins_features)

                            ############
                            prediction_path = os.path.join(test_path, folder)

                            prev_inst = ins_preds
                            prev_sem = frame_preds
                            
                            features = {}
                            for ins_id in ins_ids:
                                features[ins_id] = torch.from_numpy(np.zeros((1,1)))

                            prev_inst_orig_path = os.path.join(prediction_path,
                                                        '{0:02d}_{1:07d}_i.npy'.format(int(sequence), f_ind-fi-1))

                            prev_inst_orig = np.load(prev_inst_orig_path)
                            things = (prev_sem < 9) & (prev_sem > 0)

                            association_costs, associations = associate_instances_overlapping_frames(prev_inst_orig* things, prev_inst* things)

                            for cost, (id1, id2) in zip(association_costs, associations):
                                if id2 not in overlaps:
                                    overlap_scores[id2] = cost
                                elif overlap_scores[id2] > cost:
                                    continue
                                elif overlap_scores[id2] < cost:
                                    overlap_scores[id2] = cost
                                if id1 in overlap_history: #get track id of instance from previous frame
                                    id1 = overlap_history[id1]
                                overlaps[id2] = id1

                            prev_point_path = os.path.join(seq_path, 'velodyne', '{0:06d}.bin'.format(f_ind-fi-1))
                            #pose = poses[0][idx-i]
                            frame_points = np.fromfile(prev_point_path, dtype=np.float32)
                            points = frame_points.reshape((-1, 4))
                            hpoints = np.hstack((points[:, :3], np.ones_like(points[:, :1])))
                            new_points = np.sum(np.expand_dims(hpoints, 2) * pose.T, axis=1)
                            points = new_points[:, :3]

                            points = torch.from_numpy(points)
                            projections = do_range_projection(points.cpu().detach().numpy())
                            prev_inst = torch.from_numpy(prev_inst.astype(np.int32))

                            # add instances for assocaition which are not overlapped
                            ins_ids = np.unique(prev_inst * things)
                            for ins_id in ins_ids:
                                if ins_id == 0:
                                    continue
                                if int(ins_id) not in features:
                                    ids = np.where(prev_inst == ins_id)
                                    prev_inst[ids] = 0
                                    continue

                                ids = np.where(prev_inst == ins_id)
                                if ids[0].shape[0] < 25:
                                    prev_inst[ids] = 0
                                    continue

                                # pt_embedings = features[int(ins_id)]
                                new_ids = remove_outliers(points[ids])
                                new_ids = ids[0][new_ids]

                                (values, counts) = np.unique(prev_sem[ids], return_counts=True)
                                inst_class = values[np.argmax(counts)]

                                bbox, kalman_bbox = get_bbox_from_points(points[ids])
                                center = get_median_center_from_points(points[ids])
                                bbox_proj = get_2d_bbox(projections[:, new_ids])
                                tracker = KalmanBoxTracker(kalman_bbox, ins_id)
                                new_instances_prev[ins_id] = {'life': 5, 'bbox': bbox, 'bbox_proj': bbox_proj,
                                                        'tracker': tracker, 'kalman_bbox': kalman_bbox,
                                                            'center':center, 'class' : inst_class}

                    associations = []
                    times.append(time.time()) # assoc time from prev

                    # if there was instances from previous frames
                    if len(prev_instances.keys()) > 0:
                        #firstly associate overlapping instances
                        overlap_history, label_inst, new_instances, prev_instances = associate_overlapped_ins(overlaps, overlap_history, 
                                                                                                label_inst, new_instances, prev_instances)
                        # associate instances which are not overlapped
                        overlap_history, label_inst, new_instances, prev_instances = associate_not_overlapped_ins(associations, overlaps, 
                                                                                            overlap_history, label_inst, new_instances, prev_instances)

                    # add new instances to history
                    for ins_id, instance in new_instances.items():       # [1, 2, 4, 5, 6 ,7, 11], [17, 18, 23, 24, 28]
                        ids = np.where(label_inst == ins_id)
                        if ids[0].shape[0] < 50:                   # drop ins 11
                            continue
                        prev_instances[ins_id] = instance          # n_ins > 50 add into ---> prev_ins

                    # kill instances which are not tracked for a  while
                    dont_track_ids = []
                    for ins_id in prev_instances.keys():          # [1, 2, 4, 5, 6 ,7], [1, 2, 4, 5, 6 ,7, 17, 18, 23, 24, 28]
                        if prev_instances[ins_id]['life'] == 0:
                            dont_track_ids.append(ins_id)
                        prev_instances[ins_id]['life'] -= 1

                    for ins_id in dont_track_ids:
                        del prev_instances[ins_id]

                    times.append(time.time()) # updating ids

                    ins_preds = label_inst.cpu().numpy()         # (123389,)    # [0, 43, 44, 46, 47, 48, 49, 50] remove the ins with number less than 25

                    #clean instances which have too few points
                    for ins_id in np.unique(ins_preds):              # [0, 1, 2, 4, 5, 6, 7, 11], [0, 1, 2, 4, 6, 17, 18, 23, 24, 28]
                        if ins_id == 0:
                            continue
                        valid_ind = np.argwhere(ins_preds == ins_id)[:, 0]     # ins_id = 1 ----> n_ins = 306
                        # ins_preds[valid_ind] = ins_id+20                       # ??????????   43 -----> 63   ??????????
                        if valid_ind.shape[0] < 25:
                            ins_preds[valid_ind] = 0              
                                                                            # [0, 21, 22, 24, 25, 26, 27, 31], [0, 21, 22, 26, 37, 38, 43, 44, 48]
                    ###############################################################################
                    frame_points = np.fromfile(velo_file, dtype=np.float32)                         # 123433
                    frame_points = frame_points.reshape((-1, 4))                                    # 123433
                    proj_mask = r_mask_list[b_i]
                    frame_preds = label_sem_class


                    frame_preds = frame_preds.astype(np.int64)
                    ins_preds = ins_preds.astype(np.int32)
                    sem_pred = pred.majority_voting(frame_preds, ins_preds)

                    ############################# CA-net predictions
                    raw_features = pt_features.astype(np.float32) 
                    pt_coors = frame_points[:, :3]

                    ins_feat, n_ins, ins_ids, ins_pred, tracking_input = self.get_ins_feat(pt_coors, ins_preds, raw_features)

                    if len(ins_feat)!=0:
                        ins_pred = self.track(ins_pred, ins_feat, n_ins, ins_ids, tracking_input, [pose])[0]
                    else:
                        ins_pred = ins_pred[0]

                    # save prediction ins and sem_label results for test

                    sem_tsne =[]   
                    ############

                    #clean instances which have too few points
                    for ins_id in np.unique(ins_pred):              # [0, 1, 2, 4, 5, 6, 7, 11], [0, 1, 2, 4, 6, 17, 18, 23, 24, 28]
                        if ins_id == 0:
                            continue
                        valid_ind = np.argwhere(ins_pred == ins_id)[:, 0]     # ins_id = 1 ----> n_ins = 306
                        sem_tsne.append(sem_pred[valid_ind[0]])
                        ins_pred[valid_ind] = ins_id+20                       # ??????????   43 -----> 63   ??????????
                        if valid_ind.shape[0] < 25:
                            ins_pred[valid_ind] = 0              
                                                                            # [0, 21, 22, 24, 25, 26, 27, 31], [0, 21, 22, 26, 37, 38, 43, 44, 48]
                    for sem_id in np.unique(sem_pred):                  # [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] without 8
                        if sem_id < 1 or sem_id > 8:                           # sem_id [0, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
                            valid_ind = np.argwhere((sem_pred == sem_id) & (ins_pred == 0))[:, 0]    # (4194,) semantic class = 0 and ins_id = 0
                            ins_pred[valid_ind] = sem_id                      # label instance which doesn't belong to things  

                    #write instances to label file which is binary
                    ins_preds = ins_pred.astype(np.int32)
                    new_preds = np.left_shift(ins_preds, 16)

                    sem_preds = sem_pred.astype(np.int32)
                    inv_sem_labels = inv_learning_map[sem_preds]
                    new_preds = np.bitwise_or(new_preds,inv_sem_labels)


                    result_folder = 'pred_result'
                    save_path = os.path.join(test_path, result_folder)
                    if not os.path.exists(os.path.join(save_path, 'sequences', '{0:02d}'.format(int(sequence)), 'predictions')):
                        os.makedirs(os.path.join(save_path, 'sequences', '{0:02d}'.format(int(sequence)), 'predictions'))

                    new_preds.tofile('{}/{}/{:02d}/predictions/{:06d}.label'.format(save_path, 'sequences', int(sequence), f_ind))

                    
                    # return sem_pred, ins_pred

                    # if pls_cfg.SAVE_TRAIN_FEATURES:
                    #     save_features(test_loader, batch, s_ind, f_ind, frame_points, pt_features, proj_mask, ins_preds, frame_preds, save_preds = False)
                    # elif pls_cfg.SAVE_VAL_PRED:
                    #     save_features(test_loader, batch, s_ind, f_ind, frame_points, pt_features, proj_mask, ins_preds, frame_preds, save_preds = True)
                    if ins_feat == []: # need to modify
                        np_tsne = np.array([sem_tsne, ins_ids, ins_feat], dtype=object)
                    else:
                        ins_feat_list = [ins_feat.cpu().detach().numpy()]
                        np_tsne = np.array([sem_tsne, ins_ids[0], ins_feat_list], dtype=object)

                    filename_tsne = '{:s}_{:07d}.npy'.format(seq_name, f_ind)
                    tsne_path = '/_data/tsne/'
                    filepath_tsne = join(tsne_path, seq_path[-12:], filename_tsne)
                    if not os.path.exists(join(tsne_path, seq_path[-12:])):
                        os.makedirs(join(tsne_path, seq_path[-12:]))

                    np.save(filepath_tsne, np_tsne, allow_pickle=True)

                    ##################################################################################
                    # # Save some prediction in ply format for visual
                    if test_loader.dataset.set == 'validation':

                        # Insert false columns for ignored labels
                        frame_probs_uint8_bis = frame_probs_uint8.copy()
                        for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                            if label_value in test_loader.dataset.ignored_labels:
                                frame_probs_uint8_bis = np.insert(frame_probs_uint8_bis, l_ind, 0, axis=1)

                        # Predicted labels
                        frame_preds = test_loader.dataset.label_values[np.argmax(frame_probs_uint8_bis,
                                                                                 axis=1)].astype(np.int32)

                        np.save(filepath, frame_preds)
                        # Save some of the frame pots
                        if f_ind % 20 == 0:
                            seq_path = join(test_loader.dataset.path, 'sequences', test_loader.dataset.sequences[s_ind])
                            velo_file = join(seq_path, 'velodyne', test_loader.dataset.frames[s_ind][f_ind] + '.bin')
                            frame_points = np.fromfile(velo_file, dtype=np.float32)
                            frame_points = frame_points.reshape((-1, 4))
                            predpath = join(test_path, pred_folder, filename[:-4] + '.ply')
                            #pots = test_loader.dataset.f_potentials[s_ind][f_ind]
                            pots = np.zeros((0,))
                            if pots.shape[0] > 0:
                                write_ply(predpath,
                                          [frame_points[:, :3], frame_labels, frame_preds, pots],
                                          ['x', 'y', 'z', 'gt', 'pre', 'pots'])
                            else:
                                write_ply(predpath,
                                          [frame_points[:, :3], frame_labels, frame_preds],
                                          ['x', 'y', 'z', 'gt', 'pre'])

                            # Also Save lbl probabilities
                            probpath = join(test_path, folder, filename[:-4] + '_probs.ply')
                            lbl_names = [test_loader.dataset.label_to_names[l]
                                         for l in test_loader.dataset.label_values
                                         if l not in test_loader.dataset.ignored_labels]
                            write_ply(probpath,
                                      [frame_points[:, :3], frame_probs_uint8],
                                      ['x', 'y', 'z'] + lbl_names)

                        # keep frame preds in memory
                        all_f_preds[s_ind][f_ind] = frame_preds
                        all_f_labels[s_ind][f_ind] = frame_labels

                    else:

                        # Insert false columns for ignored labels
                        for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                            if label_value in test_loader.dataset.ignored_labels:
                                frame_probs_uint8 = np.insert(frame_probs_uint8, l_ind, 0, axis=1)

                        # Predicted labels
                        frame_preds = test_loader.dataset.label_values[np.argmax(frame_probs_uint8,
                                                                                 axis=1)].astype(np.int32)
                        np.save(filepath, frame_preds)
                        if f_inds[b_i, 1] % 100 == 0:
                            # Load points
                            seq_path = join(test_loader.dataset.path, 'sequences', test_loader.dataset.sequences[s_ind])
                            velo_file = join(seq_path, 'velodyne', test_loader.dataset.frames[s_ind][f_ind] + '.bin')
                            frame_points = np.fromfile(velo_file, dtype=np.float32)
                            frame_points = frame_points.reshape((-1, 4))
                            predpath = join(test_path, pred_folder, filename[:-4] + '.ply')
                            #pots = test_loader.dataset.f_potentials[s_ind][f_ind]
                            pots = np.zeros((0,))
                            if pots.shape[0] > 0:
                                write_ply(predpath,
                                          [frame_points[:, :3], frame_preds, pots],
                                          ['x', 'y', 'z', 'pre', 'pots'])
                            else:
                                write_ply(predpath,
                                          [frame_points[:, :3], frame_preds],
                                          ['x', 'y', 'z', 'pre'])

                    ######################################################################################
                    # Stack all prediction for this epoch
                    i0 += length


                # Average timing
                t += [time.time()]
                mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

                # Display
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'e{:03d}-i{:04d} => {:.1f}% (timings : {:4.2f} {:4.2f} {:4.2f}) / pots {:d} => {:.1f}%'
                    min_pot = int(torch.floor(torch.min(test_loader.dataset.potentials)))
                    pot_num = torch.sum(test_loader.dataset.potentials > min_pot + 0.5).type(torch.int32).item()
                    current_num = pot_num + (i + 1 - pls_cfg.validation_size) * pls_cfg.val_batch_num
                    print(message.format(test_epoch, i,
                                         100 * i / pls_cfg.validation_size,
                                         1000 * (mean_dt[0]),
                                         1000 * (mean_dt[1]),
                                         1000 * (mean_dt[2]),
                                         min_pot,
                                         100.0 * current_num / len(test_loader.dataset.potentials)))


            # Update minimum od potentials
            new_min = torch.min(test_loader.dataset.potentials)
            print('Test epoch {:d}, end. Min potential = {:.1f}'.format(test_epoch, new_min))

            if last_min + 1 < new_min:

                # Update last_min
                last_min += 1

                if test_loader.dataset.set == 'validation' and last_min % 1 == 0:

                    #####################################
                    # Results on the whole validation set
                    #####################################

                    # Confusions for our subparts of validation set
                    Confs = np.zeros((len(predictions), nc_tot, nc_tot), dtype=np.int32)
                    for i, (preds, truth) in enumerate(zip(predictions, targets)):

                        # Confusions
                        Confs[i, :, :] = fast_confusion(truth, preds, test_loader.dataset.label_values).astype(np.int32)


                    # Show vote results
                    print('\nCompute confusion')

                    val_preds = []
                    val_labels = []
                    t1 = time.time()
                    for i, seq_frames in enumerate(test_loader.dataset.frames):
                        val_preds += [np.hstack(all_f_preds[i])]
                        val_labels += [np.hstack(all_f_labels[i])]
                    val_preds = np.hstack(val_preds)
                    val_labels = np.hstack(val_labels)
                    t2 = time.time()
                    C_tot = fast_confusion(val_labels, val_preds, test_loader.dataset.label_values)
                    t3 = time.time()
                    print(' Stacking time : {:.1f}s'.format(t2 - t1))
                    print('Confusion time : {:.1f}s'.format(t3 - t2))

                    s1 = '\n'
                    for cc in C_tot:
                        for c in cc:
                            s1 += '{:7.0f} '.format(c)
                        s1 += '\n'
                    if debug:
                        print(s1)

                    # Remove ignored labels from confusions
                    for l_ind, label_value in reversed(list(enumerate(test_loader.dataset.label_values))):
                        if label_value in test_loader.dataset.ignored_labels:
                            C_tot = np.delete(C_tot, l_ind, axis=0)
                            C_tot = np.delete(C_tot, l_ind, axis=1)

                    # Objects IoU
                    val_IoUs = IoU_from_confusions(C_tot)

                    # Compute IoUs
                    mIoU = np.mean(val_IoUs)
                    s2 = '{:5.2f} | '.format(100 * mIoU)
                    for IoU in val_IoUs:
                        s2 += '{:5.2f} '.format(100 * IoU)
                    print(s2 + '\n')

                    # Save a report
                    report_file = join(report_path, 'report_{:04d}.txt'.format(int(np.floor(last_min))))
                    strg = 'Report of the confusion and metrics\n'
                    strg += '***********************************\n\n\n'
                    strg += 'Confusion matrix:\n\n'
                    strg += s1
                    strg += '\nIoU values:\n\n'
                    strg += s2
                    strg += '\n\n'
                    with open(report_file, 'w') as f:
                        f.write(strg)

            test_epoch += 1

            # Break when reaching number of desired votes
            if last_min > num_votes:
                break

        return


