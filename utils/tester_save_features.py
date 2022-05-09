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
#

# import pdb
# pdb.set_trace()
from utils.save_features import save_features

# Basic libs
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

#from utils.visualizer import show_ModelNet_models

# ----------------------------------------------------------------------------------------------------------------------
#
#           Tester Class
#       \******************/
#


class ModelTester:

    # Initialization methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, net, chkp_path=None, on_gpu=True):

        ############
        # Parameters
        ############

        self.instances = {} #store instance ids and mean, cov fors sequantial prediction
        self.next_ins_id = 1 #next ins id for new instance

        # Choose to train on CPU or GPU
        if on_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        net.to(self.device)

        ##########################
        # Load previous checkpoint
        ##########################

        checkpoint = torch.load(chkp_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        self.epoch = checkpoint['epoch']
        net.eval()
        print("Model and training state restored.")

        return

    # Test main methods
    # ------------------------------------------------------------------------------------------------------------------
    def panoptic_4d_test(self, net, test_loader, config, num_votes=100, debug=True):
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

        # Number of classes including ignored labels
        nc_tot = test_loader.dataset.num_classes
        nc_model = net.C
        # Test saving path
        test_path = None
        report_path = None

        if config.dataset_task == '4d_panoptic':

            #assoc_saving = [asc_type for idx, asc_type in enumerate(config.association_types) if config.association_weights[idx] > 0]
            #assoc_saving.append(str(config.n_test_frames))
            #assoc_saving = '_'.join(assoc_saving)
            assoc_saving = config.sampling
            config.assoc_saving = config.sampling+'_'+ config.decay_sampling
            if hasattr(config, 'stride'):
                config.assoc_saving = config.sampling + '_' + config.decay_sampling+ '_str' + str(config.stride) +'_'
            if hasattr(config, 'big_gpu') and config.big_gpu:
                config.assoc_saving = config.assoc_saving + 'bigpug_'


        if config.saving:
            test_path = join('test', config.saving_path.split('/')[-1]+ '_'+config.assoc_saving+str(config.n_test_frames))
            if not exists(test_path):
                makedirs(test_path)
            report_path = join(test_path, 'reports')
            if not exists(report_path):
                makedirs(report_path)

        if test_loader.dataset.set == 'validation':
            for folder in ['val_predictions', 'val_probs']:
                if not exists(join(test_path, folder)):
                    makedirs(join(test_path, folder))
        else:
            for folder in ['predictions', 'probs']:
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
        # Network predictions
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
                if config.n_test_frames > 1:
                    lengths = batch.lengths[0].cpu().numpy()
                    for b_i, length in enumerate(lengths):
                        f_inds = batch.frame_inds.cpu().numpy()
                        f_ind = f_inds[b_i, 1]
                        if f_ind % config.n_test_frames != config.n_test_frames-1:
                             flag = False

                if processed == test_loader.dataset.all_inds.shape[0]:
                    return
                #if not flag:
                #    continue
                #else:
                processed +=1

                if 'cuda' in self.device.type:
                    batch.to(self.device)

                with torch.no_grad():

                    outputs, centers_output, var_output, embedding = net(batch, config)    # (83365, 19); (N, 1); (N, 260); (N, 256)
                    #ins_preds = torch.zeros(outputs.shape[0])

                    probs = softmax(outputs).cpu().detach().numpy()                        # (83365, 19)

                    for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                        if label_value in test_loader.dataset.ignored_labels:              # ignore_labels = 0
                            probs = np.insert(probs, l_ind, 0, axis=1)                     # (83365, 20) add 0 in the first column
                    preds = test_loader.dataset.label_values[np.argmax(probs, axis=1)]     # (83365, )   value between [1, 19]
                    preds = torch.from_numpy(preds)
                    preds.to(outputs.device)
                    sequence = test_loader.dataset.sequences[batch.frame_inds[0][0]]       #  '08'
                    pose = test_loader.dataset.poses[batch.frame_inds[0][0]][batch.frame_inds[0][1]]   # (4, 4) transformer matrix
                    if sequence not in self.instances:
                        self.instances[sequence] = {}
                    #ins_preds = net.ins_pred(preds, centers_output, var_output, embedding, batch.points)
                    ins_preds, new_instances, ins_id = net.ins_pred_in_time(config, preds, centers_output, var_output, embedding, self.instances[sequence],  #   (166205) 0-20;     ; 21 
                                                     self.next_ins_id, batch.points, batch.times.unsqueeze(1), pose)                         # return: instance ids for all points, and new instances and next available ins_id

                    self.next_ins_id = ins_id                      #update next available ins id
                    for ins_id, instance in new_instances.items(): #add new instances to history    # instance.keys(): mean, var, life, bbox 6, bbox_proj, tracker, kalman_bbox (7, )
                        self.instances[sequence][ins_id] = instance          # ins_id : (1,2,3,4,5,6,7,9,10), (),(15, 16, 17, 18, 19, 20) ; self.instances[sequence] is dict

                    dont_track_ids = []
                    for ins_id in self.instances[sequence].keys():
                        if self.instances[sequence][ins_id]['life'] == 0:
                            dont_track_ids.append(ins_id)
                        self.instances[sequence][ins_id]['life'] -= 1

                    for ins_id in dont_track_ids:
                        del self.instances[sequence][ins_id]

                # Get probs and labels                                   # N = 83365
                embedding = embedding.cpu().detach().numpy()             # (N, 256)              
                stk_probs = softmax(outputs).cpu().detach().numpy()      # (N, 19)  83365
                ins_preds = ins_preds.cpu().detach().numpy()             # (N, )    83365
                centers_output = centers_output.cpu().detach().numpy()   # (N, 1)  ; between [0, 1]   83365
                lengths = batch.lengths[0].cpu().numpy()                 # 83365
                f_inds = batch.frame_inds.cpu().numpy()                  # [[0, 1]]
                r_inds_list = batch.reproj_inds                          # (119195, )  between [0, 83364]
                r_mask_list = batch.reproj_masks                         # (123389, )  True or False ; np.count_nonzero(proj_mask == True) --> 119195
                f_inc_r_inds_list = batch.f_inc_reproj_inds              # 
                f_inc_r_mask_list = batch.f_inc_reproj_masks             # 

                labels_list = batch.val_labels                           # between [0, 19]   123389 
                torch.cuda.synchronize(self.device)

                t += [time.time()]

                # Get predictions and labels per instance
                # ***************************************

                i0 = 0
                for b_i, length in enumerate(lengths):                  # batch length 83365

                    # Get prediction
                    pt_features = embedding[i0:i0 + length]             # (83365, 256)

                    probs = stk_probs[i0:i0 + length]                   # (83365, 19)
                    c_probs = centers_output[i0:i0 + length]            # (83365)
                    ins_probs = ins_preds[i0:i0 + length]               # (83365)
                    proj_inds = r_inds_list[b_i]                        # (119195, )   between [0, 83364]
                    proj_mask = r_mask_list[b_i]                        # (123389); ture or false; np.count_nonzero(proj_mask == True) --> 119195
                    frame_labels = labels_list[b_i]                     # 123389
                    s_ind = f_inds[b_i, 0]
                    f_ind = f_inds[b_i, 1]

                    # Project predictions on the frame points           # proj_inds: (119195, )   between [0, 83364]
                    proj_probs = probs[proj_inds]                       # 119195
                    proj_c_probs = c_probs[proj_inds]                   # 119195
                    proj_ins_probs = ins_probs[proj_inds]               # 119195
                    proj_pt_features = pt_features[proj_inds]

                    # Safe check if only one point:
                    if proj_probs.ndim < 2:
                        proj_c_probs = np.expand_dims(proj_c_probs, 0)
                        proj_probs = np.expand_dims(proj_probs, 0)
                        proj_ins_probs = np.expand_dims(proj_ins_probs, 0)

                    # Save probs in a binary file (uint8 format for lighter weight)
                    seq_name = test_loader.dataset.sequences[s_ind]
                    if test_loader.dataset.set == 'validation':
                        folder = 'val_probs'
                        pred_folder = 'val_predictions'
                    else:  # test
                        folder = 'probs'
                        pred_folder = 'predictions'
                    filename = '{:s}_{:07d}.npy'.format(seq_name, f_ind)
                    filepath = join(test_path, folder, filename)
                    filename_i = '{:s}_{:07d}_i.npy'.format(seq_name, f_ind)
                    filename_c = '{:s}_{:07d}_c.npy'.format(seq_name, f_ind)
                    filepath_i = join(test_path, folder, filename_i)
                    filepath_c = join(test_path, folder, filename_c)
                    #if exists(filepath):
                    #    frame_probs_uint8 = np.load(filepath)
                    #    ins_preds = np.load(filepath_i)
                    #else:

                    frame_probs_uint8 = np.zeros((proj_mask.shape[0], nc_model), dtype=np.uint8)    # (123389, 19)
                    frame_c_probs = np.zeros((proj_mask.shape[0], 1))                               # (123389, 1)
                    ins_preds = np.zeros((proj_mask.shape[0]), dtype=np.uint8)                        # (123389,)
                    pt_features = np.zeros((proj_mask.shape[0], 256))                               # 123389

                    frame_probs = frame_probs_uint8[proj_mask, :].astype(np.float32) / 255          # proj_mask: true or false
                    frame_probs = test_smooth * frame_probs + (1 - test_smooth) * proj_probs        # np.count_nonzero(proj_mask == True) --> 126340
                    frame_probs_uint8[proj_mask, :] = (frame_probs * 255).astype(np.uint8)          # (123389, 19)
                    ins_preds[proj_mask] = proj_ins_probs                                           # (123389, )
                    frame_c_probs[proj_mask] = proj_c_probs                                         # (123389, 1)
                    pt_features[proj_mask] = proj_pt_features                                       # (123389, 256)

                    frame_probs_uint8_p = frame_probs_uint8.copy()
                    for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                        if label_value in test_loader.dataset.ignored_labels:
                            frame_probs_uint8_p = np.insert(frame_probs_uint8_p, l_ind, 0, axis=1)

                    # Predicted labels
                    frame_preds = test_loader.dataset.label_values[np.argmax(frame_probs_uint8_p,
                                                                                     axis=1)].astype(np.int32)

                    seq_path = join(test_loader.dataset.path, 'sequences', test_loader.dataset.sequences[s_ind])
                    velo_file = join(seq_path, 'velodyne', test_loader.dataset.frames[s_ind][f_ind] + '.bin')
                    frame_points = np.fromfile(velo_file, dtype=np.float32)                         # (123398)
                    frame_points = frame_points.reshape((-1, 4))

                    ############################################

                    if config.SAVE_FEATURES:
                        save_features(test_loader, batch, s_ind, f_ind, frame_points, pt_features, proj_mask, ins_preds, frame_preds, save_preds = False)
                    elif config.SAVE_VAL_PRED:
                        save_features(test_loader, batch, s_ind, f_ind, frame_points, pt_features, proj_mask, ins_preds, frame_preds, save_preds = True)

                    ############################################

                    
                    #np.save(filepath, frame_probs_uint8)
                    #print ('Saving {}'.format(filepath_i))
                    # np.save(filepath_i, ins_preds)
                    # np.save(filepath_c, frame_c_probs)

                    ins_features = {}                                                               # ins_features.keys()  [1,2,3,4,5,6,7,9,10]
                    for ins_id in np.unique(ins_preds):
                        if int(ins_id) in self.instances[sequence]:
                            ins_features[int(ins_id)] = self.instances[sequence][int(ins_id)]['mean']     # (1, 260)
                    filename_f = '{:s}_{:07d}_f.npy'.format(seq_name, f_ind)
                    filepath_f = join(test_path, folder, filename_f)
                    # np.save(filepath_f, ins_features)

                    # if config.n_test_frames > 1:                                                 ??????????????????????????
                    #     for fi in range(len(f_inc_r_inds_list[b_i])):
                    #         proj_inds = f_inc_r_inds_list[b_i][fi]                                  # (126423, )
                    #         proj_mask = f_inc_r_mask_list[b_i][fi]                                  # (127351, )
                    #         proj_ins_probs = ins_probs[proj_inds]                                   # (126423, )
                    #         proj_probs = probs[proj_inds]
                    #         if proj_probs.ndim < 2:
                    #             proj_ins_probs = np.expand_dims(proj_ins_probs, 0)
                    #             proj_probs = np.expand_dims(proj_probs, 0)

                    #         frame_probs_uint8_p = np.zeros((proj_mask.shape[0], nc_model), dtype=np.uint8)
                    #         frame_probs = frame_probs_uint8_p[proj_mask, :].astype(np.float32) / 255
                    #         frame_probs = test_smooth * frame_probs + (1 - test_smooth) * proj_probs
                    #         frame_probs_uint8_p[proj_mask, :] = (frame_probs * 255).astype(np.uint8)

                    #         ins_preds = np.zeros((proj_mask.shape[0]))
                    #         ins_preds[proj_mask] = proj_ins_probs

                    #         for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                    #             if label_value in test_loader.dataset.ignored_labels:
                    #                 frame_probs_uint8_p = np.insert(frame_probs_uint8_p, l_ind, 0, axis=1)

                    #             # Predicted labels
                    #         frame_preds = test_loader.dataset.label_values[np.argmax(frame_probs_uint8_p,
                    #                                                                  axis=1)].astype(np.int32)

                    #         filename_i = '{:s}_{:07d}_{}_i.npy'.format(seq_name, f_ind - fi - 1, f_ind)
                    #         filepath_i = join(test_path, folder, filename_i)
                    #         #('Saving {}'.format(filepath_i))
                    #         np.save(filepath_i, ins_preds)

                    #         filename_p = '{:s}_{:07d}_{}.npy'.format(seq_name, f_ind-fi-1, f_ind)
                    #         filepath_p = join(test_path, folder, filename_p)
                    #         #print('Saving {}'.format(filepath_p))
                    #         np.save(filepath_p, frame_preds)       

                    #         ins_features = {}
                    #         for ins_id in np.unique(ins_preds):
                    #             if int(ins_id) in self.instances[sequence]:
                    #                 ins_features[int(ins_id)] = self.instances[sequence][int(ins_id)]['mean']
                    #         filename_f = '{:s}_{:07d}_{}_f.npy'.format(seq_name, f_ind-fi-1, f_ind)
                    #         filepath_f = join(test_path, folder, filename_f)
                    #         #np.save(filepath_f, ins_features)

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
                    current_num = pot_num + (i + 1 - config.validation_size) * config.val_batch_num
                    print(message.format(test_epoch, i,
                                         100 * i / config.validation_size,
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
























