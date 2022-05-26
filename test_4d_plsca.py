#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Callable script to start a training on ModelNet40 dataset
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Zixuan Chen - 2022
#

# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

# Common libs
# import pdb
# pdb.set_trace()
import signal
import os
import numpy as np
import sys
import torch
from easydict import EasyDict as edict
# Dataset
from datasets.SemanticKitti import *
from torch.utils.data import DataLoader

from utils.config import Config
from utils.plsca_tester import ModelTester
from models.architectures import KPCNN, KPFCNN

import cont_assoc.models.contrastive_models as ca_models
import cont_assoc.models.contrastive_models as c_models


np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

def model_choice(chosen_log):

    ###########################
    # Call the test initializer
    ###########################

    # Automatically retrieve the last trained model
    if chosen_log in ['last_ModelNet40', 'last_ShapeNetPart', 'last_S3DIS']:

        # Dataset name
        test_dataset = '_'.join(chosen_log.split('_')[1:])

        # List all training logs
        logs = np.sort([os.path.join('results', f) for f in os.listdir('results') if f.startswith('Log')])

        # Find the last log of asked dataset
        for log in logs[::-1]:
            log_config = Config()
            log_config.load(log)
            if log_config.dataset.startswith(test_dataset):
                chosen_log = log
                break

        if chosen_log in ['last_ModelNet40', 'last_ShapeNetPart', 'last_S3DIS']:
            raise ValueError('No log of the dataset "' + test_dataset + '" found')

    # Check if log exists
    if not os.path.exists(chosen_log):
        raise ValueError('The given log does not exists: ' + chosen_log)

    return chosen_log


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

if __name__ == '__main__':

    ###############################
    # Choose the model to visualize
    ###############################

    #   Here you can choose which model you want to test with the variable test_model. Here are the possible values :
    #
    #       > 'last_XXX': Automatically retrieve the last trained model on dataset XXX
    #       > '(old_)results/Log_YYYY-MM-DD_HH-MM-SS': Directly provide the path of a trained model

    chosen_log = 'results/Log_2020-10-06_16-51-05'  # => ModelNet40

    # Choose the index of the checkpoint to load OR None if you want to load the current checkpoint
    chkp_idx = None

    # Choose to test on validation or test split
    on_val = True

    # Deal with 'last_XXXXXX' choices
    chosen_log = model_choice(chosen_log)

    ############################
    # Initialize the environment
    ############################

    # Set which gpu is going to be used
    GPU_ID = '0'
    if torch.cuda.device_count() > 1:
        GPU_ID = '0, 1'

    ###############
    # Previous chkp
    ###############

    # Find all checkpoints in the chosen training folder
    chkp_path = os.path.join(chosen_log, 'checkpoints')
    chkps = [f for f in os.listdir(chkp_path) if f[:4] == 'chkp']

    # Find which snapshot to restore
    if chkp_idx is None:
        chosen_chkp = 'current_chkp.tar'
    else:
        chosen_chkp = np.sort(chkps)[chkp_idx]
    chosen_chkp = os.path.join(chosen_log, 'checkpoints', chosen_chkp)

    # Initialize configuration class
    pls_cfg = Config()
    pls_cfg.load(chosen_log)


    ##################################
    # Change model parameters for test
    ##################################

    # Change parameters for the test here. For example, you can stop augmenting the input data.

    pls_cfg.global_fet = False
    pls_cfg.validation_size = 200
    pls_cfg.input_threads = 16
    pls_cfg.n_frames = 4
    pls_cfg.n_test_frames = 4 #it should be smaller than pls_cfg.n_frames
    if pls_cfg.n_frames < pls_cfg.n_test_frames:
        pls_cfg.n_frames = pls_cfg.n_test_frames
    pls_cfg.big_gpu = True
    pls_cfg.dataset_task = '4d_panoptic'
    #pls_cfg.sampling = 'density'
    pls_cfg.sampling = 'importance'
    pls_cfg.decay_sampling = 'None'
    pls_cfg.stride = 1
    pls_cfg.first_subsampling_dl = 0.061


    ##############
    # Prepare Data
    ##############

    print()
    print('Data Preparation')
    print('****************')

    if on_val:
        set = 'validation'
    else:
        set = 'test'

    # Initiate dataset
    if pls_cfg.dataset.startswith('ModelNet40'):
        test_dataset = ModelNet40Dataset(pls_cfg, train=False)
        test_sampler = ModelNet40Sampler(test_dataset)
        collate_fn = ModelNet40Collate
    elif pls_cfg.dataset == 'S3DIS':
        test_dataset = S3DISDataset(pls_cfg, set='validation', use_potentials=True)
        test_sampler = S3DISSampler(test_dataset)
        collate_fn = S3DISCollate
    elif pls_cfg.dataset == 'SemanticKitti':
        test_dataset = SemanticKittiDataset(pls_cfg, set=set, balance_classes=False, seqential_batch=True)
        test_sampler = SemanticKittiSampler(test_dataset)
        collate_fn = SemanticKittiCollate
    else:
        raise ValueError('Unsupported dataset : ' + pls_cfg.dataset)

    # Data loader
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             sampler=test_sampler,
                             collate_fn=collate_fn,
                             num_workers=0,#pls_cfg.input_threads,
                             pin_memory=True)

    # Calibrate samplers
    test_sampler.calibration(test_loader, verbose=True)

    print('\nModel Preparation')
    print('*****************')

    # Define network model
    t1 = time.time()
    if pls_cfg.dataset_task == 'classification':
        pls_net = KPCNN(pls_cfg)
    elif pls_cfg.dataset_task in ['cloud_segmentation', 'slam_segmentation']:
        pls_net = KPFCNN(pls_cfg, test_dataset.label_values, test_dataset.ignored_labels)
    else:
        raise ValueError('Unsupported dataset_task for testing: ' + pls_cfg.dataset_task)


    ## aggregation config
    config_ag = 'config/contrastive_instances.yaml'            ##############
    ag_cfg = edict(yaml.safe_load(open(config_ag)))

    ca_net = c_models.ContrastiveTracking(ag_cfg)
    ca_chkp_path = 'experiments/CA-Net/default/version_1/checkpoints/last.ckpt'

    # Define a visualizer class
    tester = ModelTester(pls_net, ca_net, pls_chkp_path=chosen_chkp, ca_chkp_path=ca_chkp_path)              ############ need to modify
    print('Done in {:.1f}s\n'.format(time.time() - t1))

    print('\nStart test')
    print('**********\n')
    
    pls_cfg.dataset_task = '4d_panoptic'

    tester.panoptic_4d_test(test_loader, pls_cfg, ag_cfg)

    ##############################################
    # model.panoptic_model.evaluator.print_results()
    # print("#############################################################")
    # model.evaluator4D.calculate_metrics()
    # model.evaluator4D.print_results()

    
    
    # # Training
    # if config.dataset_task == 'classification':
    #     a = 1/0
    # elif config.dataset_task == 'cloud_segmentation':
    #             tester.cloud_segmentation_test(net, test_loader, config)
    # elif config.dataset_task == 'slam_segmentation':
    #     tester.slam_segmentation_test(net, test_loader, config)
    # elif config.dataset_task == '4d_panoptic':
    #     tester.panoptic_4d_test(net, test_loader, config)
    # else:
    #     raise ValueError('Unsupported dataset_task for testing: ' + config.dataset_task)
