from classify import *
from generator import *
from discri import *
from torch.nn import  DataParallel
import torch
import time
import random
import os, logging
import numpy as np
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import yaml
from brep_mi import attack



if __name__ == "__main__":
    global args, logger

    parser = ArgumentParser(description='A tool that applies Label Only Model Inversion Attack using labels only.')
    parser.add_argument('--target_model', default='FaceNet64', help='VGG16 | IR152 | FaceNet64')
    parser.add_argument('--target_model_path', type=str, help='path to target_model')
    parser.add_argument('--evaluator_model', default='FaceNet', help='VGG16 | IR152 | FaceNet64| FaceNet')
    parser.add_argument('--evaluator_model_path', default='models/target_ckp/FaceNet_95.88.tar', help='path to evaluator_model')
    parser.add_argument('--generator_model_path', type=str, help='path to generator model')
    parser.add_argument('--device', type=str, default='0', help='Device to use. Like cuda, cuda:0 or cpu')
    parser.add_argument('--experiment_name', type=str, default='default_test1', help='experiment name for experiment directory', required = True)
    
    parser.add_argument('--config_file', type=str, help='config file that has attack params', required = True)
    parser.add_argument('--private_imgs_path', type=str, default='', help='Path to groundtruth images to copy them to attack dir. Empty string means, our tool will not copy.')
    parser.add_argument('--n_classes', type=int, default=1000, help='num of classes of target model')
    parser.add_argument('--n_classes_evaluator', type=int, default=1000, help='num of classes of evaluator model')
    args = parser.parse_args()

    print(args)
    print("=> loading models ...")    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    
    
    # loading attack params
    with open (args.config_file) as config_file:
        attack_params = yaml.load(config_file)        
    print (attack_params)
    
    
    #loading the models
    n_classes = args.n_classes

    if args.target_model.startswith("VGG16"):
        target_model = VGG16(n_classes)
    elif args.target_model.startswith('IR152'):
        target_model = IR152(n_classes)
    elif args.target_model == "FaceNet64":
        target_model = FaceNet64(n_classes)
        
    path_target_model = args.target_model_path
    target_model = torch.nn.DataParallel(target_model).cuda()
    ckp_target_model = torch.load(path_target_model)
    target_model.load_state_dict(ckp_target_model['state_dict'], strict=False)
    
   
    path_G = args.generator_model_path
    G = Generator(attack_params['z_dim'])
    G = torch.nn.DataParallel(G).cuda()
    ckp_G = torch.load(path_G)
    G.load_state_dict(ckp_G['state_dict'], strict=False)

    if args.evaluator_model == 'FaceNet':
        E = FaceNet(args.n_classes_evaluator)
    elif args.evaluator_model == 'FaceNet64':
        E = FaceNet64(args.n_classes_evaluator)
        
    E = torch.nn.DataParallel(E).cuda()
    path_E = args.evaluator_model_path
    ckp_E = torch.load(path_E)
    E.load_state_dict(ckp_E['state_dict'], strict=False)
    
    
    

    target_model.eval()
    G.eval()
    E.eval()
    
    
    # prepare working dirs
    attack_imgs_dir = 'decision/attack_imgs/'+args.experiment_name
    os.makedirs(attack_imgs_dir, exist_ok = True)
    
    
    # do the attack
    attack( attack_params,
            target_model,
            E,
            G,
            attack_imgs_dir,
            args.private_imgs_path)
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    


    
   
    
