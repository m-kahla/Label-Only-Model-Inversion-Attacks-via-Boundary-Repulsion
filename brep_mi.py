from torchvision.utils import save_image
from torch.autograd import grad
import torch
import time
import random
import os, logging
import numpy as np
from brep_mi_utils import *
import shutil


# Sample "#points_count" points around a sphere centered on "current_point" with radius =  "sphere_radius"
def gen_points_on_sphere(current_point, points_count, sphere_radius):
    
    # get random perturbations
    points_shape = (points_count,) + current_point.shape
    perturbation_direction = torch.randn(*points_shape).cuda()
    dims = tuple([i for i in range(1, len(points_shape))])
    
    # normalize them such that they are uniformly distributed on a sphere with the given radius
    perturbation_direction = (sphere_radius/ torch.sqrt(torch.sum(perturbation_direction ** 2, axis = dims, keepdims = True))) * perturbation_direction
    
    # add the perturbations to the current point
    sphere_points = current_point + perturbation_direction
    return sphere_points, perturbation_direction




def attack_single_target(current_point, target_class, current_loss, G,
                    target_model, evaluator_model, attack_params, criterion, current_iden_dir ):
    current_iter = 0
    last_iter_when_radius_changed = 0
    
    # create log file
    log_file = open(os.path.join(current_iden_dir,'train_log'),'w')
    losses= []
    target_class_tensor = torch.tensor([target_class]).cuda()
    current_sphere_radius = attack_params['current_sphere_radius']
    
    last_success_on_eval = False
    # Outer loop handle all sphere radii
    while  current_iter - last_iter_when_radius_changed < attack_params['max_iters_at_radius_before_terminate']:
        
        # inner loop handle one single sphere radius
        while current_iter - last_iter_when_radius_changed < attack_params['max_iters_at_radius_before_terminate']:
            
            
            new_radius = False
            
            #step size is similar to learning rate
            # we limit max step size to 3. But feel free to change it
            step_size = min(current_sphere_radius / 3,3)
            
            # sample points on the sphere
            new_points, perturbation_directions = gen_points_on_sphere(current_point,attack_params['sphere_points_count'], current_sphere_radius)
            
            # get the predicted labels of the target model on the sphere points
            new_points_classification = is_target_class(G(new_points),target_class,target_model)
            
            
            # handle case where all(or some percentage) sphere points lie in decision boundary. We increment sphere size
            if new_points_classification.sum() > 0.75 * attack_params['sphere_points_count'] :# == attack_params['sphere_points_count']:
                save_tensor_images(G(current_point.unsqueeze(0))[0].detach(),
                                   os.path.join(current_iden_dir,                                 "last_img_of_radius_{:.4f}_iter_{}.png".format(current_sphere_radius, current_iter)))
                # update the current sphere radius
                current_sphere_radius = current_sphere_radius * attack_params['sphere_expansion_coeff']
                
                log_file.write("new sphere radius at iter: {} ".format(current_iter))
                new_radius = True
                last_iter_when_radius_changed = current_iter
            
            
            # get the update direction, which is the mean of all points outside boundary if 'repulsion_only' is used. Otherwise it is the mean of all points * their classification (1,-1)
            if attack_params['repulsion_only'] == True:
                new_points_classification = (new_points_classification - 1)/2
                
            grad_direction = torch.mean(new_points_classification.unsqueeze(1) * perturbation_directions, axis = 0) / current_sphere_radius

            # move the current point with stepsize towards grad_direction
            current_point_new = current_point + step_size * grad_direction
            current_point_new = current_point_new.clamp(min=attack_params['point_clamp_min'], max=attack_params['point_clamp_max'])
            
            current_img = G(current_point_new.unsqueeze(0))
            if is_target_class(current_img,target_class,target_model)[0] == -1:
                log_file.write("current point is outside target class boundary")
                break

            current_point = current_point_new
            _,current_loss = decision(current_img,target_model,score=True, criterion = criterion, target=target_class_tensor)

            if current_iter % 50 == 0 or (current_iter < 200 and current_iter % 20 == 0  ) :
                save_tensor_images(current_img[0].detach(), os.path.join(current_iden_dir, "iter{}.png".format(current_iter)))
            
            eval_decision = decision_Evaluator(current_img ,evaluator_model)
            correct_on_eval = True if  eval_decision==target_class else False
            if new_radius:
                point_before_inc_radius = current_point.clone()
                last_success_on_eval = correct_on_eval
                break
            iter_str = "iter: {}, current_sphere_radius: {}, step_size: {:.2f} sum decisions: {}, loss: {:.4f}, eval predicted class {}, classified correct on Eval {}".format(
                current_iter,current_sphere_radius, step_size,
                new_points_classification.sum(),
                current_loss.item(),
                eval_decision,
                correct_on_eval)
            
            log_file.write(iter_str+'\n')
            losses.append(current_loss.item())
            current_iter +=1
            

    log_file.close()
    #acc = 1 if decision_Evaluator(G(current_point.unsqueeze(0)),evaluator_model)==target_class  else 0
    acc = 1 if last_success_on_eval is True  else 0
    return acc




def attack(attack_params,
                    target_model,
                    evaluator_model,
                    generator_model,
                    attack_imgs_dir,
                    private_domain_imgs_path):
    
    # attack the same targets using same initial points as saved experiment
    if 'targets_from_exp' in attack_params:
        print("loading intial points from experiment dir: {}".format(attack_params['targets_from_exp']))
        points = gen_initial_points_from_exp(attack_params['targets_from_exp'])
        
    #attack same targets as experiment, but generate new random initial points    
    elif 'gen_idens_as_exp' in attack_params:
        print("attacking same targets as experiment dir: {}".format(attack_params['gen_idens_as_exp']))
        points = gen_idens_as_exp( attack_params['gen_idens_as_exp'],                                           
                                   attack_params['batch_dim_for_initial_points'],
                                   generator_model,
                                   target_model,
                                   attack_params['point_clamp_min'],
                                   attack_params['point_clamp_max'],
                                   attack_params['z_dim'])
    #attack target classes from iden_range_min to iden_range_max
    elif attack_params['targeted_attack']:
        print("attacking the targets from: {} to {}".format(attack_params['iden_range_min'], attack_params['iden_range_max']))
        points = gen_initial_points_targeted(   attack_params['batch_dim_for_initial_points'],
                                                generator_model,
                                                target_model,
                                                attack_params['point_clamp_min'],
                                                attack_params['point_clamp_max'],
                                                attack_params['z_dim'],
                                                attack_params['iden_range_min'],
                                                attack_params['iden_range_max'])
    #attack any N labels
    else:
        print("attacking any {} targets".format(attack_params['num_targets']))
        points = gen_initial_points_untargeted(attack_params['num_targets'],
                                           attack_params['batch_dim_for_initial_points'],
                                           generator_model,
                                           target_model,
                                           attack_params['point_clamp_min'],
                                           attack_params['point_clamp_max'],
                                           attack_params['z_dim'])
    
    #points.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    correct_on_eval = 0
    current_iter = 0
    for target_class in points:
        current_iter += 1
        current_point = points[target_class].cuda()
        print(" {}/{}: attacking iden {}".format(current_iter, len(points), target_class))
        target_class_tensor = torch.tensor([target_class]).cuda()

        # save the first generated image, and current point (z) to the iden_dir
        current_iden_dir = os.path.join(attack_imgs_dir,"iden_{}".format(target_class))
        os.makedirs(current_iden_dir, exist_ok=True)
        first_img = generator_model(current_point.unsqueeze(0))
        save_tensor_images(first_img[0].detach(), os.path.join(current_iden_dir, "original_first_point.png".format(current_iter)))
        np.save(os.path.join(current_iden_dir, 'initial_z_point'),
                current_point.cpu().detach().numpy())
        
        # copy the groundtruth images of the target to the attack dir
        # please put all groundtruth images in one single image called all.png
        # the path to the groundtruth image of label should be  "$dataset_dir/label/all.png"
        if len(private_domain_imgs_path) > 0:
            shutil.copy(os.path.join(private_domain_imgs_path,str(target_class),'all.png'),
                        os.path.join(current_iden_dir, 'groundtruth_imgs.png'))
        
        # first image should always be inside target class
        assert is_target_class(first_img,target_class,target_model).item() == 1
        
        
        _, initial_loss = decision(generator_model(current_point.unsqueeze(0)),target_model,score=True, criterion= criterion, target=target_class_tensor)
        
        
        correct_on_eval += attack_single_target(current_point, target_class, initial_loss, generator_model, target_model, evaluator_model, attack_params, criterion, current_iden_dir )
        current_acc_on_eval = correct_on_eval / current_iter
        print("current acc on eval model: {:.2f}%".format(current_acc_on_eval*100))
        
    total_acc_on_eval = correct_on_eval / len(points)
    print("total acc on eval model: {:.2f}%".format(total_acc_on_eval*100))
    
    
    