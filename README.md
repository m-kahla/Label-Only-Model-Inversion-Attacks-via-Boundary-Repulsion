# Implementation of paper "Label-Only-Model-Inversion-Attacks-via-Boundary-Repulsion" CVPR2022
link: https://openaccess.thecvf.com/content/CVPR2022/html/Kahla_Label-Only_Model_Inversion_Attacks_via_Boundary_Repulsion_CVPR_2022_paper.html


In this paper, we are the first in literature to apply model inversion attack to steal private datasets just by querying a model trained on them and getting the label.

We use the same models to attack as our previous baselines. You can download them at link : https://drive.google.com/drive/folders/1U4gekn72UX_n1pHdm9GQUQwwYVDvpTfN

You can also download the generator model from link: https://drive.google.com/drive/folders/1L3frX-CE4j36pe5vVWuy3SgKGS9kkA70?usp=sharing.

to run the attack, simply use main.py with the desired arguments:


     usage: main.py [-h] [--target_model TARGET_MODEL]
                     [--target_model_path TARGET_MODEL_PATH]
                     [--evaluator_model EVALUATOR_MODEL]
                     [--evaluator_model_path EVALUATOR_MODEL_PATH]
                     [--generator_model_path GENERATOR_MODEL_PATH] [--device DEVICE]
                     --experiment_name EXPERIMENT_NAME --config_file CONFIG_FILE
                     [--private_imgs_path PRIVATE_IMGS_PATH] [--n_classes N_CLASSES]
                     [--n_classes_evaluator N_CLASSES_EVALUATOR]

    A tool that applies Label Only Model Inversion Attack using labels only.

    optional arguments:
      -h, --help            show this help message and exit
      --target_model TARGET_MODEL
                            VGG16 | IR152 | FaceNet64
      --target_model_path TARGET_MODEL_PATH
                            path to target_model
      --evaluator_model EVALUATOR_MODEL
                            VGG16 | IR152 | FaceNet64| FaceNet
      --evaluator_model_path EVALUATOR_MODEL_PATH
                            path to evaluator_model
      --generator_model_path GENERATOR_MODEL_PATH
                            path to generator model
      --device DEVICE       Device to use. Like cuda, cuda:0 or cpu
      --experiment_name EXPERIMENT_NAME
                            experiment name for experiment directory
      --config_file CONFIG_FILE
                            config file that has attack params
      --private_imgs_path PRIVATE_IMGS_PATH
                            Path to groundtruth images to copy them to attack dir.
                            Empty string means, our tool will not copy.
      --n_classes N_CLASSES
                            num of classes of target model
      --n_classes_evaluator N_CLASSES_EVALUATOR
                            num of classes of evaluator model


example to run an attack on the models given above:

    python3 main.py --target_model=FaceNet64 --target_model_path=models/FaceNet64_88.50.tar  --device=0 --experiment_name=celebA_facenet_config2  --config_file=/config2.yaml  --private_imgs_path='' --n_classes=1000 --n_classes_evaluator=1000 --evaluator_model=FaceNet --evaluator_model_path=models/target_ckp/FaceNet_95.88.tar --generator_model_path=models/celeba_G.tar

For any questions, please email kahla@vt.edu
