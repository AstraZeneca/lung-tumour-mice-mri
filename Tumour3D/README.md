# Tumour Segmentation using 3D MRI
Training the models for the task of segmenting only lung tumour using 3D MRI, exploring the challenge of segmenting sole tumour and importance of spatial background.

*Note that training the 3D models is very computationally demanding; while we managed to train all these models under one GPU (while drastically decreasing batch size and increasing nworkers) and you might need to adjust the parameters according to your system)* 

You can execute the following from root directory to reproduce the experiment:
### Setup
~~~
python -m src.setup
~~~
### Data preparation
~~~
python -m src.prepare_data.py --exp exp3
~~~
### Experiments
~~~
python -m src.run_train --exp exp3
python -m src.run_eval --exp exp3
~~~
This is the same as:
~~~
# 2D
python -m src.deepmeta.train --model unet --loss weighted_ce -classes 2 --save_dir Tumour3D/results --data_path Tumour3D/data --dim 2d --epochs 100 -bs 64 -nworkers 64
python -m src.deepmeta.train --model unet3p --loss unet3p_loss -classes 2 --save_dir Tumour3D/results --data_path Tumour3D/data --dim 2d --epochs 100 -bs 64 -nworkers 64
python -m src.deepmeta.train --model deepmeta --loss fusion_loss -classes 2 --save_dir Tumour3D/results --data_path Tumour3D/data --dim 2d --epochs 100 -bs 64 -nworkers 64

python -m src.deepmeta.predict --save --model unet -classes 2 --model_path/Tumour3D/results/models/best_unet_model_2d --data_dir Tumour3D/data/test --save_dir Tumour3D/results/outputs/unet_2d --dim 2d
python -m src.deepmeta.predict --save --model unet3p -classes 2 --model_path/Tumour3D/results/models/best_unet3p_model_2d --data_dir Tumour3D/data/test --save_dir Tumour3D/results/outputs/unet3p_2d --dim 2d
python -m src.deepmeta.predict --save --model deepmeta -classes 2 --model_path/Tumour3D/results/models/best_deepmeta_model_2d --data_dir Tumour3D/data/test --save_dir Tumour3D/results/outputs/deepmeta_2d --dim 2d

#Note that you might need to adjust the pathway to your own home dir
export nnUNet_raw='Tumour3D/nnUNet_raw'
export nnUNet_preprocessed='Tumour3D/nnUNet_preprocessed'
export nnUNet_results='Tumour3D/nnUNet_results'

nnUNetv2_plan_and_preprocess -d 1

nnUNetv2_train -d 1 -c 2d 0
nnUNetv2_train -d 1 -c 2d 1
nnUNetv2_train -d 1 -c 2d 2 
nnUNetv2_train -d 1 -c 2d 3 
nnUNetv2_train -d 1 -c 2d 4

nnUNetv2_predict -d 1 -c 2d -i Tumour3D/data/nnunet_test/images -o Tumour3D/results/outputs/nnunet_2d 

python -m src.nnunet.eval -classes 2 --gt_path Tumour3D/data/nnunet_test/labels --pt_path Tumour3D/results/outputs/nnunet_2d --save_path Tumour3D/results/outputs/nnunet_2d

# 3D
python -m src.deepmeta.train --model unet --loss weighted_ce -classes 2 --save_dir Tumour3D/results --data_path Tumour3D/data --dim 3d --epochs 100 -bs 2 -nworkers 64
python -m src.deepmeta.train --model unet3p --loss unet3p_loss -classes 2 --save_dir Tumour3D/results --data_path Tumour3D/data --dim 3d --epochs 100 -bs 2 -nworkers 64
python -m src.deepmeta.train --model deepmeta --loss fusion_loss -classes 2 --save_dir Tumour3D/results --data_path Tumour3D/data --dim 3d --epochs 100 -bs 2 -nworkers 64

nnUNetv2_train -d 1 -c 3d_fullres 0
nnUNetv2_train -d 1 -c 3d_fullres 1
nnUNetv2_train -d 1 -c 3d_fullres 2 
nnUNetv2_train -d 1 -c 3d_fullres 3 
nnUNetv2_train -d 1 -c 3d_fullres 4 

nnUNetv2_predict -d 1 -c 3d_fullres -i Tumour3D/data/nnunet_test/images -o Tumour3D/results/outputs/nnunet_3d

python -m src.nnunet.eval -classes 2 --gt_path Tumour3D/data/nnunet_test/labels --pt_path Tumour3D/results/outputs/nnunet_3d --save_path Tumour3D/results/outputs/nnunet_3d'

~~~
