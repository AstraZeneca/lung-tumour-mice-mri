# Lung+Tumour Segmentation using 2D MRI
Training the models for the task of segmenting both lung and tumour, exploring the importance of organ background. 

To reproduce the experiment, you can run the following from the root directory:
### Setup
~~~
python -m src.setup
~~~
### Data preparation
~~~
python -m src.prepare_data --exp exp1
~~~
### Reproduce experiments
~~~
python -m src.run_train --exp exp1
python -m src.run_eval --exp exp1
~~~

This is the same as:
~~~
python -m src.deepmeta.train --model unet --loss weighted_ce -classes 3 --save_dir Lung+Tumour/results --data_path Lung+Tumour/data --dim 2d --epochs 100 -bs 64 -nworkers 20
python -m src.deepmeta.train --model unet3p --loss unet3p_loss -classes 3 --save_dir Lung+Tumour/results --data_path Lung+Tumour/data --dim 2d --epochs 100 -bs 64 -nworkers 20
python -m src.deepmeta.train --model deepmeta --loss fusion_loss -classes 3 --save_dir Lung+Tumour/results --data_path Lung+Tumour/data --dim 2d --epochs 100 -bs 64 -nworkers 20

python -m src.deepmeta.predict --save --model unet -classes 3 --model_path/Lung+Tumour/results/models/best_unet_model_2d --data_dir Lung+Tumour/data/test --save_dir Lung+Tumour/results/outputs/unet_2d --dim 2d
python -m src.deepmeta.predict --save --model unet3p -classes 3 --model_path/Lung+Tumour/results/models/best_unet3p_model_2d --data_dir Lung+Tumour/data/test --save_dir Lung+Tumour/results/outputs/unet3p_2d --dim 2d
python -m src.deepmeta.predict --save --model deepmeta -classes 3 --model_path/Lung+Tumour/results/models/best_deepmeta_model_2d --data_dir Lung+Tumour/data/test --save_dir Lung+Tumour/results/outputs/deepmeta_2d --dim 2d

#Note that you might need to adjust the pathway to your own home dir
export nnUNet_raw='Lung+Tumour/nnUNet_raw'
export nnUNet_preprocessed='Lung+Tumour/nnUNet_preprocessed'
export nnUNet_results='Lung+Tumour/nnUNet_results'

nnUNetv2_plan_and_preprocess -d 1
nnUNetv2_train -d 1 -c 2d 0
nnUNetv2_train -d 1 -c 2d 1
nnUNetv2_train -d 1 -c 2d 2 
nnUNetv2_train -d 1 -c 2d 3 
nnUNetv2_train -d 1 -c 2d 4 

nnUNetv2_predict -d 1 -c 2d -i Lung+Tumour/data/nnunet_test/images -o Lung+Tumour/results/outputs/nnunet_2d 
python -m src.nnunet.eval -classes 3 --gt_path Lung+Tumour/data/nnunet_test/labels --pt_path Lung+Tumour/results/outputs/nnunet_2d
~~~
