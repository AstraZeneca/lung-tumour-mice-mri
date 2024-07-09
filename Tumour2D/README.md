# Tumour Segmentation using 2D MRI
Training the models for the task of segmenting only lung Tumour, exploring the challenge of segmenting sole tumor. 

To reproduce the experiment, you can run the following from the root directory:

### Setup
~~~
python -m src.setup
~~~
### Data preparation
~~~
python -m src.prepare_data --exp exp2
~~~
### Reproduce experiments
~~~
python -m src.run_train --exp exp2
python -m src.run_eval --exp exp2
~~~

This is the same as:
~~~
python -m src.deepmeta.train --model unet --loss weighted_ce -classes 2 --save_dir Tumour2D/results --data_path Tumour2D/data --dim 2d --epochs 100 -bs 64 -nworkers 20
python -m src.deepmeta.train --model unet3p --loss unet3p_loss -classes 2 --save_dir Tumour2D/results --data_path Tumour2D/data --dim 2d --epochs 100 -bs 64 -nworkers 20
python -m src.deepmeta.train --model deepmeta --loss fusion_loss -classes 2 --save_dir Tumour2D/results --data_path Tumour2D/data --dim 2d --epochs 100 -bs 64 -nworkers 20

python -m src.deepmeta.predict --save --model unet -classes 2 --model_path/Tumour2D/results/models/best_unet_model_2d --data_dir Tumour2D/data/test --save_dir Tumour2D/results/outputs/unet_2d --dim 2d
python -m src.deepmeta.predict --save --model unet3p -classes 2 --model_path/Tumour2D/results/models/best_unet3p_model_2d --data_dir Tumour2D/data/test --save_dir Tumour2D/results/outputs/unet3p_2d --dim 2d
python -m src.deepmeta.predict --save --model deepmeta -classes 2 --model_path/Tumour2D/results/models/best_deepmeta_model_2d --data_dir Tumour2D/data/test --save_dir Tumour2D/results/outputs/deepmeta_2d --dim 2d

#Note that you might need to adjust the pathway to your own home dir
export nnUNet_raw='Tumour2D/nnUNet_raw'
export nnUNet_preprocessed='Tumour2D/nnUNet_preprocessed'
export nnUNet_results='Tumour2D/nnUNet_results'

nnUNetv2_plan_and_preprocess -d 1
nnUNetv2_train -d 1 -c 2d 0
nnUNetv2_train -d 1 -c 2d 1
nnUNetv2_train -d 1 -c 2d 2 
nnUNetv2_train -d 1 -c 2d 3 
nnUNetv2_train -d 1 -c 2d 4 

nnUNetv2_predict -d 1 -c 2d -i Tumour2D/data/nnunet_test/images -o Tumour2D/results/outputs/nnunet_2d 
python -m src.nnunet.eval -classes 2 --gt_path Tumour2D/data/nnunet_test/labels --pt_path Tumour2D/results/outputs/nnunet_2d
~~~
