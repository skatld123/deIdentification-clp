set -ex
# python train.py --dataroot ./datasets/license-plate --name license-plate_cyclegan --gpu_ids 0,1 --batch_size 16 --model cycle_gan --pool_size 50 --no_dropoutG
# GAN의 모드 설정 --gan_mode wgangp 
# --epoch 200 --save_epoch_freq 5 -> 10
# --use_wandb --wandb_project_name de-id-gan
# python train.py --dataroot ./datasets/license-plate_v2 --name license-plate_cyclegan_v2 --gpu_ids 0,1 --batch_size 16 --model cycle_gan --pool_size 50 --epoch 200 --save_epoch_freq 10 --use_wandb --wandb_project_name de-id-gan
python train.py --dataroot ./datasets/license-plate_v2 --name license-plate_cyclegan_vanilla_v2 --gpu_ids 0,1 --batch_size 16 --model cycle_gan --gan_mode wgangp --pool_size 50 --epoch 200 --save_epoch_freq 20 --use_wandb --wandb_project_name de-id-gan-vanilla