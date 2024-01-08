set -ex
python test.py --model cycle_gan --dataroot ./datasets/license-plate_v2/ --name license-plate_cyclegan_v2_no_aug \
--gpu_ids 0 --phase test --no_dropout --direction AtoB \
--results_dir /root/deid-lp-GAN/results/

python test.py --model cycle_gan --dataroot ./datasets/license-plate_v2/ --name license-plate_cyclegan_vanilla_v2 \
--gpu_ids 0,1 --phase test --no_dropout --direction AtoB \
--results_dir /root/deid-lp-GAN/results/ --num_test 917