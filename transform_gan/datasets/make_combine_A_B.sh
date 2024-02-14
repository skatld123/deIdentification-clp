mkdir /root/deid-lp-GAN/datasets/license-plate_v2/trainA/train
mkdir /root/deid-lp-GAN/datasets/license-plate_v2/trainB/train
mkdir /root/deid-lp-GAN/datasets/license-plate_v2/valA/val
mkdir /root/deid-lp-GAN/datasets/license-plate_v2/valB/val
mkdir /root/deid-lp-GAN/datasets/license-plate_v2/testA/test
mkdir /root/deid-lp-GAN/datasets/license-plate_v2/testB/test

mv /root/deid-lp-GAN/datasets/license-plate_v2/trainA/*.jpg /root/deid-lp-GAN/datasets/license-plate_v2/trainA/train/
mv /root/deid-lp-GAN/datasets/license-plate_v2/trainB/*.jpg /root/deid-lp-GAN/datasets/license-plate_v2/trainB/train/
mv /root/deid-lp-GAN/datasets/license-plate_v2/valA/*.jpg /root/deid-lp-GAN/datasets/license-plate_v2/valA/val/
mv /root/deid-lp-GAN/datasets/license-plate_v2/valB/*.jpg /root/deid-lp-GAN/datasets/license-plate_v2/valB/val/
mv /root/deid-lp-GAN/datasets/license-plate_v2/testA/*.jpg /root/deid-lp-GAN/datasets/license-plate_v2/testA/test/
mv /root/deid-lp-GAN/datasets/license-plate_v2/testB/*.jpg /root/deid-lp-GAN/datasets/license-plate_v2/testB/test/

python combine_A_and_B.py --fold_A /root/deid-lp-GAN/datasets/license-plate_v2/trainA --fold_B /root/deid-lp-GAN/datasets/license-plate_v2/trainB --fold_AB /root/deid-lp-GAN/datasets/license-plate_v2/train
python combine_A_and_B.py --fold_A /root/deid-lp-GAN/datasets/license-plate_v2/valA --fold_B /root/deid-lp-GAN/datasets/license-plate_v2/valB --fold_AB /root/deid-lp-GAN/datasets/license-plate_v2/val
python combine_A_and_B.py --fold_A /root/deid-lp-GAN/datasets/license-plate_v2/testA --fold_B /root/deid-lp-GAN/datasets/license-plate_v2/testB --fold_AB /root/deid-lp-GAN/datasets/license-plate_v2/test

mv license-plate_v2/trainA/train/* license-plate_v2/trainA/
mv license-plate_v2/trainB/train/* license-plate_v2/trainB/

mv license-plate_v2/valA/val/* license-plate_v2/valA/
mv license-plate_v2/valB/val/* license-plate_v2/valB/

mv license-plate_v2/testA/test/* license-plate_v2/testA/
mv license-plate_v2/testB/test/* license-plate_v2/testB/

rm -rf license-plate_v2/trainA/train
rm -rf license-plate_v2/trainB/train

rm -rf license-plate_v2/valA/val
rm -rf license-plate_v2/valB/val

rm -rf license-plate_v2/testA/test
rm -rf license-plate_v2/testB/test

# 부록
mv license-plate_v2/train/train/* license-plate_v2/train/
rm -rf license-plate_v2/train/train

mv license-plate_v2/val/val/* license-plate_v2/val/
rm -rf license-plate_v2/val/val

mv license-plate_v2/test/test/* license-plate_v2/test/
rm -rf license-plate_v2/test/test