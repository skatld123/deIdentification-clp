import os
import cv2
import numpy as np
from tqdm import tqdm
from transform_gan.test import transform

def deIdentify_blur_or_mask(result_dir, data_path, crop_path, save_path, metric=1) :
    '''
    metric = 1:Blur, 2:Mask, 3:Virtual License Plate
    '''
    print("Start making VLP and perspective transform...")
    not_apply = []
    replacement = cv2.imread('/root/deIdentification-clp/virtual-plate/test.jpg')  # with alpha channel
    os.makedirs(save_path, exist_ok=True)
    for img_name, value in tqdm(result_dir.items()) :
        # 배경 및 배경 마스크 생성
        if "key_points" in value :
            background = cv2.imread(os.path.join(data_path, img_name + ".jpg"))
            bh, bw, _ = background.shape
            background_mask = np.zeros_like(background)
            for idx, (sub, box, kp) in enumerate(zip(value["subimage"], value["boxes"], value["key_points"])) :
                sub_img = cv2.imread(os.path.join(crop_path, sub))
                sh, sw, _ = sub_img.shape
                
                h, w, _ = replacement.shape
                src_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
                dst_pts = np.array(kp, dtype=np.float32)

                # 명도 채도 색상 추출
                hsv_image = cv2.cvtColor(sub_img, cv2.COLOR_BGR2HSV)
                # roi = hsv_image[dst_pts]
                h, s, v = cv2.split(hsv_image)
                
                # 크롭된 이미지의 마스크 생성
                transform_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
                output = cv2.warpPerspective(replacement, transform_matrix, (
                    sw, sh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
                # cv2.imwrite(os.path.join(save_path, f"output_ori_{idx}.jpg"), output)
                
                # 블러 적용
                if metric == 1 :
                    blur_kernel_size = (31, 31)  # 블러 커널 크기
                    blur_sigma = 80              # 블러 시그마 값
                    # 가우시안 블러 적용
                    output = cv2.GaussianBlur(output, blur_kernel_size, blur_sigma)
                    # 마스크를 사용하여 원본 이미지와 블러 이미지 병합
                    method = "blur"
                elif metric ==2 : 
                    output = np.zeros((sh, sw, 3), np.uint8)
                    method = "mask"
                else : 
                    output = cv2.GaussianBlur(output, (5, 5), 0)
                
                    output = cv2.cvtColor(output, cv2.COLOR_BGR2HSV)
                    # 명도 채도 색상 조정
                    output[:, :, 0] = h
                    output[:, :, 1] = s
                    output = cv2.cvtColor(output, cv2.COLOR_HSV2BGR)
                    method = "vlp"
                cv2.fillConvexPoly(sub_img, dst_pts.astype(np.int32), (0, 0, 0))
                # 변환된 이미지를 원본 이미지에 적용
                result = cv2.add(sub_img, output)
                vlp_path = os.path.join(save_path, method)
                os.makedirs(vlp_path, exist_ok=True)
                cv2.imwrite(os.path.join(vlp_path, sub), result)
                
                background[box[1]:box[3], box[0]:box[2]] = result
            # 결과 이미지 저장
            cv2.imwrite(os.path.join(save_path, img_name + ".jpg"), background)
        else :
            cv2.imwrite(os.path.join(save_path, img_name + ".jpg"), background)
            not_apply.append(img_name + ".jpg")
    print(f"Images without VLP applied : {not_apply}")
    print("Finish making VLP and perspective transform!")
    return vlp_path

def deIdentify(result_dir, vlp_path, entireImg_path, tflp_path, output_path, checkpoints_dir) :
    print("Start De-Identification...")
    not_apply = []
    if not os.path.exists(output_path) : os.mkdir(output_path)
    
    # CycleGAN 비식별화
    if not os.path.exists(tflp_path) : os.mkdir(tflp_path)
    transform(dataroot=vlp_path, gpu_ids=[0,1], num_test=len(os.listdir(vlp_path)),
              results_dir=tflp_path, checkpoints_dir=checkpoints_dir)
    
    for img_name, value in tqdm(result_dir.items()) :
        # 배경 및 배경 마스크 생성
        if "boxes" in value :
            background = cv2.imread(os.path.join(entireImg_path, img_name + ".jpg"))
            for idx, (sub, box) in enumerate(zip(value["subimage"], value["boxes"])) :
                # 변환된 이미지를 불러옴
                tf_lp_img = cv2.imread(os.path.join(tflp_path, sub))
                
                w = box[3] - box[1] 
                h = box[2] - box[0] 
                
                rs_tf_lp_img = cv2.resize(tf_lp_img, dsize=(h, w), interpolation=cv2.INTER_LINEAR)
                
                background[box[1]:box[3], box[0]:box[2]] = rs_tf_lp_img
            # 결과 이미지 저장
            print("saved at " + os.path.join(output_path, img_name + ".jpg"))
            cv2.imwrite(os.path.join(output_path, img_name + ".jpg"), background)
        else :
            cv2.imwrite(os.path.join(output_path, img_name + ".jpg"), background)
            not_apply.append(img_name + ".jpg")
    print(f"Images without de-identification applied : {not_apply}")
    print("Finish De-Identification!")
