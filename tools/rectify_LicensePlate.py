import os
import cv2
import numpy as np
from tqdm import tqdm

def deIdentify(result_dir, data_path, crop_path, save_path) :
    print("Start De-Identification...")
    not_apply = []
    replacement = cv2.imread('/root/deIdentification-clp/virtual-plate/test.jpg')  # with alpha channel
    if not os.path.exists(save_path) : os.mkdir(save_path)
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
                output = cv2.GaussianBlur(output, (5, 5), 0)
                
                output = cv2.cvtColor(output, cv2.COLOR_BGR2HSV)
                # 명도 채도 색상 조정
                output[:, :, 0] = h
                output[:, :, 1] = s
                
                output = cv2.cvtColor(output, cv2.COLOR_HSV2BGR)
                # cv2.imwrite(os.path.join(save_path, f"output_trans_{idx}.jpg"), output)
                
                cv2.fillConvexPoly(sub_img, dst_pts.astype(np.int32), (0, 0, 0))
                # 변환된 이미지를 원본 이미지에 적용
                result = cv2.add(sub_img, output)
                # cv2.imwrite(os.path.join(save_path, f"output_{idx}.jpg"), result)
                
                background[box[1]:box[3], box[0]:box[2]] = result
            # 결과 이미지 저장
            cv2.imwrite(os.path.join(save_path, img_name + ".jpg"), background)
        else :
            cv2.imwrite(os.path.join(save_path, img_name + ".jpg"), background)
            not_apply.append(img_name + ".jpg")
    print(f"Images without de-identification applied : {not_apply}")
    print("Finish De-Identification!")

def rectifyPlate(result_dic, data_path, crop_path, save_path) :
    print("Start Rectifing...")
    not_apply = []
    replacement = cv2.imread('/root/deIdentification-clp/virtual-plate/test.jpg')  # with alpha channel
    rh, rw, _ = replacement.shape
    if not os.path.exists(save_path) : os.mkdir(save_path)
    for img_name, value in tqdm(result_dic.items()) :
        # 배경 및 배경 마스크 생성
        if "key_points" in value :
            background = cv2.imread(os.path.join(data_path, img_name + ".jpg"))
            bh, bw, _ = background.shape
            background_mask = np.zeros_like(background)
            for idx, (sub, box, kp) in enumerate(zip(value["subimage"], value["boxes"], value["key_points"])) :
                sub_img = cv2.imread(os.path.join(crop_path, sub))
                sh, sw, _ = sub_img.shape
                
                h, w, _ = replacement.shape
                src_pts = np.array(kp, dtype=np.float32)
                dst_pts = np.array([[0, 0], [rw, 0], [rw, rh], [0, rh]], dtype=np.float32)
                
                # 크롭된 이미지의 마스크 생성
                transform_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
                
                rectified_img = cv2.warpPerspective(sub_img, transform_matrix, (rw, rh))
                
                # 결과 이미지 저장
                cv2.imwrite(os.path.join(save_path, f'{img_name}_{idx:03d}.jpg'), rectified_img)
        else : 
            not_apply.append(img_name + ".jpg")
    print(f"Noting found from Landmark Detection : {not_apply}")
    print("Finish Rectify!")