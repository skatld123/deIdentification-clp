import os
import cv2
import numpy as np

def deIdentify(result_dir, data_path, crop_path, save_path) :
    replacement = cv2.imread('/root/deIdentification-clp/virtual-plate/test.jpg')  # with alpha channel
    for img_name, value in result_dir.items() :
        # 배경 및 배경 마스크 생성
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
            cv2.imwrite(os.path.join(save_path, f"output_ori_{idx}.jpg"), output)
            
            # 블러 적용
            output = cv2.GaussianBlur(output, (5, 5), 0)
            
            output = cv2.cvtColor(output, cv2.COLOR_BGR2HSV)
            # 명도 채도 색상 조정
            output[:, :, 0] = h
            output[:, :, 1] = s
            
            output = cv2.cvtColor(output, cv2.COLOR_HSV2BGR)
            cv2.imwrite(os.path.join(save_path, f"output_trans_{idx}.jpg"), output)
            
            cv2.fillConvexPoly(sub_img, dst_pts.astype(np.int32), (0, 0, 0))
            # cv2.imwrite(os.path.join(save_path, f"bg_{idx}.jpg"), background)
            # 변환된 이미지를 원본 이미지에 적용
            result = cv2.add(sub_img, output)
            cv2.imwrite(os.path.join(save_path, f"output_{idx}.jpg"), result)
            
            background[box[1]:box[3], box[0]:box[2]] = result
        # 결과 이미지 저장
        cv2.imwrite(os.path.join(save_path, img_name + ".jpg"), background)
