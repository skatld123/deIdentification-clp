
import ast
import os
from PIL import Image
import cv2
from tqdm import tqdm


def convert_row_from_csv(row):
    return [row[0], float(row[1]), float(row[2]), ast.literal_eval(row[3])]

def cropping_image_from_array(input_dir, output_dir, detections, cls, save_img) :
    dic_bbox_with_point = {}
    print("Cropping only LicensePlate Bounding Box from Input Image")
    index = 0
    if os.listdir(input_dir) == 0 : 
        print(f"ERROR : Empty Cropped directory : {input_dir}")
        return
    if not os.path.exists(output_dir) : os.mkdir(output_dir)
    for idx, detection in enumerate(tqdm(detections)):
        # 박스 변환
        if isinstance(detection[3], tuple) : box = list(detection[3])
        else : box = ast.literal_eval(detection[3])

        file_name, label, conf = detection[0], float(detection[1]), float(detection[2])
        if label == cls:
            img_path = os.path.join(input_dir, file_name + ".jpg")
            img = Image.open(img_path)
            cropped_img = img.crop(box)
            if save_img :
                cropped_img.save(os.path.join(output_dir, f'{file_name}_crop_{index:04d}.jpg'))
            if file_name in dic_bbox_with_point:
                dic_bbox_with_point[file_name]['subimage'].append(
                    f'{file_name}_crop_{index:04d}.jpg')
                dic_bbox_with_point[file_name]['boxes'].append(box)
            else:
                dic_bbox_with_point[file_name] = {'subimage': [
                    f'{file_name}_crop_{index:04d}.jpg'], 'boxes': [box]}
            index += 1
        else:
            continue
    print(
        f"Cropped result images saved at : {output_dir} \nCropped result images cnt : {len(os.listdir(output_dir))}")
    return dic_bbox_with_point

# def point_local2global(bbox_with_point, detection_result) :
#     # detection result xyxy임
#     global_dic = {}
#     for key in bbox_with_point :
#         img = cv2.imread(os.path.join(cf.cfg_crop['input'], key + ".jpg"))
#         mh, mw, _ = img.shape
#         subImgList = bbox_with_point[key]['subimage']
#         subImgboxes = bbox_with_point[key]['boxes']
#         for subImgName, subImgbox in zip(subImgList, subImgboxes) : 
#             sx1, sy1, sx2, sy2 = subImgbox
#             sw = sx2 - sx1
#             sh = sy2 - sy1
#             if os.path.splitext(subImgName)[0] in detection_result :
#                 cropImg = detection_result[os.path.splitext(subImgName)[0]]
#             else :
#                 print(f"no detection in {os.path.splitext(subImgName)[0]}")
#                 continue
#             boxes, scores, labels = cropImg['boxes'], cropImg['scores'], cropImg['labels']
#             for idx, (box, score, label) in enumerate(zip(boxes, scores, labels)) :
#                 rx1, ry1, rx2, ry2 = box
#                 x1, y1, x2, y2 = convert_scaled_predict((sw, sh), (rx1, ry1, rx2, ry2))
#                 mx1, my1, mx2, my2 = (x1 + sx1) / mw, (y1 + sy1) / mh, (x2 + sx1) / mw, (y2 + sy1) / mh
#                 if key in global_dic :
#                     global_dic[key]['boxes'].append([mx1, my1, mx2, my2])
#                     global_dic[key]['scores'].append(score)
#                     global_dic[key]['labels'].append(label)
#                 else : 
#                     global_dic[key] = {'boxes' : [[mx1, my1, mx2, my2]], 'scores' : [score], 'labels' : [label]}
#                 # cv2.rectangle(img, (mx1, my1), (mx2, my2), (255,0,0), 4)
#         # cv2.imwrite(f"/root/deIdentification-clp/result/{key}.jpg", img)
#     return global_dic

# TODO JSON으로 원하는 클래스 자르기 
# def cropping_image_from_json(input_dir, output_dir, json_dir, cls, save_img) :
#     dic_bbox_with_point = {}
#     with open(json_dir, "r") as json_file:
#         predicts = json.load(json_file)
#     if not os.path.exists(output_dir) : os.mkdir(output_dir)

#     print("Cropping only LicensePlate Bounding Box from Input Image")
#     index = 0
#     predic
    
#     for idx, detection in enumerate(tqdm(json_dir)):
#         # 박스 변환
#         if isinstance(detection[3], tuple) : box = list(detection[3])
#         else : box = ast.literal_eval(detection[3])

#         file_name, label, conf = detection[0], float(detection[1]), float(detection[2])
#         if label == cls:
#             img_path = os.path.join(input_dir, file_name + ".jpg")
#             img = Image.open(img_path)
#             cropped_img = img.crop(box)
#             if save_img :
#                 cropped_img.save(f'{output_dir}{file_name}_crop_{index:04d}.jpg')
#             if file_name in dic_bbox_with_point:
#                 dic_bbox_with_point[file_name]['subimage'].append(
#                     f'{file_name}_crop_{index:04d}.jpg')
#                 dic_bbox_with_point[file_name]['boxes'].append(box)
#             else:
#                 dic_bbox_with_point[file_name] = {'subimage': [
#                     f'{file_name}_crop_{index:04d}.jpg'], 'boxes': [box]}
#             index += 1
#         else:
#             continue
#     print(
#         f"Cropped result images saved at : {output_dir} \nCropped result images cnt : {len(os.listdir(output_dir))}")
#     return dic_bbox_with_point