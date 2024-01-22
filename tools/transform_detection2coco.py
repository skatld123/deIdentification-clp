import json
import os 

"""
    기존의 검출 결과 형식을 COCO Evalution을 위한 형식으로 변경하는 코드
"""

def custom_to_coco_result(label_path, predict_path, output_path) :
    detection_result_coco = []
    with open(label_path, "r") as json_file:
        coco_gt = json.load(json_file)
        
    with open(predict_path, "r") as json_file:
        pd_custom = json.load(json_file)
    
    for image_info in coco_gt['images'] :
        image_file = None
        image_id = image_info['id']
        image_name = os.path.splitext(image_info['file_name'])[0]
        image_h = image_info['height']
        image_w = image_info['width']
        # 커스텀에서 GT에 존재하는 파일을 찾음
        if image_name in pd_custom :
            dt_info = pd_custom[image_name] # boxes[[]], scores[], labels []
            dt_boxes = dt_info["boxes"]
            dt_scores = dt_info["scores"]
            dt_labels = dt_info["labels"]
            # 찾았을 경우 GT에 존재하는 어노테이션을 찾음
            for box, score, lbl in zip(dt_boxes, dt_scores, dt_labels) :
                box = rpToCOCO((image_w, image_h), box)
                detection_result_coco.append(make_detections_results(bbox=box, category_id=int(lbl + 1), image_id=image_id, scores=score))
        else :
            print(f"{image_name} can not found at ResultFile" )
                
    with open(output_path, "w") as json_file:
        json.dump(detection_result_coco, json_file, indent=4)
    return detection_result_coco
    
def make_detections_results(image_id, category_id, bbox, scores) :
    '''
    obj detection의 결과 포맷을 만드는 메서드
    : bbox : [x1, y1, w, h]
    '''
    result = {
        "image_id" : image_id,
        "category_id" : category_id,
        "bbox" : bbox,
        "score" : scores
    }
    return result

def rpToCOCO(imgsz, boxes) :
    '''
    (w, h), (rx1, ry1, rx2, ry2) -> (x1, y1, w, h)
    '''
    # imsz (w,h), boxes (x1, y1, x2, y2)
    x1 = boxes[0] * imgsz[0]
    y1 = boxes[1] * imgsz[1]
    x2 = boxes[2] * imgsz[0]
    y2 = boxes[3] * imgsz[1]
    w = x2 - x1
    h = y2 - y1
    return (x1, y1, w, h)
    
if __name__ == '__main__' :
    
    label_path = '/root/dataset_clp/dataset_v2/test/test.json'
    predict_custom_path = '/root/deIdentification-clp/result/one_stage_result/result.json'
    output_path = '/root/deIdentification-clp/result/one_stage_result'
    detection_result_coco = []
    
    with open(label_path, "r") as json_file:
        coco_gt = json.load(json_file)
        
    with open(predict_custom_path, "r") as json_file:
        pd_custom = json.load(json_file)
    
    for image_info in coco_gt['images'] :
        image_file = None
        image_id = image_info['id']
        image_name = os.path.splitext(image_info['file_name'])[0]
        image_h = image_info['height']
        image_w = image_info['width']
        # 커스텀에서 GT에 존재하는 파일을 찾음
        if image_name in pd_custom :
            dt_info = pd_custom[image_name] # boxes[[]], scores[], labels []
            dt_boxes = dt_info["boxes"]
            dt_scores = dt_info["scores"]
            dt_labels = dt_info["labels"]
            # 찾았을 경우 GT에 존재하는 어노테이션을 찾음
            for box, score, lbl in zip(dt_boxes, dt_scores, dt_labels) :
                box = rpToCOCO((image_w, image_h), box)
                detection_result_coco.append(make_detections_results(bbox=box, category_id=lbl, image_id=image_id, scores=score))
        else :
            print(f"{image_name} can not found at ResultFile" )
                
    output_json_path = os.path.join(output_path, 'result_coco' +'_detection.json')
    with open(output_json_path, "w") as json_file:
        json.dump(detection_result_coco, json_file, indent=4)