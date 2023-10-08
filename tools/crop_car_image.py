import math
import os
import cv2


def convertToAbsoluteValues(size, box):
    
    xCenter = round(float(box[0]) * size[0])
    yCenter = round(float(box[1]) * size[1])
    width = round(float(box[2]) * size[0])
    height = round(float(box[3]) * size[1])
    
    xIn = xCenter - (width // 2)
    yIn = yCenter - (height // 2)
    xEnd = xCenter + (width // 2)
    yEnd = yCenter + (height // 2)
    
    if xIn < 0:
        xIn = 0
    if yIn < 0:
        yIn = 0
    if xEnd >= size[0]:
        xEnd = size[0] - 1
    if yEnd >= size[1]:
        yEnd = size[1] - 1
        
    return (xIn, yIn, xEnd, yEnd)

def convertVOC2YOLO(size, box):
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = w / 2.0 + box[0]
    y = h / 2.0 + box[1]
    x = round(x / size[0], 4)
    y = round(y / size[1], 4)
    w = round(w / size[0], 4)
    h = round(h / size[1], 4)
    return (x,y,w,h)

def getArea(box):
    return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)


def getUnionAreas(boxA, boxB, interArea=None):
    area_A = getArea(boxA)
    area_B = getArea(boxB)
    
    if interArea is None:
        interArea = getIntersectionArea(boxA, boxB)
        
    return float(area_A + area_B - interArea)

def getIntersectionArea(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # intersection area
    return (xB - xA + 1) * (yB - yA + 1)

def boxesIntersect(boxA, boxB):
    if boxA[0] > boxB[2]:
        return False  # boxA is right of boxB
    if boxB[0] > boxA[2]:
        return False  # boxA is left of boxB
    if boxA[3] < boxB[1]:
        return False  # boxA is above boxB
    if boxA[1] > boxB[3]:
        return False  # boxA is below boxB
    return True

def main() :
    input_img_dir = "/root/dataset_clp/dataset_v2/test"
    output_img_dir = "/root/dataset_clp/crop_dataset/test"
    
    os.makedirs(output_img_dir, exist_ok=True)
    
    if not os.path.exists(input_img_dir) :
        print("Input Image DIR is wrong")
        return
    else : 
        img_dir = os.path.join(input_img_dir, "images")
        lbl_dir = os.path.join(input_img_dir, "labels")
        
        oimg_dir = os.path.join(output_img_dir, "images")
        olbl_dir = os.path.join(output_img_dir, "labels")

        os.makedirs(oimg_dir, exist_ok=True)
        os.makedirs(olbl_dir, exist_ok=True)
        
        img_list = sorted(os.listdir(img_dir))
        lbl_list = sorted(os.listdir(lbl_dir))
        
        if not len(img_list) == len(lbl_list) :
            print("이미지 개수와 라벨 개수가 다릅니다.")
            return
        
        for img_path, lbl_path in zip(img_list, lbl_list) :
            print(lbl_path)
            img = cv2.imread(os.path.join(img_dir, img_path))
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            h, w, _ = img.shape
            vehicle_list = []
            license_plate_list = []
            with open(os.path.join(lbl_dir, lbl_path), 'r') as f:
                annotations = f.readlines()
                for annotation in annotations :
                    cls, rx, ry, rw, rh = map(float, annotation.strip().split())
                    x1, y1, x2, y2 = convertToAbsoluteValues((w, h), (rx, ry, rw, rh))
                    if cls == 1 : 
                        vehicle_list.append([cls, x1, y1, x2, y2])
                    else :  
                        license_plate_list.append([cls, x1, y1, x2, y2])
            new_lbl_path = os.path.join(olbl_dir, base_name)
            new_img_path = os.path.join(oimg_dir, base_name)
            for idx, car in enumerate(vehicle_list) : 
                crop_area = [car[1], car[2], car[3], car[4]]
                crop_img = img[car[2] : car[4], car[1] : car[3]]
                ch, cw, _ = crop_img.shape
                for lp in license_plate_list :
                    inner_area = [lp[1], lp[2], lp[3], lp[4]]
                    if boxesIntersect(crop_area, inner_area) :
                        new_x1 = lp[1] - car[1]
                        new_y1 = lp[2] - car[2]
                        new_x2 = lp[3] - car[1]
                        new_y2 = lp[4] - car[2]
                        new_box = [new_x1, new_y1, new_x2, new_y2]
                        rectcolor = (0, 0, 255)
                        # cv2.rectangle(crop_img, (new_x1, new_y1), (new_x2, new_y2), rectcolor, 4)
                        nx, ny, nw, nh = convertVOC2YOLO((cw, ch), new_box)
                        
                        if not os.path.exists(f'{new_lbl_path}_{idx:03d}.txt') :
                            with open(f'{new_lbl_path}_{idx:03d}.txt',"w") as f :
                                f.write(f'0 {nx} {ny} {nw} {nh}\n')
                        else :
                            with open(f'{new_lbl_path}_{idx:03d}.txt',"a") as f :
                                f.write(f'0 {nx} {ny} {nw} {nh}\n')
                                
                        if not os.path.exists(f'{new_img_path}_{idx:03d}.jpg') :
                            cv2.imwrite(f'{new_img_path}_{idx:03d}.jpg', crop_img)
                            
    print(f'outputImg : {len(os.listdir(oimg_dir))}')
    print(f'outputlbl : {len(os.listdir(olbl_dir))}')
    if not len(oimg_dir) == len(olbl_dir) :
        print("이미지 개수와 라벨 개수가 다릅니다.")

if __name__ == '__main__' : 
    main()