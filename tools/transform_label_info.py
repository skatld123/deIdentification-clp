import os 

def convertVOC2YOLO(size, box):
    # size (w,h)
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = w / 2.0 + box[0]
    y = h / 2.0 + box[1]
    x = round(x / size[0], 4)
    y = round(y / size[1], 4)
    w = round(w / size[0], 4)
    h = round(h / size[1], 4)
    return (x,y,w,h)

def changeFormat(label_path, save_path) :
    file_list = os.listdir(label_path)
    for file in file_list :
        f = open(os.path.join(label_path, file), 'r')
        cls, conf, rx1, ry1, rx2, ry2 = f.readline().strip().split()
        w, h = 4032, 3040
        x1 = round(float(rx1) * w)
        y1 = round(float(ry1) * h)
        x2 = round(float(rx2) * w)
        y2 = round(float(ry2) * h)
        
        nx, ny, nw, nh = convertVOC2YOLO((w, h), [x1, y1, x2, y2])
        
        f = open(os.path.join(save_path, file), 'w+')
        info = f'{cls} {conf} {nx} {ny} {nw} {nh}\n'
        f.write(info)
        

if __name__ == '__main__' : 
    label_path = '/root/deIdentification-clp/result/ensemble_result/labels'
    save_path = '/root/deIdentification-clp/result/ensemble_result/labels_abs'
    changeFormat(label_path, save_path)