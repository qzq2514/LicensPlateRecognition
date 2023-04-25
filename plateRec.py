import cv2
import numpy as np
from Rect import *

net=cv2.dnn.readNetFromCaffe\
    ("models/plateRec/CGIM_rec_deploy.prototxt",
     "models/plateRec/CGIM_rec_deploy.caffemodel")

label_dict={
0:"background" ,
1:"1" , 2:"2" , 3:"3" , 4:"4" , 5:"5" , 6:"6" , 7:"7" , 8:"8" , 9:"9" , 10:"0" ,
11:"A" , 12:"B" , 13:"C" , 14:"D" , 15:"E" , 16:"F" , 17:"G" , 18:"H" , 19:"J" , 20:"K" ,
21:"L" , 22:"M" , 23:"N" , 24:"P" , 25:"Q" , 26:"R" , 27:"S" , 28:"T" , 29:"U" , 30:"V" ,
31:"W" , 32:"X" , 33:"Y" , 34:"Z" , 35:"I" , 36:"two" , 37:"three" }

#输入元素类型为Rect的list集合和IoU阈值,返回最终保留的rects下标
def NMS(rects,IoU_threshold):
    xmin = np.array([rect.xmin for rect in rects])
    ymin = np.array([rect.ymin for rect in rects])
    xmax = np.array([rect.xmax for rect in rects])
    ymax = np.array([rect.ymax for rect in rects])
    scores = np.array([rect.conf for rect in rects])

    areas = (xmax-xmin+1)*(ymax-ymin+1)

    #按照置信度从高到低排序
    order = scores.argsort()[::-1]
    keep_indices=[]

    while order.size > 0:
        #order[0]是当前分数最大的窗口,肯定保留
        cur_max_socre_ind=order[0]
        keep_indices.append(cur_max_socre_ind)

        #计算窗口cur_max_socre_ind与其他窗口的重叠部分面积
        xxmin = np.maximum(xmin[cur_max_socre_ind], xmin[order[1:]])
        yymin = np.maximum(ymin[cur_max_socre_ind], ymin[order[1:]])
        xxmax = np.minimum(xmax[cur_max_socre_ind], xmax[order[1:]])
        yymax = np.minimum(ymax[cur_max_socre_ind], ymax[order[1:]])

        w = np.maximum(0.0,xxmax-xxmin+1)
        h = np.maximum(0.0,yymax-yymin+1)
        inter = w * h   #当前最大的rect与其他rects的相交面积

        union = areas[cur_max_socre_ind] + areas[order[1:]] - inter

        overlap = inter/union

        # 得到满足与当前最大得分的Rect交并比小于阈值的其他rects的原始下标inds
        # 下标的顺序还是一样的,即返回的原rects下标为inds[0]+1是当前score最大的rect
        # ps:  np.where返回的是展开后下标(indices),即indices[0]是所有满足条件的元素下标的第一维的值
        inds = np.where(overlap<=IoU_threshold)[0]

        # 之所加一的原因是由于overlap长度比order长度少1(不包含cur_max_socre_ind),
        # 所以inds+1对应到保留的窗口
        order = order[inds+1]
    return keep_indices

def getRecRects(image):
    image=image.copy()

    (h, w, c) = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.007843, (450, 300), 127.5, False, False)
    net.setInput(blob, "data")
    detection = net.forward("detection_out")
    plateRecRecs = []
    for i in np.arange(0, detection.shape[2]):
        conf = detection[0, 0, i, 2]
        indx = int(detection[0, 0, i, 1])
        if conf > 0 and indx in label_dict.keys():
            print("{}:{}".format(label_dict[indx],conf))
            box = detection[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # width=endX-startX
            # height=endY-startY
            plateRecRecs.append(Rect(label_dict[indx],startX,startY,endX,endY,conf))

    plateRecRecs = sorted(plateRecRecs, key=lambda rect: rect.xmin)
    for rect in plateRecRecs:
        name = rect.name
        conf = rect.conf
        # print("{}:{}".format(name, conf))
    nms_indices = NMS(plateRecRecs, 0.7)
    # print(nms_indices)
    plateRecRecs_nms = [rect for ind, rect in enumerate(plateRecRecs)
                        if ind in nms_indices]
    return plateRecRecs_nms  # plateRecRecs_nms    plateRecRecs