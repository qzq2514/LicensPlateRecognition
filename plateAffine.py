import os
import cv2
import numpy as np
import time

#hdf5:   保存模型和参数
#h5:    仅保存参数
#json和yaml:保存模型结构
from keras.models import Model
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D, Input, AveragePooling2D
import keras
from keras.models import load_model
from nets import LprLocNet

input_width = 80
input_height = 40
channals_num=3

show_width = 400
show_height = 200

#plateCornerDetFull90 plateCornerDetFull_all_relu90 plateCornerDetFull_no_agu90
image_path = "D:/forTensorflow/plateLandmarkDetTrain2/MD_WV_All/test/images/"
landmark_model_path="models/landmark/plateCornerDetFull90.h5"
lanmark_object = LprLocNet.LprLocNet(input_width,input_height, channals_num,is_training=True)
lanmark_model=load_model(landmark_model_path)

def getAffinePlate(image_ori):
    org_height, org_width = image_ori.shape[:2]
    processed_image = lanmark_object.preprocess(image_ori)
    inputs = np.array([processed_image])
    time1 = time.time()
    s = lanmark_model.predict(inputs)
    # print("time:", time.time() - time1)
    pst1 = []

    for i in range(4):
        # print(s[0][i], s[0][i + 1])
        x = int(s[0][2 * i] * org_width)
        y = int(s[0][2 * i + 1] * org_height)
        pst1.append([x, y])
        # cv2.circle(image_ori, (x, y), 2, colors[i], 2)
    pst1 = np.float32(pst1)

    # 原始仿射
    past2 = np.float32([[0, 0], [show_width, 0], [show_width, show_height], [0, show_height]])
    M = cv2.getPerspectiveTransform(pst1, past2)
    dst = cv2.warpPerspective(image_ori, M, (show_width, show_height))

    # 宽松仿射-左右放宽30像素,上下不放宽
    padding = 20
    past2 = np.float32([[padding, 30 / 2], [show_width - padding, 30 / 2], [show_width - padding, show_height - 30 / 2],
                        [padding, show_height - 30 / 2]])
    M2 = cv2.getPerspectiveTransform(pst1, past2)
    dst2 = cv2.warpPerspective(image_ori, M2, (show_width, show_height))

    return dst

if __name__ == '__main__':
    lanmark_model.summary()
    print(type(lanmark_model))

    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (150, 150, 150)]
    for j, image_file in enumerate(os.listdir(image_path)):
        if not image_file.endswith("jpg"):
           continue
        image_ori = cv2.imread(os.path.join(image_path, image_file))
        dst=getAffinePlate(image_ori)

        cv2.imshow("img", image_ori)
        cv2.imshow("img2", dst)
        cv2.waitKey(0)
