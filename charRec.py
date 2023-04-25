#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import json
import time
import cv2
import os

model_path_char = 'models/charRec/charData4_stack_830-8500.pb'

char_set = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
            "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
            "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
            "U", "V","W","X","Y","Z","CL","PC"]

channals=3
char_img_width=28
char_img_height=28

graph_char_rec = tf.Graph()
sess=tf.Session()

with tf.gfile.FastGFile(model_path_char, "rb") as fr:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(fr.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name="")

sess.run(tf.global_variables_initializer())

_inputs = sess.graph.get_tensor_by_name('inputs_char:0')
# _is_training = tf.get_default_graph().get_tensor_by_name('is_training_char:0')
_softmax_output = sess.graph.get_tensor_by_name('softmax_output_char:0')

def pre_process(char_image):
    image = char_image.copy()
    image_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image_gray_3channals=cv2.cvtColor(image_gray,cv2.COLOR_GRAY2BGR)

    image = cv2.resize(image_gray_3channals, (char_img_width, char_img_height))
    image = np.resize(image, (char_img_height, char_img_width, channals))
    image = np.array(image, dtype=np.float32)
    return image

def getCharRec(char_image):
    char_image=pre_process(char_image)
    # print("shape1:",char_image.shape)
    image_data=char_image[np.newaxis,:]
    # print("shape2:", image_data.shape)

    softmax_out=sess.run(_softmax_output,feed_dict={_inputs:image_data})

    label_id=np.argmax(softmax_out,1)
    return char_set[label_id[0]]

if __name__=="__main__":
    pic_dir="D:/forTensorflow/charRecTrain/charData4_stack_826/A"
    for filename in os.listdir(pic_dir):
        filen_path=os.path.join(pic_dir,filename)

        char_image=cv2.imread(filen_path)
        rec=getCharRec(char_image)
        print("rec:",rec)
