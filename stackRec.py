import tensorflow as tf
import numpy as np
import os
import cv2
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

input_width=30
input_height=60
batch_size = 1
char_set = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
            "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
            "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
            "U", "V","W","X","Y","Z","#"]

# model_path = "models/CTC_StackRec/CTC_2num_alphabet_org-20000.pb"
model_path_stack="models/CTC_StackRec/CTC_num2_40000_frezee_my.pb"
max_step_downsampling_num=2


sess1=tf.Session()
with tf.gfile.FastGFile(model_path_stack, "rb") as fr:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(fr.read())
    sess1.graph.as_default()
    tf.import_graph_def(graph_def, name="")

sess1.run(tf.global_variables_initializer())

inputs = sess1.graph.get_tensor_by_name('inputs:0')
seq_len_placeholder = sess1.graph.get_tensor_by_name('seq_len_gt:0')
dense_predictions = sess1.graph.get_tensor_by_name('dense_predictions:0')

def decode_prediction(decode_predictions_):
    is_correct=False
    for ind, val in enumerate(decode_predictions_):
        pred_number = ''
        for code in val:
            pred_number += char_set[code]
        pred_number = pred_number.strip("#")
    return pred_number

def get_stack_rec(stack_image):

        color_image_resized = cv2.resize(stack_image, (input_width, input_height))

        color_image_tran_batch=color_image_resized[np.newaxis,:]

        seq_len = np.ones(batch_size) * input_height/max_step_downsampling_num

        eval_dict={inputs: color_image_tran_batch,seq_len_placeholder: seq_len}

        dense_predictions_ = sess1.run(dense_predictions, feed_dict=eval_dict)

        stack_rec = decode_prediction(dense_predictions_)
        return stack_rec

