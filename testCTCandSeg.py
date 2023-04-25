import os
import cv2
import time
import stackRec_openvino as stackRec
import charRec as charRec



def tensor_char_stackRec(image):
    image_h,image_w=image.shape[:2]
    chat_top_img=image[:int(image_h/2),0:image_w,:]
    char_bottom_img=image[int(image_h/2):, 0:image_w,:]

    char_top_rec = charRec.getCharRec(chat_top_img)
    char_bottom_rec = charRec.getCharRec(char_bottom_img)

    return char_top_rec+char_bottom_rec

if __name__ == '__main__':
    pics_dir="D:/forTensorflow/stackChars/stackChars2Num_all_test/"
    total_num=0
    correct_num_CTC = 0
    correct_num_seg = 0
    correct_num_all = 0
    for image_name in os.listdir(pics_dir):
        image_path=os.path.join(pics_dir,image_name)
        image = cv2.imread(image_path)
        tag = image_name[:image_name.find("_")]
        total_num += 1

        #CTC识别
        pred_CTC = stackRec.get_stack_rec(image)
        ctc_correct = tag==pred_CTC
        correct_num_CTC = correct_num_CTC+1 if ctc_correct else correct_num_CTC
        correct_num_all = correct_num_all+1 if ctc_correct else correct_num_all
        print_info="CTC:{}------->{} \n" \
                   "total_num:{},accuarcy:{}"
        print(print_info.format(tag,pred_CTC,total_num,correct_num_CTC/total_num))

        #强行分割识别
        pred_seg = tensor_char_stackRec(image)
        seg_correct = tag == pred_seg
        correct_num_seg = correct_num_seg + 1 if seg_correct else correct_num_seg
        print_info = "Seg:{}------->{} \n" \
                     "total_num:{},accuarcy:{} "
        print(print_info.format(tag, pred_seg, total_num, correct_num_seg / total_num))

        #综合识别
        if len(pred_CTC)!=2:
            correct_num_all = correct_num_all+1 if seg_correct else correct_num_all
        print_info = "All:{}------->{} | {} \n" \
                     "total_num:{},accuarcy:{}"
        print(print_info.format(tag, pred_CTC,pred_seg, total_num, correct_num_all / total_num))
        print("--------------------------------------")
        # cv2.imshow("image",image)
        # cv2.waitKey(0)