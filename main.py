import os
import cv2
import time
import plateAffine as plateAffine
import plateRec_my as plateRec
import stackRec_openvino as stackRec
import charRec as charRec

test_pics_dir = "./test_images"
# test_pics_dir = "F:/forTensorflow/plateLandmarkDetTrain2/with_tag/MD_WV_all/test/images/"
# test_pics_dir = "D:/forTensorflow/plateLandmarkDetTrain2/with_tag/MD/images/"   #WV_tu/images/
# test_pics_dir = "D:/forTensorflow/plateLandmarkDetTrain2/MD_WV_All/test/images/"
# test_pics_dir = "D:/forCaffe/plateRec/onlyStack/MD/test"

# test_pics_dir = "D:/forCaffe/plateRec/onlyStack/MD_WV"
# test_pics_dir = "D:/forCaffe/plateRec/All/plates/"



def tensor_char_stackRec(image):
    image_h,image_w=image.shape[:2]
    chat_top_img=image[:int(image_h/2),0:image_w,:]
    char_bottom_img=image[int(image_h/2):, 0:image_w,:]

    char_top_rec = charRec.getCharRec(chat_top_img)
    char_bottom_rec = charRec.getCharRec(char_bottom_img)

    return char_top_rec+char_bottom_rec

if __name__ == '__main__':

    total_num=0;correct_num=0
    total_stack_num=0;correct_stack_num=0
    used_time=0
    for filename in os.listdir(test_pics_dir):
        # ground_truth=filename.split("_")[0].split("-")[1][:-1]

        ind1=filename.find("[TAG-")
        ind2=filename.find("]")
        ground_truth=filename[ind1+5:ind2]

        file_path=os.path.join(test_pics_dir,filename)
        print(file_path,"---------------------",ground_truth)
        plate_image_org=cv2.imread(file_path)

        start_time=time.time()
        affine_plate=plateAffine.getAffinePlate(plate_image_org)

        plate_image=affine_plate.copy()   #affine_plate
        image_gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        plate_iamge = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
        rec_rects=plateRec.getRecRects(plate_iamge)

        plate_show=plate_image.copy()
        plate_rec_tag=""
        have_stack=False
        plate_iamge_H,plate_iamge_W=plate_iamge.shape[:2]
        for rect in rec_rects:
            xmin,ymin,xmax,ymax=rect.xmin,rect.ymin,rect.xmax,rect.ymax
            show_tag = ""
            if rect.name=="two":
                xmin=max(0,xmin-2)   # xmin-2
                xmax=min(plate_iamge_W, xmax+5) # xmax+5
                ymax=min(plate_iamge_H, ymax+2)
                stack_char_image=plate_image[ymin:ymax,xmin:xmax]
                try:
                    stack_rec_ctc=stackRec.get_stack_rec(stack_char_image)
                except:
                    total_stack_num-=1
                    total_num-=1
                    print("Error happen in get_stack_rec function!")

                # stack_rec_ctc=""                    #仅仅使用字符分割
                stack_rec_seg=None
                if len(stack_rec_ctc)!=2:   #True:#
                    stack_rec_seg=tensor_char_stackRec(stack_char_image)
                    plate_rec_tag += stack_rec_seg
                    show_tag = stack_rec_seg
                else:
                    plate_rec_tag += stack_rec_ctc
                    show_tag = stack_rec_ctc
                # print("stack_rec:{}-----{}".format(stack_rec_ctc,stack_rec_seg))
                total_stack_num+=1
                have_stack=True
            else:
                plate_rec_tag += rect.name
                show_tag=rect.name

            cv2.putText(plate_show,show_tag,(xmin+2, ymin-2),1,2,(0,0,255),thickness=2)
            cv2.rectangle(plate_show, (xmin+2, ymin+2), (xmax+2, ymax+2), (0, 0, 255),thickness=2)

        used_time += time.time() - start_time

        total_num+=1
        is_correct = ground_truth==plate_rec_tag
        correct_num = correct_num+1 if is_correct else correct_num
        correct_stack_num = correct_stack_num+1 if is_correct and have_stack else correct_stack_num
        print_info="{}------->{} \n" \
                   "total_num:{},accuarcy:{} \n" \
                   "stack_total_num:{},accuarcy:{}\n" \
                   "avg_time:{}".\
                    format(ground_truth,plate_rec_tag,
                    total_num,correct_num/total_num,
                    total_stack_num,
                    0 if total_stack_num==0 else
                    correct_stack_num/total_stack_num,
                    used_time/total_num)
        print(print_info)
        print("------------------------------")

        # if is_correct and not have_stack:
        # plate_show = cv2.resize(plate_show,(0,0), fx=2, fy=2)
        # cv2.putText(plate_show, plate_rec_tag, (20, 50), 1, 4, (0, 0, 255), 4)
        #     cv2.imwrite("results/{}".format(filename),plate_show)
        cv2.imshow("plate_image_org", plate_image_org)
        cv2.imshow("plate_show", plate_show)
        cv2.waitKey(0)



