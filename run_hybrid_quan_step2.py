import torch
import numpy as np
import cv2
import copy
from rknn.api import RKNN

from plate_recognition.plate_rec import init_model, cv_imread
from utils.post_process import detect_Recognition_plate, detect_Recognition_plate_onnx, detect_Recognition_plate_rknn, draw_result, load_model


# RKNN模型参数设定
ONNX_MODEL = './weights/plate_detect.onnx'                # onnx文件路径，rknn.load_onnx()调用需要
RKNN_MODEL = './weights/plate_detect.rknn'                # rknn导出路径位置
IMG_PATH = '/mnt/c/Users/jxb/MyFiles/车牌/地库/苏F188W6.jpg'             # 选择测试的图片


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# detect_model = load_model("./weights/plate_detect.pt", device)  #初始化检测模型
plate_rec_model = init_model(device, "./weights/plate_rec_color.pth", is_color = True)      #初始化识别模型
onnx_path = "./weights/plate_detect.onnx"


if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN(verbose=False)

    # Build model
    print('--> Hybrid quantization step2')
    ret = rknn.hybrid_quantization_step2(model_input='./plate_detect.model',
                                         data_input='./plate_detect.data',
                                         model_quantization_cfg='./plate_detect.quantization.cfg')
    if ret != 0:
        print('hybrid_quantization_step2 failed!')
        exit(ret)
    print('done')

    # Accuracy analysis
    print('--> Accuracy analysis')
    ret = rknn.accuracy_analysis(inputs=[IMG_PATH],
                                 output_dir="./snapshot/hybrid")
    if ret != 0:
        print('Accuracy analysis failed!')
        exit(ret)
    print('done')

    # # Export RKNN model
    # print('--> Export rknn model')
    # ret = rknn.export_rknn(RKNN_MODEL)
    # if ret != 0:
    #     print('Export rknn model failed!')
    #     exit(ret)
    # print('done')

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    ## Inference
    # (1) 读取识别车牌图片
    img = cv_imread(IMG_PATH)
    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # (2) 通过pt, onnx, rknn文件检测车牌，并识别别车牌
    # dict_list_torch = detect_Recognition_plate(detect_model,
    #                                            img,
    #                                            device,
    #                                            plate_rec_model,
    #                                            img_size = 640,
    #                                            is_color = True)
    dict_list_onnx = detect_Recognition_plate_onnx(onnx_path,
                                                img,
                                                device,
                                                plate_rec_model,
                                                img_size = 640,
                                                is_color = True) 
    dict_list_rknn = detect_Recognition_plate_rknn(rknn,
                                                img,
                                                device,
                                                plate_rec_model,
                                                img_size = 640,
                                                is_color = True)
    
    # (3) 将onnx模型文件输出结果 与 rknn模型文件推理结果 输出比对
    # information_template = "- "*60 + "\n" + "{}" + "\n" + "- "*60
    # print("-> "*10 + " ONNX OUTPUT " + "<- "*10)
    # print(information_template.format(dict_list_onnx))
    # print("-> "*10 + " RKNN OUTPUT " + "<- "*10)
    # print(information_template.format(dict_list_rknn))
 

    # (4) 将结果保存为图片
    # img_torch = copy.deepcopy(img)
    # print("torch output:    ", end="")
    # ori_img_torch = draw_result(img_torch, dict_list_torch)
    # cv2.imwrite("./results/result_torch.jpg", ori_img_torch)

    img_onnx = copy.deepcopy(img)
    print("onnx output:    ", end="")
    ori_img_onnx = draw_result(img_onnx, dict_list_onnx)
    cv2.imwrite("./results/result_onnx.jpg", ori_img_onnx) 

    img_rknn = copy.deepcopy(img)
    print("rknn output:    ", end="")
    ori_img_rknn = draw_result(img_rknn, dict_list_rknn)
    cv2.imwrite("./results/result_rknn.jpg", ori_img_rknn) 

    # release the RKNN model
    rknn.release()
