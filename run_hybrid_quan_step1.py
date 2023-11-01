import torch
import numpy as np
import cv2
import copy
from rknn.api import RKNN


# RKNN模型参数设定
ONNX_MODEL = './weights/plate_detect.onnx'                # onnx文件路径，rknn.load_onnx()调用需要
DATASET = './dataset.txt'                       # 在选择量化的前提下，进行量化校准的图片


if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN(verbose=False)

    # pre-process config
    print('--> Config model')
    rknn.config(
        mean_values = [[114.57069762499403, 112.06498372297212, 112.0936524574251]],
        std_values = [[56.552368489533805, 59.74804638037939, 59.92431607351682]],
        quantized_dtype = "asymmetric_quantized-8",
        quantized_algorithm = 'normal',
        quantized_method = 'channel',
        optimization_level = 2,
        target_platform = "rk3588")
    print('done')

    # Load ONNX model
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Hybird quantization step 1
    ## (1)Generate the configuration file {model_name}.quantization.cfg
    ## (2)Generate the temporary model file {model_name}.model
    ## (3)Generate the data file {model_name}.data
    print("---> Hybird quantization step 1")
    ret = rknn.hybrid_quantization_step1(dataset=DATASET,
                                         proposal=False)
    if ret != 0:
        print("Hybird quantization step 1 failed!")
        exit(ret)
    print("done")
   
    # release the RKNN model
    rknn.release()
