# Discription:
#   Read the pred data from the directory `preds` to check the pred outputs

import numpy as np
import torch

onnx_pred_path = "./preds/onnx_pred.npy"    # onnx prediction outputs
rknn_pred_path = "./preds/rknn_pred.npy"    # rknn prediction outputs


def cosine_similarity(matrix1, matrix2):
    vector1 = matrix1.flatten()
    vector2 = matrix2.flatten()

    return np.dot(vector1, vector2) / (
        np.linalg.norm(vector1) * np.linalg.norm(vector2)
    )


if __name__ == "__main__":
    onnx_pred = np.load(onnx_pred_path)
    rknn_pred = np.load(rknn_pred_path)

    ### 统计量打印
    print("*"*25 + " 统计量打印 " + "*"*25)
    print(onnx_pred.shape)
    print("onnx sum: ", onnx_pred.sum())
    print("onnx mean: ", onnx_pred.mean())
    print("onnx std: ", onnx_pred.std())
    print(rknn_pred.shape)
    print("rknn sum: ", rknn_pred.sum())
    print("rknn mean: ", rknn_pred.mean())
    print("rknn std: ", rknn_pred.std())

    ### 余弦相似度
    print("*"*25 + " Cosine Similarity " + "*"*25)
    print("The prediction size: ", onnx_pred.shape)
    for i in range(onnx_pred.shape[-1]):
        print(f"onnx_pred[:,:,{i}] & rknn_pred[:,:,{i}] cosine value :     ", 
              cosine_similarity(onnx_pred[:, :, i], rknn_pred[:, :, i]))
