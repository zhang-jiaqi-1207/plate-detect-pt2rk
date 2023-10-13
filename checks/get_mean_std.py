import cv2
import numpy as np

image_path = "./imgs/Quicker_20220930_180919.png"
save_txt_path = "./checks/mean_std.txt"


def cv_imread(path):  #可以读取中文路径的图片
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    return img


if __name__ == "__main__":
    img = cv_imread(image_path)
    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    print(img.shape)

    mean_lst = []
    std_lst = []
    content = ""

    # Iterate over the color channels (B, G, R) 
    for ch in range(3):
        print("- "*50)

        (mean_val,) = img[:,:,ch].mean(),
        (std_val,) = img[:,:,ch].std(),

        result = "mean : {}\nstd : {}\n".format(mean_val, std_val)
        print(result)

        mean_lst.append(mean_val)
        std_lst.append(std_val)
    
    # Saving the data to a disk file
    content += "mean: [" + ", ".join(str(num) for num in mean_lst[::-1]) + "]"
    content += "\n"
    content += "std: [" + ", ".join(str(num) for num in std_lst[::-1]) + "]"
    with open(save_txt_path, "w") as f:
        f.write(content)
