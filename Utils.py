import pickle
import numpy as np


def parse_int32(byte_list, start_i):
    val = (byte_list[start_i] << 24) | (byte_list[start_i+1] << 16) \
            | (byte_list[start_i+2] << 8) | byte_list[start_i+3]
    return val


def parse_dataset(img_filepath, label_filepath):
    with open(img_filepath, "br") as f:
        img_raw = f.read()

    num_img = parse_int32(img_raw, 4)
    n_r = parse_int32(img_raw, 8)
    n_c = parse_int32(img_raw, 12)

    img_idx = 16
    imgs = []
    for i in range(num_img):
        start = img_idx + (i * n_r * n_c)
        end = img_idx + ((i + 1) * n_r * n_c)
        img = np.frombuffer(img_raw[start:end], dtype=np.uint8)\
            .reshape(n_r,n_c)
        imgs.append(img.copy())

    with open(label_filepath, "br") as f:
        label_raw = f.read()

    num_label = parse_int32(label_raw, 4)
    labels = label_raw[8:len(label_raw)]

    return num_img, n_r, n_c, imgs, num_label, labels
