import os
from pickletools import uint8

import cv2
import h5py
import numpy as np

from chess.image.board_maker import get_image
from dataset.h5py_util import H5pySave, H5pyLoad
from image.image_util import image_radius, over_draw_image, binary_image_pack, binary_image_unpack
from image.noise_maker import set_line_noise, set_salt_and_pepper_noise
from image.scaling_image import get_edge_line_image
from video.video_to_data import VideoToData
from video.youtube import download


def main():
    video_path = "C:/Data/chess_board_detect/video"
    youtube_url = "https://www.youtube.com/watch?v=cb7VlXqFla4"
    video_name = "basis_youtube"
    dataset_path = "C:/Data/chess_board_detect/train_data/chess_dataset.h5"

    os.makedirs(video_path, exist_ok=True)
    if not os.path.exists(os.path.join(video_path, video_name + ".mp4")):
        download(video_path, video_name, youtube_url)

    h5py_save = H5pySave(dataset_path, 512, True)

    class VideoToDataChess(VideoToData):
        def __init__(self, video_path):
            super().__init__(video_path)
        def __start__(self, image):
            h5py_save.__set_item__(get_data(image, True))
        def __to_data__(self, image):
            h5py_save.__set_item__(get_data(image, False))
        def __end__(self, image):
            h5py_save.__save_item__()

    video_to_data_chess = VideoToDataChess(os.path.join(video_path, video_name + ".mp4"))
    video_to_data_chess.__run__()


def get_data(basis_image, is_first):
    result_dic = {
        'is_true': np.array([np.random.choice([True, False])]) if not is_first else np.array([True]),
        'train': None,
        'label': None
    }

    random_size_basis_image = set_basis_size_random(basis_image, 257, 3)
    if not result_dic['is_true'][0]:
        train_image = get_edge_line_image(random_size_basis_image, 20)
        train_image = cv2.resize(train_image, (512, 512), interpolation=cv2.INTER_NEAREST)
        train_image = binary_image_pack(train_image)
        label_image = np.zeros_like(train_image)
        result_dic['train'] = train_image
        result_dic['label'] = label_image
        return result_dic

    train_image, label_image = get_true_images(random_size_basis_image)
    result_dic['train'] = train_image
    result_dic['label'] = label_image

    return result_dic

def get_true_images(basis_image):
    b_h, b_w = basis_image.shape[:2]
    basis_zero_image = np.zeros((b_h, b_w, 3), dtype=np.uint8)

    board_image = get_image(out_put_size=256)
    set_salt_and_pepper_noise(board_image, 0.025)
    random_radius_size = int(np.random.choice([i for i in range(11)]))

    if random_radius_size != 0:
        board_image = image_radius(board_image, random_radius_size)
        basis_image = cv2.cvtColor(basis_image, cv2.COLOR_BGR2BGRA)
        basis_zero_image = cv2.cvtColor(basis_zero_image, cv2.COLOR_BGR2BGRA)
        train_image, coordinate = over_draw_image(basis_image, board_image, is_random=True, is_alpha=True)

    else:
        train_image, coordinate = over_draw_image(basis_image, board_image, is_random=True)

    basis_zero_image[coordinate[0]:coordinate[1], coordinate[2]:coordinate[3]] = board_image
    train_image = get_edge_line_image(train_image, 20)
    label_image = get_edge_line_image(basis_zero_image, 20)
    train_image = cv2.resize(train_image, (512, 512), interpolation=cv2.INTER_NEAREST)
    label_image = cv2.resize(label_image, (512, 512), interpolation=cv2.INTER_NEAREST)
    train_image = binary_image_pack(train_image)
    label_image = binary_image_pack(label_image)

    return train_image, label_image

def set_basis_size_random(basis_image, standard_size, max_value: int):
    h, w = basis_image.shape[:2]
    new_h = min(np.random.randint(standard_size, standard_size * max_value), h)
    new_w = min(np.random.randint(standard_size, standard_size * max_value), w)
    if new_h > h:
        new_h = h
    if new_w > w:
        new_w = w

    start_x = np.random.randint(0, w - new_w)
    start_y = np.random.randint(0, h - new_h)

    return basis_image[start_y:start_y + new_h, start_x:start_x + new_w]

if __name__ == "__main__":
    # main()

    h5py_load = H5pyLoad("C:/Data/chess_board_detect/train_data/chess_dataset.h5", 10)
    file_shape = h5py_load.shape()
    index = file_shape['is_true'][0]

    for i in range(index):
        file = h5py_load.__get_item__(i)
        for key in file:
            if key == 'is_true':
                print(file[key])
            else:
                cv2.imshow(key, binary_image_unpack(file[key], (512, 512)))
                cv2.waitKey(0)
                cv2.destroyAllWindows()

