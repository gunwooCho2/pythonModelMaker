import numpy as np
import cv2

from image.image_util import over_draw_image, image_radius
from image.noise_maker import set_salt_and_pepper_noise, set_line_noise
from image.scaling_image import get_edge_line_image

def get_image(shell_size = 32, line_size=None, out_put_size=None, out_line=True):
    if line_size is None:
        line_size = (0, 3)

    line_size = np.random.randint(*line_size)

    if out_put_size is None:
        out_put_size = shell_size * 8 + line_size * 7

    shell_image_dic = {
        0: __get_shell(shell_size, (227, 236, 249)),
        1: __get_shell(shell_size, (121, 134, 144))
    }

    shell_images = [shell_image_dic.get(0), shell_image_dic.get(1)]
    vertical_line = None
    horizontal_line = None
    line_rgb = (0, 0, 0)

    if line_size != 0:
        vertical_line = __get_line(shell_size, line_size, line_rgb)
        horizontal_line = __get_line(shell_size * 8 + line_size * 7, line_size, line_rgb, vertical=False)

    horizontal_image = __get_add_shells(shell_images * 4, vertical_line)
    flipped_horizontal_image = cv2.flip(horizontal_image, 1)

    board_image = __get_add_shells([horizontal_image, flipped_horizontal_image] * 4, horizontal_line, vertical=True)

    if out_line and line_size != 0:
        vertical_total_line = __get_line(shell_size * 8 + line_size * 9, line_size, line_rgb)
        board_image = np.concatenate((horizontal_line, board_image, horizontal_line), axis=0)
        board_image = np.concatenate((vertical_total_line, board_image, vertical_total_line), axis=1)

    board_image = board_image.astype(np.uint8)
    return cv2.resize(board_image, (out_put_size, out_put_size))

def __get_add_shells(shell_images, line_image, vertical=False):

    if line_image is None:
        images = shell_images
    else:
        images = [item for sublist in [[i, line_image] for i in shell_images] for item in sublist]
        images.pop()

    if vertical:
        return np.concatenate(tuple(images), axis=0)
    else:
        return np.concatenate(tuple(images), axis=1)

def __get_shell(shell_size, rgb):
    return np.tile(rgb, (shell_size, shell_size, 1))

def __get_line(shell_size, line_size, rgb, vertical = True):
    if vertical:
        return np.tile(rgb, (shell_size, line_size, 1))
    else:
        return np.tile(rgb, (line_size, shell_size, 1))

if __name__ == "__main__":
    while True:
        # output_size = 256
        # board_image = get_image(out_put_size=output_size, out_line=True)
        # set_line_noise(board_image, (50, 100), (1, 2), (10, 20))
        # set_salt_and_pepper_noise(board_image, 0.025)

        basis_board_image = get_image(out_put_size=400)
        source_board_image = get_image(out_put_size=256)
        basis_board_image = cv2.cvtColor(basis_board_image, cv2.COLOR_BGR2BGRA)
        source_board_image = image_radius(source_board_image, 5)
        basis_board_image = over_draw_image(basis_board_image, source_board_image, is_alpha=True)[0]
        #
        set_salt_and_pepper_noise(basis_board_image, 0.025)
        board_image = get_edge_line_image(basis_board_image, 20)

        cv2.imshow("Board Image", board_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()