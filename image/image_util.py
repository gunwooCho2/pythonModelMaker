import cv2
import numpy as np
from numba.cloudpickle import instance


class ImageUtilError(Exception):
    pass

def get_random_color(color_type: str):
    if color_type == 'RGB':
        return tuple(np.random.randint(0, 256, 3).tolist())
    elif color_type == 'GRAY':
        return np.random.randint(0, 256)
    elif color_type == 'BINARY':
        return int(np.random.choice([0, 255]))
    else:
        raise ImageUtilError("Invalid color type")

def over_draw_image(basis_image, source_image, position=(0, 0), is_random=False, is_alpha=False):
    b_size = basis_image.shape
    s_size = source_image.shape

    if len(b_size) == 3:
        if len(s_size) != 3 or b_size[2] != s_size[2]:
            raise ImageUtilError("Image channel mismatch")

    x_gap, y_gap = b_size[1] - s_size[1], b_size[0] - s_size[0]
    if x_gap < 1 or y_gap < 1:
        raise ImageUtilError("basic image muse be bigger than source image")

    source_x, source_y = s_size[1], s_size[0]

    def _top():
        half_x_gap = x_gap // 2
        return half_x_gap, 0, half_x_gap + source_x, source_y

    def _top_right():
        return x_gap , 0, x_gap + source_x, source_y

    def _right():
        half_y_gap = y_gap // 2
        return x_gap, half_y_gap, x_gap + source_x, half_y_gap + source_y

    def _bottom_right():
        return x_gap, y_gap, x_gap + source_x, y_gap + source_y

    def _bottom():
        half_x_gap = x_gap // 2
        return half_x_gap, y_gap, half_x_gap + source_x, y_gap + source_y

    def _bottom_left():
        return 0, y_gap, source_x, y_gap + source_y

    def _left():
        half_y_gap = y_gap // 2
        return 0, half_y_gap, source_x, half_y_gap + source_y

    def _top_left():
        return 0, 0, source_x, source_y

    def _center():
        half_x_gap = x_gap // 2
        half_y_gap = y_gap // 2
        return half_x_gap, half_y_gap, half_x_gap + source_x, half_y_gap + source_y

    def _random():
        random_x, random_y = np.random.randint(0, x_gap), np.random.randint(0, y_gap)
        return random_x, random_y, source_x + random_x, source_y + random_y

    get_coords_functions = {
        (1, 0): _top,
        (1, 1): _top_right,
        (0, 1): _right,
        (-1, 1): _bottom_right,
        (-1, 0): _bottom,
        (-1, -1): _bottom_left,
        (0, -1): _left,
        (1, -1): _top_left,
        (0, 0): _center,
    }

    def __over_draw_alpha_image(basis_image, source_image, start_x, start_y, end_x, end_y):
        roi = basis_image[start_y:end_y, start_x:end_x]
        mask = source_image[..., 3] == 255
        roi[mask] = source_image[mask]
        return basis_image

    result_image = basis_image.copy()
    if is_random:
        start_x, start_y, end_x, end_y = _random()
    else:
        start_x, start_y, end_x, end_y = get_coords_functions[position]()

    if is_alpha:
        result_image = __over_draw_alpha_image(result_image, source_image, start_x, start_y, end_x, end_y)
    else:
        result_image[start_y:end_y, start_x:end_x] = source_image
    return result_image, (start_y, end_y, start_x, end_x)

def image_radius(image, radius_size):
    image_shape = image.shape
    if len(image_shape) != 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    elif image_shape[2] != 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    h, w = image_shape[0], image_shape[1]

    if isinstance(radius_size, int):
        if radius_size > h // 2 or radius_size > w // 2:
            raise ImageUtilError("Radius size must be smaller than half of image dimensions")
    elif isinstance(radius_size, tuple):
        for size in radius_size:
            if size > h // 2 or size > w // 2:
                raise ImageUtilError("Each radius size must be smaller than half of image dimensions")
    else:
        raise ImageUtilError("Radius size must be int or tuple")

    def __set_circle(center, r):
        patch = np.zeros((r, r, 4), dtype=np.uint8)
        color = (255, 255, 255, 255)
        cv2.circle(patch, center, r, color, -1)
        return patch

    def __over_circle(patch, over_coordinate):
        r0, c0 = over_coordinate
        roi = image[r0:r0 + patch.shape[0], c0:c0 + patch.shape[1]]
        roi[..., 3][patch[..., 3] == 0] = 0

    def __top_left(r):
        center = (r, r)
        patch = __set_circle(center, r)
        __over_circle(patch, (0, 0))

    def __top_right(r):
        center = (0, r)
        patch = __set_circle(center, r)
        __over_circle(patch, (0, w - r))

    def __bottom_right(r):
        center = (0, 0)
        patch = __set_circle(center, r)
        __over_circle(patch, (h - r, w - r))

    def __bottom_left(r):
        center = (r, 0)
        patch = __set_circle(center, r)
        __over_circle(patch, (h - r, 0))

    total_functions = [__top_left, __top_right, __bottom_right, __bottom_left]

    if isinstance(radius_size, int):
        for func in total_functions:
            func(radius_size)
    elif isinstance(radius_size, tuple):
        for i, r in enumerate(radius_size):
            total_functions[i](r)

    return image

def binary_image_pack(image):
    # if len(image.shape) == 3 or not (
    #         np.array_equal(np.unique(image), [0]) or
    #         np.array_equal(np.unique(image), [255]) or
    #         np.array_equal(np.unique(image), [0, 255])
    # ):
    #     raise ImageUtilError("Image must be binary.")

    image = image.flatten()
    image[image == 255] = 1
    return np.packbits(image)

def binary_image_unpack(pack_image, size, is_normalization=True):
    image = np.unpackbits(pack_image)
    if is_normalization:
        image[image == 1] = 255
    return image.reshape(size)



