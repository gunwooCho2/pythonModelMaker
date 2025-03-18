import h5py
import os
import numpy as np

class H5pyUtilError(Exception):
    pass

class H5pySave:
    def __init__(self, save_path, batch_size, reset=False):
        if reset and os.path.exists(save_path):
            os.remove(save_path)

        self.save_path = save_path
        self.is_file = False if os.path.exists(save_path) else True
        self.batch_size = batch_size
        self.temporary_dic = {'index': 0}

        dir_path, _ = os.path.split(save_path)
        os.makedirs(dir_path, exist_ok=True)

    def __create_dataset__(self, items):
        with h5py.File(self.save_path, 'w') as f:
            for key in items:
                item = items[key]
                max_shape_val = (None, *item.shape) if isinstance(item, np.ndarray) else (None,)
                f.create_dataset(key, data=[item], maxshape=max_shape_val, compression='gzip')

    def __set_item__(self, items):
        if self.is_file:
            self.__create_dataset__(items)
            self.is_file = False

        for key in items:
            if key == 'index':
                raise H5pyUtilError('Index cannot be changed.')
            self.temporary_dic.setdefault(key, []).append(items[key])
        self.temporary_dic['index'] += 1

        if self.temporary_dic['index'] == self.batch_size:
            self.__save_item__()

    def __save_item__(self):
        with h5py.File(self.save_path, 'a') as f:
            for key in self.temporary_dic:
                if key != 'index':
                    current_size = f[key].shape[0]
                    new_size = current_size + self.temporary_dic['index']
                    f[key].resize(new_size, axis=0)
                    f[key][current_size:new_size] = self.temporary_dic[key]
        self.temporary_dic = {'index': 0}



class H5pyLoad:
    def __init__(self, save_path, batch_size):
        if not os.path.exists(save_path):
            raise H5pyUtilError(f"Can't find file path: {save_path}")

        self.save_path = save_path
        self.batch_size = batch_size
        self.temporary_dic = {'index': 0}

        self.__set_item__(0)

    def shape(self):
        result = {}
        with h5py.File(self.save_path, 'r') as f:
            keys = list(f.keys())
            for key in keys:
                dataset = f[key]
                result[key] = dataset.shape
        return result

    def __get_item__(self, view_index):
        view_index = view_index - self.temporary_dic['index']
        try:
            result = {}
            for key in self.temporary_dic:
                if key != 'index':
                    result[key] = self.temporary_dic[key][view_index]
        except IndexError:
            self.__set_item__(view_index)
            result = self.__get_item__(view_index)
        return result

    def __get_item_all__(self):
        i = 0
        while True:
            yield self.temporary_dic
            i += 1
            try:
                self.__set_item__(i * self.batch_size)
            except H5pyUtilError:
                break

    def __set_item__(self, view_index):
        with h5py.File(self.save_path, 'r') as f:
            keys = list(f.keys())
            for key in keys:
                dataset = f[key]
                dataset_size = dataset.shape[0]
                if dataset_size <= view_index:
                    raise H5pyUtilError("View index is bigger than dataset size.")
                self.temporary_dic[key] = dataset[view_index: view_index + self.batch_size]
            self.temporary_dic['index'] = view_index

