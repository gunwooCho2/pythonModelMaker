import cv2
from tqdm import tqdm
from abc import ABC, abstractmethod

class VideoToData(ABC):
    def __init__(self, video_path):
        self.video_path = video_path
        self.is_first = True

    def __run__(self):
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        last_image = None

        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if self.is_first:
                    self.__start__(frame)
                    self.is_first = False
                self.__to_data__(frame)
                last_image = frame
                pbar.update(1)
        cap.release()

        self.__end__(last_image)

    @abstractmethod
    def __start__(self, image):
        pass
    @abstractmethod
    def __to_data__(self, image):
        pass
    @abstractmethod
    def __end__(self, image):
        pass

