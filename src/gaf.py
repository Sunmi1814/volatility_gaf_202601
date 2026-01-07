# src/gaf.py
import numpy as np
from pyts.image import GramianAngularField

class GAFEncoder:
    def __init__(self, size):
        # sample_range=None : pyts의 자동 스케일링을 꺼야 우리가 수동으로 제어 가능
        # method='summation' : s = cos(a+b) 방식 사용
        self.gaf = GramianAngularField(image_size=size, method="summation", sample_range=None)

    def encode(self, x):
        """
        x: 이미 [-1, 1]로 scaling된 1D array
        """
        return self.gaf.fit_transform(x.reshape(1, -1))[0]