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
        :param x: 1D array of log returns
        :return: 2D GAF Image (size x size)
        """
        # 핵심 로직 (Fixed Scaling)
        # 시장 변동성이 +/- 5% (0.05)를 넘어가면 이미지를 꽉 채우고(-1 or 1),
        # 그보다 작으면 흐릿하게 표현하여 '강도'를 보존함.
        limit = 0.05
        x_scaled = np.clip(x / limit, -1, 1)
        
        # 1D array -> 2D GAF Image 변환
        return self.gaf.fit_transform(x_scaled.reshape(1, -1))[0]