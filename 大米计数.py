# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 20:46:45 2023

@author: lvyan
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 读取图片
img = cv2.imread('rice.png', cv2.IMREAD_GRAYSCALE)

# 2. 高斯滤波
img_blur = cv2.GaussianBlur(img, (5, 5), 0)

# 3. 二值化图像
_, thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_OTSU)

# 4. 定义结构元素
kernel = np.ones((5, 5), np.uint8)

# 5. 腐蚀
erosion = cv2.erode(thresh, kernel, iterations=2)

# 6. 膨胀
dilation = cv2.dilate(thresh, kernel, iterations=2)

# 7. 开运算
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# 8. 闭运算
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# 9. 计数大米数量
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closing)
num_rice = num_labels - 1  # 减去背景

# 使用matplotlib显示结果
fig, axes = plt.subplots(1, 5, figsize=(20, 20))

axes[0].imshow(img, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(erosion, cmap='gray')
axes[1].set_title('Erosion')
axes[1].axis('off')

axes[2].imshow(dilation, cmap='gray')
axes[2].set_title('Dilation')
axes[2].axis('off')

axes[3].imshow(opening, cmap='gray')
axes[3].set_title('Opening')
axes[3].axis('off')

axes[4].imshow(closing, cmap='gray')
axes[4].set_title(f'Closing - {num_rice} Grains')
axes[4].axis('off')

plt.savefig("rice_count.png",dpi=500,bbox_inches="tight")
plt.show()

print(f"Total number of rice grains: {num_rice}")









