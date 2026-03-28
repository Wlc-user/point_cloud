import cv2
import numpy as np
import os

# 创建一个简单的测试图像
image = np.zeros((400, 600, 3), dtype=np.uint8)
image[:] = (200, 200, 200)  # 浅灰色背景

# 画一些形状
cv2.rectangle(image, (50, 50), (250, 350), (0, 255, 0), 2)
cv2.circle(image, (400, 200), 100, (255, 0, 0), -1)
cv2.putText(image, "Test Image", (200, 380), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# 保存测试图像
cv2.imwrite('test_image.jpg', image)
print("测试图像已创建: test_image.jpg")
