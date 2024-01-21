import cv2

# 读取两张图片
image1 = cv2.imread('./000/00000000.png')
image2 = cv2.imread('./000/00000001.png')

# 确保两张图片的尺寸一致，如果不一致，可以使用resize函数来调整它们的大小
if image1.shape != image2.shape:
    image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

# 对两张图片进行相减
result = cv2.subtract(image1, image2)

# 保存结果图片
cv2.imwrite('result.png', result)

# 显示结果图片
cv2.imshow('Subtraction Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
