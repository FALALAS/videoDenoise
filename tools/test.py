import numpy as np

# 创建一个3x4的二维矩阵
matrix = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print("原始矩阵:\n", matrix)

# 指定要减去的数为5
num_to_subtract = 5
result = matrix - num_to_subtract
print("\n结果矩阵:\n", result)