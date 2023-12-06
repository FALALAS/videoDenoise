import os

# 将这个路径替换为您的 '0001clean' 文件夹的实际路径
folder = '0001clean'

# 遍历文件夹中的所有文件
for filename in os.listdir(folder):
    if filename.startswith("frame_"):
        # 构建旧文件的完整路径
        old_file = os.path.join(folder, filename)

        # 从文件名中去除 "frame_" 部分
        new_filename = filename.replace("frame_", "")
        new_file = os.path.join(folder, new_filename)

        # 重命名文件
        os.rename(old_file, new_file)

