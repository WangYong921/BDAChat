from PIL import Image

# 打开图像文件
img = Image.open("hcpost.jpg")

# 获取图像尺寸
img_width, img_height = img.size

# 设置每个块的大小
block_size = 2048

# 分割图像并保存
block_count = 1
for i in range(0, img_width, block_size):
    for j in range(0, img_height, block_size):
        # 计算每个块的坐标
        left = i
        upper = j
        right = min(i + block_size, img_width)
        lower = min(j + block_size, img_height)

        # 截取每个块
        block = img.crop((left, upper, right, lower))

        # 保存每个块为新图像
        block.save(f"block_{block_count}_post_disaster.png")
        block_count += 1