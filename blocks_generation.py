from PIL import Image

img = Image.open("img.jpg")

img_width, img_height = img.size

block_size = 1024

block_count = 1
for i in range(0, img_width, block_size):
    for j in range(0, img_height, block_size):
        left = i
        upper = j
        right = min(i + block_size, img_width)
        lower = min(j + block_size, img_height)
        block = img.crop((left, upper, right, lower))
        block.save(f"block_{block_count}_post_disaster.png")
        block_count += 1