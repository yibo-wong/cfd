from PIL import Image
import os

# 读取图像文件
image_folder = './fig/lax/gif/'  # 图像文件夹路径
image_files = sorted([img for img in os.listdir(
    image_folder) if img.endswith('.png')])

# 打开图像并存储在列表中
images = [Image.open(os.path.join(image_folder, img)) for img in image_files]

# 保存为GIF
gif_path = './fig/lax/gif/output.gif'
images[0].save(gif_path, save_all=True, append_images=images[1:],
               optimize=False, duration=100, loop=0)

print(f"GIF saved as {gif_path}")
