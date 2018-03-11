import os
from skimage import exposure

from PIL import Image
from PIL import ImageEnhance


FILE_DIR = 'E:/TC/fusai/500_cut/2017/'
FILE_DIR_OUTPUT = 'E:/TC/fusai/500_cut/2017_enhance/'


imgs = os.listdir(FILE_DIR)

for img in imgs:
    image = Image.open(FILE_DIR+img)
    brightness = [0.7, 1.3]
    contrast = [0.7, 1.3]


    for bright in brightness:
        for cont in contrast:
            # gamma调整
            # img_gam = exposure.adjust_gamma(image, 1.0)

            # 亮度调整
            enh_bri = ImageEnhance.Brightness(image)
            image_brightened = enh_bri.enhance(bright)

            # 对比度调整
            enh_con = ImageEnhance.Contrast(image_brightened)
            image_contrasted = enh_con.enhance(cont)

            # 保存
            save_name = img.split('.')[0] + '_'  + str(bright) + '_' + str(cont) + '_' + '.png'
            image_contrasted.save(FILE_DIR_OUTPUT + save_name)

# # 亮度增强
# enh_bri = ImageEnhance.Brightness(image)
# image_brightened = enh_bri.enhance(brightness)
# image_brightened.show()

# # 色度增强
# enh_col = ImageEnhance.Color(image)
# color = 2
# image_colored = enh_col.enhance(color)
# image_colored.show()

# # 对比度增强
# enh_con = ImageEnhance.Contrast(image)
# image_contrasted = enh_con.enhance(contrast)
# image_contrasted.show()

# # 锐度增强
# enh_sha = ImageEnhance.Sharpness(image)
# sharpness = 3.0
# image_sharped = enh_sha.enhance(sharpness)
# image_sharped.show()