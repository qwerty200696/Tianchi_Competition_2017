import cv2
from skimage import morphology, measure
import tifffile as tiff

FILE_INPUT = "E:/Tianchi/NEW_DATA2/224_rgb/pinjie_new.tif"
FILE_OUTPUT = "E:/Tianchi/NEW_DATA2/224_rgb/pinjie_new_0.3_250_255.tif"

img = cv2.imread(FILE_INPUT)
img = img > 0
img_new = morphology.remove_small_objects(img, min_size=250, connectivity=1)

labels = measure.label(img_new, connectivity=1)
regions = measure.regionprops(labels)

for region in regions:
    if region.extent<0.3:
        for i in region.coords:
            img_new[i[0], i[1], i[2]] = False

    width = region.bbox[3] - region.bbox[0]
    height = region.bbox[4] - region.bbox[1]
    if  height != 0 and width != 0:
        if width / height > 4 and height<40:
            for i in region.coords:
                img_new[i[0], i[1], i[2]] = False

        if width / height < 0.25 and width<40:
            for i in region.coords:
                img_new[i[0], i[1], i[2]] = False


img_new = img_new.astype('uint8')
img_new[img_new > 0] = 255
tiff.imsave(FILE_OUTPUT,img_new[:,:,0])