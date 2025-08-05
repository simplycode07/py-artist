import sys
import cv2
from PIL import Image, ImageDraw
import numpy as np

# if len(sys.argv) > 1:
#     original_image = cv2.imread(sys.argv[1])
# else:
#     original_image = cv2.imread("input.jpg")

if len(sys.argv) <= 1:
    print("Not enough parameters provided")
    print("parameters: filename block_size effective_area strength_multiplier")
    sys.exit()

original_image = cv2.imread(sys.argv[1])
block_size = int(sys.argv[2])
effective_area = int(sys.argv[3])
strength_multiplier = float(sys.argv[4])

brightness = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
height, width = brightness.shape
brightness_arr = np.array(brightness)
img_array = np.zeros((height//block_size, width//block_size), dtype=np.uint8)

for y in range(0, height, block_size):
    for x in range(0, width, block_size):
        y_end = min(y + block_size, height)
        x_end = min(x + block_size, width)

        block = brightness_arr[y:y_end, x:x_end]
        avg = int(np.mean(block))

        img_array[y//block_size - 1, x//block_size] = avg


# result = Image.fromarray(img_array)
# result.save('result.png')

# for y in range(0, height, block_size):
#     for x in range(0, width, block_size):

def signof(i):
    return 1 if i > 0 else -1

def dis(i):
    if i == 0:
        return 0
    return signof(i) / (i**2)
        

text_positions = []


def clamp(start:int, value:int, end:int) -> int:
    if value < start:
        return start
    if value > end:
        return end

    return value

result = Image.new("RGB", (width, height), (255, 255, 255))
for pix_j in range(0, len(img_array)):
    for pix_i in range(0, len(img_array[0])):
        if img_array[pix_j, pix_i] > 140:
            continue

        text_pos_x = block_size * pix_i + block_size // 2
        text_pos_y = block_size * pix_j + block_size // 2

        original_text_pos = (text_pos_x, text_pos_y)
        change_pos_x = 0.0
        change_pos_y = 0.0
        for j in range(-effective_area, effective_area):
            for i in range(-effective_area, effective_area):
                clamped_pix_i = clamp(0, pix_i + i, len(img_array[0]) - 1)
                clamped_pix_j = clamp(0, pix_j + j, len(img_array) - 1)
                
                if clamped_pix_i == pix_i and clamped_pix_j == pix_j: continue

                brightness_pix = (int(img_array[clamped_pix_j, clamped_pix_i]) - 128) / 128

                change_pos_x += brightness_pix * dis(i) * strength_multiplier
                change_pos_y += brightness_pix * dis(j) * strength_multiplier

    
        text_pos_x += round(change_pos_x)
        text_pos_y += round(change_pos_y)

        text_pos_x = clamp(0, text_pos_x, width)
        text_pos_y = clamp(0, text_pos_y, height)
        
        text_positions.append((text_pos_x, text_pos_y))

        
# print(text_positions)

idraw = ImageDraw.Draw(result)

for pos in text_positions:
    idraw.text(pos, "e", fill=0)

print(f'bl_size:{block_size} eff_area:{effective_area} stren_mul:{strength_multiplier}')
result.save(f'{sys.argv[1]} bl_size:{block_size} eff_area:{effective_area} stren_mul:{strength_multiplier}.png')


