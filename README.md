# pyar-tist


this script achieves stippling like effect on any input image, but instead of dots, it places any desired text.  
this was inspired by a random digital art post online which had somewhat the exact same effect,  
i knew we can replicate it using code so i made this in a few hours. (yes i have lots of free time to code and work on random projects.)

## input and result:

![comparison_image](comparison.png)

<img width="500" height="512" alt="image" src="https://github.com/user-attachments/assets/6913acfb-174c-46f6-a3f9-0ee12e6ec792" />

### try this version if you want to see the intermediate image processing steps saved as images
```py
#!/usr/local/bin/python3
from PIL import Image, ImageDraw
import numpy as np
import cv2
import random



img = cv2.imread('p.jpeg')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
block_size = 32
height, width = gray_img.shape
img_array = np.array(gray_img)

for y in range(0, height, block_size):
    for x in range(0, width, block_size):
        if y + block_size > height:
            y_end = height
        else:
            y_end = y + block_size

        if x + block_size > width:
            x_end = width
        else:
            x_end = x + block_size
        block = img_array[y:y_end, x:x_end]
        avg = int(np.mean(block))
        img_array[y:y_end, x:x_end] = avg


result = Image.fromarray(img_array)
result.save('result.png')
img_gray = cv2.imread('result.png')
ret, thresh_img = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY)
cv2.imwrite('bin.png', thresh_img)

imgtowrite = Image.open('bin.png').convert("L")
idraw = ImageDraw.Draw(imgtowrite)
text_positions = []
for y in range(0, height, block_size):
    for x in range(0, width, block_size):

        y_end = min(y + block_size, height)
        x_end = min(x + block_size, width)

        coords = [
            (xi, yi)
            for yi in range(y, y_end)
            for xi in range(x, x_end)
        ]
        if coords:
            pos = random.choice(coords)
            pixel_value = imgtowrite.getpixel(pos)

            if pixel_value == 255:
                continue
            else: 
                idraw.rectangle([(x, y), (x_end-1, y_end-1)], fill=255)
                text_positions.append(pos)

for pos in text_positions:
    idraw.text(pos, "hello world.", fill=0)
imgtowrite.save('final_result.png')
```


