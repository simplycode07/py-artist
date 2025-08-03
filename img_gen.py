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
ret, thresh_img = cv2.threshold(img_array, 128, 255, cv2.THRESH_BINARY)
imgtowrite = Image.fromarray(thresh_img).convert("L")
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
imgtowrite.save('result.png')
            
