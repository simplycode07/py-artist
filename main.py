import sys
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

# if len(sys.argv) > 1:
#     original_image = cv2.imread(sys.argv[1])
# else:
#     original_image = cv2.imread("input.jpg")

if len(sys.argv) <= 1:
    print("Not enough parameters provided")
    print("parameters: filename block_size effective_area strength_multiplier")
    sys.exit()

image_path = Path(sys.argv[1])
block_size = int(sys.argv[2])
effective_area = int(sys.argv[3])
strength_multiplier = float(sys.argv[4])
string = "e" if len(sys.argv) < 5 else sys.argv[5]


def signof(i):
    return 1 if i > 0 else -1


def dis(i, j):
    if i == 0 and j == 0:
        return 0
    return signof(i) / (j**2 + i**2)


def clamp(start: int, value: int, end: int) -> int:
    if value < start:
        return start
    if value > end:
        return end

    return value


def convert_image_to_art(original_image):
    # brightness = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    brightness = original_image
    height, width, _ = brightness.shape
    brightness_arr = np.array(brightness)
    img_array = np.zeros(
        (height//block_size, width//block_size, 3), dtype=np.uint8)

    img_array_bw = np.zeros(
        (height//block_size, width//block_size), dtype=np.int8)
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            y_end = min(y + block_size, height)
            x_end = min(x + block_size, width)

            block = brightness_arr[y:y_end, x:x_end]
            avg = block.mean(axis=(0, 1)).astype(np.uint8)

            img_array[y//block_size - 1, x//block_size - 1] = avg[::-1]
            img_array_bw[y//block_size - 1, x//block_size - 1] = int(np.mean(avg)) - 128

    text_positions = []

    result = Image.new("RGB", (width, height), (255, 255, 255))
    pixels_done = 0
    for pix_j in range(0, len(img_array)):
        for pix_i in range(0, len(img_array[0])):

            # uncomment this if you dont want text everywhere
            # if img_array[pix_j, pix_i] > 140:
            #     continue

            text_pos_x = block_size * pix_i + block_size // 2
            text_pos_y = block_size * pix_j + block_size // 2

            original_text_pos = (text_pos_x, text_pos_y)
            change_pos_x = 0.0
            change_pos_y = 0.0
            for j in range(-effective_area, effective_area):
                for i in range(-effective_area, effective_area):
                    clamped_pix_i = clamp(0, pix_i + i, len(img_array[0]) - 1)
                    clamped_pix_j = clamp(0, pix_j + j, len(img_array) - 1)

                    if clamped_pix_i == pix_i and clamped_pix_j == pix_j:
                        continue

                    brightness_pix = (
                        int(np.mean(img_array[clamped_pix_j, clamped_pix_i])) - 128) / 128
                    change_pos_x += brightness_pix * \
                        dis(i, j) * strength_multiplier
                    change_pos_y += brightness_pix * \
                        dis(j, i) * strength_multiplier

                    # change_pos_x += img_array_bw[clamped_pix_j, clamped_pix_i] * \
                    #     dis(i, j) * strength_multiplier
                    # change_pos_y += img_array_bw[clamped_pix_j, clamped_pix_i] * \
                    #     dis(j, i) * strength_multiplier

            text_pos_x += round(change_pos_x)
            text_pos_y += round(change_pos_y)

            # text_pos_x = clamp(0, text_pos_x, width)
            # text_pos_y = clamp(0, text_pos_y, height)

            text_positions.append([string[pixels_done % len(string)], text_pos_x, text_pos_y, tuple(img_array[pix_j, pix_i])])
            pixels_done += 1

    idraw = ImageDraw.Draw(result)

    for chr, x, y, color in text_positions:
        # print(color, type(color))
        idraw.text((x, y), chr, fill=color)

    # result.save(f'{sys.argv[1]} bl_size:{block_size} eff_area:{effective_area} stren_mul:{strength_multiplier}.png')
    return result


if __name__ == "__main__":
    if image_path.suffix in [".mp4", ".mov"]:
        original_video = cv2.VideoCapture(str(image_path))
        fps = int(original_video.get(cv2.CAP_PROP_FPS))
        width = int(original_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(original_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(original_video.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # You can also try 'XVID'
        out = cv2.VideoWriter(
            f'{sys.argv[1]} bl_size:{block_size} eff_area:{effective_area} stren_mul:{strength_multiplier}.mp4', fourcc, fps, (width, height))

        frame_count = 0

        print(f"Processing {total_frames} frames...")
        print(f"Video properties: {width}x{height} @ {fps}fps")

        while True:
            ret, frame = original_video.read()
            if not ret:
                break

            converted_frame = np.array(convert_image_to_art(frame))
            converted_frame_bgr = cv2.cvtColor(converted_frame, cv2.COLOR_RGB2BGR)
            out.write(converted_frame_bgr)
            frame_count += 1

            sys.stdout.write("\r")
            sys.stdout.write(
                f"[{'#' * int(frame_count / total_frames * 20):.<20}] {int(frame_count / total_frames * 100)}%")
            sys.stdout.flush()

        print()

        original_video.release()
        out.release()
        cv2.destroyAllWindows()
        print(
            f'bl_size:{block_size} eff_area:{effective_area} stren_mul:{strength_multiplier}')

    else:
        converted_image = convert_image_to_art(cv2.imread(str(image_path)))
        print(
            f'bl_size:{block_size} eff_area:{effective_area} stren_mul:{strength_multiplier}')
        converted_image.save(
            f'{sys.argv[1]} bl_size:{block_size} eff_area:{effective_area} stren_mul:{strength_multiplier}.png')
