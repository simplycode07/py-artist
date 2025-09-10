import sys
import cv2
import numpy as np
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path
from PIL import Image, ImageDraw

if len(sys.argv) <= 1:
    print("Not enough parameters provided")
    print("parameters: filename block_size effective_area strength_multiplier")
    sys.exit()

image_path = Path(sys.argv[1])
block_size_glob = int(sys.argv[2])
effective_area_glob = int(sys.argv[3])
strength_multiplier_glob = float(sys.argv[4])
string_glob = "e" if len(sys.argv) < 6 else sys.argv[5]


def signof(i):
    return 1 if i > 0 else -1

def dis(i, j):
    if i == 0 and j == 0:
        return 0
    return signof(i) / (j**2 + i**2)


def clamp(start: int, value: int, end: int) -> int:
    value = max(value, start)
    value = min(value, end)

    return value


def calculate_text_position(img_array, img_array_bw, pix_i, pix_j):
    strength_multiplier = strength_multiplier_glob
    effective_area = effective_area_glob
    text_pos_x = block_size * pix_i + block_size / 2
    text_pos_y = block_size * pix_j + block_size / 2

    change_pos_x = 0.0
    change_pos_y = 0.0
    for j in range(-effective_area, effective_area + 1):
        for i in range(-effective_area, effective_area + 1):
            clamped_pix_i = clamp(0, pix_i + i, len(img_array[0]) - 1)
            clamped_pix_j = clamp(0, pix_j + j, len(img_array) - 1)

            if clamped_pix_i == pix_i and clamped_pix_j == pix_j:
                continue

            change_pos_x += img_array_bw[clamped_pix_j, clamped_pix_i] * \
                dis(i, j) * strength_multiplier / 128
            change_pos_y += img_array_bw[clamped_pix_j, clamped_pix_i] * \
                dis(j, i) * strength_multiplier / 128

    text_pos_x += (change_pos_x)
    text_pos_y += (change_pos_y)

    return text_pos_x, text_pos_y

def worker(args):
    img_array, img_array_bw, pix_i, pix_j = args
    return (pix_j, pix_i), calculate_text_position(img_array, img_array_bw, pix_i, pix_j)

def parallel_positions(img_array, img_array_bw):
    tasks = [(img_array, img_array_bw, i, j) 
             for j in range(len(img_array)) 
             for i in range(len(img_array[0]))]

    with Pool(cpu_count()) as pool:
        results = pool.map(worker, tasks)

    return results

def convert_image_to_art(original_image):
    # brightness = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    block_size = block_size_glob
    # effective_area = effective_area_glob

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
            img_array_bw[y//block_size - 1, x //
                         block_size - 1] = int(np.mean(avg)) - 128


    result = Image.new("RGB", (width, height), (255, 255, 255))
    pixels_done = 0

    string = string_glob

    calculated_text_positions = parallel_positions(img_array, img_array_bw)
    idraw = ImageDraw.Draw(result)

    for position, text_position in calculated_text_positions:
        idraw.text(text_position, string[pixels_done % len(string)], fill=tuple(img_array[*position]))
        pixels_done += 1

    return result


if __name__ == "__main__":
    block_size = block_size_glob
    effective_area = effective_area_glob

    if image_path.suffix in [".mp4", ".mov"]:
        original_video = cv2.VideoCapture(str(image_path))
        fps = int(original_video.get(cv2.CAP_PROP_FPS))
        width = int(original_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(original_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(original_video.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # You can also try 'XVID'
        out = cv2.VideoWriter(
            f'{sys.argv[1]} bl_size:{block_size} eff_area:{effective_area} stren_mul:{strength_multiplier_glob}.mp4', fourcc, fps, (width, height))

        frame_count = 0

        print(f"Processing {total_frames} frames...")
        print(f"Video properties: {width}x{height} @ {fps}fps")

        while True:
            ret, frame = original_video.read()
            if not ret:
                break

            converted_frame = np.array(convert_image_to_art(frame))
            converted_frame_bgr = cv2.cvtColor(
                converted_frame, cv2.COLOR_RGB2BGR)
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
            f'bl_size:{block_size} eff_area:{effective_area} stren_mul:{strength_multiplier_glob}')

    else:
        converted_image = convert_image_to_art(cv2.imread(str(image_path)))
        print(
            f'bl_size:{block_size} eff_area:{effective_area} stren_mul:{strength_multiplier_glob}')
        converted_image.save(
            f'{sys.argv[1]} bl_size:{block_size} eff_area:{effective_area} stren_mul:{strength_multiplier_glob}.png')
