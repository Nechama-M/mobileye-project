import os
from random import randint
import numpy as np
from PIL import Image, ImageOps


def read_images(path):
    image = Image.open(path)
    label_path = path.replace("leftImg8bit", "gtFine", 1).replace("_leftImg8bit", "_gtFine_labelIds", 1)
    label_image = np.array(Image.open(label_path), dtype='uint8')

    return image, label_image


def find_pixels_that_are_19(label_image):
    width, height = label_image.shape
    pixels_that_are_19 = []
    i = 0
    while i < width:
        j = 0
        while j < height:
            if label_image[i][j] == 19:
                pixels_that_are_19.append((i, j))
            j += 1
        i += 1

    return pixels_that_are_19


def filter_single_pixel_for_each_tfl(pixels_that_are_19):
    tfl = [pixels_that_are_19[0]]
    for i in pixels_that_are_19:
        if (i[0] - tfl[-1][0] > 81) or i[1] - tfl[-1][1] > 81:
            tfl.append(i)

    return tfl


def crop_image_around_coordinate(image, coordinate):
    width, height = image.size
    if coordinate[1] > 40 and coordinate[0] > 30 and coordinate[1] + 41 < width and coordinate[0] + 51 < height:
        cropped_image = image.crop((round(coordinate[1]) - 40, round(coordinate[0]) - 30, round(coordinate[1]) + 41, round(coordinate[0]) + 51))
    else:
        bordered_img = ImageOps.expand(image, border=51, fill='white')
        cropped_image = bordered_img.crop(((round(coordinate[1]) + 51) - 40, (round(coordinate[0]) + 51) - 30, round(coordinate[1]) + 41, round(coordinate[0]) + 51))

    return cropped_image


def get_non_tfl_coordinates(image):
    width, height = image.shape
    while True:
        x = randint(0, width)
        y = randint(0, height)
        tmp_cropped_image = crop_image_around_coordinate(Image.fromarray(image), (x, y))
        tmp_list = find_pixels_that_are_19(np.array(tmp_cropped_image, dtype='uint8'))
        if not tmp_list:
            return x, y


def write_to_file(file_name, obj):
    with open(file_name, "a") as file:
        file.write(obj)


def save_to_binary_file(cropped_tfl_image, cropped_non_tfl_image, dir):
    np.ndarray.tofile(np.asarray(cropped_tfl_image), f"data_dir\\{dir}\\data.bin")
    write_to_file(f"data_dir\\{dir}\\labels.bin", "1")
    np.ndarray.tofile(np.asarray(cropped_non_tfl_image), f"data_dir\\{dir}\\data.bin")
    write_to_file(f"data_dir\\{dir}\\labels.bin", "0")


def init_data_set():
    path = "images\\leftImg8bit"
    for root, dirs, files in os.walk(path):
        for file in files:
            path = os.path.join(root, file)
            print(path)
            dir_ = path[19: path.find("\\", 19)]

            # read images (label_imaged in uint8 format)
            image, label_image = read_images(path)
            # find pixels that are equal to 19 on labeled images
            pixels_that_are_19 = find_pixels_that_are_19(label_image)
            if pixels_that_are_19:
                # filter one pixel per traffic light
                tfl = filter_single_pixel_for_each_tfl(pixels_that_are_19)
                # crop around each tfl
                for coordinate in tfl:
                    cropped_tfl_image = crop_image_around_coordinate(image, coordinate)
                    coordinate_non_tfl = get_non_tfl_coordinates(label_image)
                    cropped_non_tfl_image = crop_image_around_coordinate(image, coordinate_non_tfl)
                    save_to_binary_file(cropped_tfl_image, cropped_non_tfl_image, dir_)


if __name__ == '__main__':
    init_data_set()