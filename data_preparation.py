import numpy as np
from imageio import imread
from skimage.color import rgb2gray
import os
import pandas as pd

import skimage.filters
from skimage.feature import canny

from skimage.transform import hough_line, hough_line_peaks


def get_only_vertical(lines):
    """
    Это функция для отбора вертикальных прямых - возвращает из списка прямых только те,
    у которых модуль угла наклона в радианах < 0.1
    """
    angles = list(map(lambda x: x[0], lines))
    lines_arr = np.asarray(lines, dtype=object)
    return lines_arr[np.fabs(angles) < 0.1].tolist()


def get_only_horizontal(lines):
    """
    Это функция для отбора горизонтальных прямых - возвращает из списка прямых только те,
    у которых модуль угла наклона в радианах > 0.9
    """
    angles = list(map(lambda x: x[0], lines))
    lines_arr = np.asarray(lines, dtype=object)
    return lines_arr[np.fabs(angles) > 0.9].tolist()


def max_difference_vertical(vertical_lines, same_angle_th=0.02):
    """
    Находит максимальное расстояние между вертикальными прямыми, которые параллельны (с некоторой точностью)
    """
    sorted_lines = np.array(sorted(vertical_lines))
    max_diff = 0
    for i in range(len(sorted_lines)):
        same_a = same_b = i
        while same_a > 0 and abs(sorted_lines[i][0] - sorted_lines[same_a][0]) < same_angle_th:
            same_a -= 1
        while same_b < len(sorted_lines) - 1 and abs(sorted_lines[i][0] - sorted_lines[same_b][0]) < same_angle_th:
            same_b += 1
        if same_a == same_b:
            continue
        diff = abs(max(sorted_lines[same_a:same_b, 1]) - min(sorted_lines[same_a:same_b, 1]))
        max_diff = max(diff, max_diff)
    return max_diff


def highest_horizontal(horizontal_lines):
    """
    Находит самую верхнюю горзонтальную прямую
    """
    if not len(horizontal_lines):
        return 0, 0
    y0 = np.array(list(map(lambda x: x[1] / np.sin(x[0]), horizontal_lines)))
    min_idx = np.argmin(y0)
    return y0[min_idx], horizontal_lines[min_idx][0]


threshold_methods = [skimage.filters.threshold_otsu,
                     skimage.filters.threshold_yen,
                     skimage.filters.threshold_isodata,
                     skimage.filters.threshold_li,
                     skimage.filters.threshold_mean,
                     skimage.filters.threshold_minimum,
                     skimage.filters.threshold_mean,
                     skimage.filters.threshold_triangle,
                     ]


def find_area_of_box(gray_image, same_angle_th=0.02):
    """
    Функция для поиска максимальной высоты между параллельными горизонатальными прямыми, угла наклона
    этих кривых и площади темной поверхности, найденной с помощью threshold с максимальным значением
    из всех возвращаемых методами
    """
    results = np.asarray([f(gray_image) for f in threshold_methods])
    threshold = np.max(results)
    thresholded_img = gray_image >= threshold
    lines = get_lines_hough(thresholded_img)
    hor_lines = get_only_horizontal(lines)
    sorted_lines = np.array(sorted(hor_lines))
    max_diff = 0
    angle = 0
    for i in range(len(sorted_lines)):
        same_a = same_b = i
        while same_a > 0 and abs(sorted_lines[i][0] - sorted_lines[same_a][0]) < same_angle_th:
            same_a -= 1
        while same_b < len(sorted_lines) - 1 and abs(sorted_lines[i][0] - sorted_lines[same_b][0]) < same_angle_th:
            same_b += 1
        if same_a == same_b:
            continue
        diff = abs(max(sorted_lines[same_a:same_b, 1]) - min(sorted_lines[same_a:same_b, 1]))
        max_diff = max(diff, max_diff)
        if diff == max_diff:
            angle = sorted_lines[i, 0]
    return max_diff, angle, sum((gray_image >= threshold).flatten())


def get_lines_hough(gray_image):
    h, theta, d = hough_line(canny(gray_image))
    _, angle, dist = hough_line_peaks(h, theta, d)
    return sorted(list(zip(angle, dist)))


def get_lines_hough_thresholded(gray_image, threshold):
    return get_lines_hough(gray_image >= threshold)


def get_lines_hough_threshold_yen(gray_image):
    return get_lines_hough_thresholded(gray_image, skimage.filters.threshold_yen(gray_image))


def get_values_for_image(image, is_pos):
    """
    Подсчёт всех значений для одного изображения
    """
    gray_image = rgb2gray(image)
    lines = get_lines_hough_threshold_yen(gray_image)
    horizontal_lines = get_only_horizontal(lines)
    vertical_lines = get_only_vertical(lines)
    parallel_max_difference_vert = max_difference_vertical(vertical_lines)
    top_hor_line_pos, top_hor_line_angle = highest_horizontal(horizontal_lines)
    box_height, box_angle, box_area = find_area_of_box(gray_image)
    vert_lines_number = len(vertical_lines)
    hor_lines_number = len(horizontal_lines)
    return (vert_lines_number, hor_lines_number, parallel_max_difference_vert,
            top_hor_line_pos, top_hor_line_angle, box_height, box_angle, box_area, int(is_pos))


columns_names = ['vert_lines', 'hor_lines', 'max_parallel_vert', 'top_hor_line_pos',
                 'top_hor_line_angle', 'box_height', 'box_angle', 'box_area', 'target']


def create_dataset(path_to_images):
    # меняем имена директорий в случае необходимости
    train_pos_dir = os.path.join(path_to_images, 'train/positive/')
    train_neg_dir = os.path.join(path_to_images, 'train/negative/')

    test_pos_dir = os.path.join(path_to_images, 'test/positive/')
    test_neg_dir = os.path.join(path_to_images, 'test/negative/')

    # соберём информацию для обучения
    # сначала положительные примеры
    train_data = []
    os.chdir(train_pos_dir)
    for image_name in os.listdir(train_pos_dir):
        if image_name != '.DS_Store':
            img = imread(image_name)
            res_tuple = get_values_for_image(img, True)
            train_data.append(res_tuple)
    os.chdir(train_neg_dir)
    # теперь отрицательные примеры
    for image_name in os.listdir(train_neg_dir):
        if image_name != '.DS_Store':
            img = imread(image_name)
            res_tuple = get_values_for_image(img, False)
            train_data.append(res_tuple)
    train_df = pd.DataFrame.from_records(train_data, columns=columns_names)

    # соберём информацию для тестирования
    test_data = []
    os.chdir(test_pos_dir)
    for image_name in os.listdir(test_pos_dir):
        if image_name != '.DS_Store':
            img = imread(image_name)
            res_tuple = get_values_for_image(img, True)
            test_data.append(res_tuple)
    os.chdir(test_neg_dir)
    for image_name in os.listdir(test_neg_dir):
        if image_name != '.DS_Store':
            img = imread(image_name)
            res_tuple = get_values_for_image(img, False)
            test_data.append(res_tuple)
    test_df = pd.DataFrame.from_records(test_data, columns=columns_names)
    return train_df, test_df


if __name__ == '__main__':
    # путь к данным для обучения - можно изменить для повторного запуска выделения фич
    path_to_images = '/Users/evgenia/Desktop/DSET'
    train_df, test_df = create_dataset(path_to_images)

    # сохранение датасетов в csv файлы для дальнешего использования
    train_df.to_csv('train_df.csv')
    test_df.to_csv('test_df.csv')
