# Ceren Korkmaz 21995445

import cv2
import numpy as np
import math



path_txt = r"D:\2022-BAHAR\416\ass1\GT"
# read txt file
with open(path_txt + r"\4.txt", "r") as txt:
    lines = txt.readlines()
circle_count = lines[0]
radius = []
min_r = []
max_r = []
for i in range(1, 1 + int(circle_count)):
    lines[i] = lines[i].split(" ")
    radius.append(int(lines[i][2][:-8]))
    min_r.append(int(lines[i][2][:-8]) - 5)
    max_r.append(int(lines[i][2][:-8]) + 5)

path = r"D:\2022-BAHAR\416\ass1\dataset"
# read image
img_o = cv2.imread(path + r"\4.jpg", cv2.IMREAD_COLOR)

def hough(img_o, r_min, r_max):
    # convert to grayscale
    img_gray = cv2.cvtColor(img_o, cv2.COLOR_BGR2GRAY)
    # blur img
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)

    # Canny Edge Detection
    img_canny = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)

    theta = np.arange(0, 360, 1) * np.pi / 180
    acc = np.zeros_like(img_canny)

    # hough
    img_shape = img_o.shape
    x_max = img_shape[0]
    y_max = img_shape[1]

    r_arr = np.zeros_like(img_canny)
    for r in range(r_min, r_max + 1):
        for x in range(x_max):
            for y in range(y_max):
                for angle in range(0, 360, 10):
                    if img_canny[x][y] == 255:
                        a = int(x - (math.cos(theta[angle]) * r))
                        b = int(y - (math.sin(theta[angle]) * r))
                        if 0 < a < x_max and 0 < b < y_max:
                            acc[a][b] += 1
                            r_arr[a][b] = r

    result = np.where(acc == np.max(acc))
    res_r = int(r_arr[int(result[0][0]), int(result[1][0])])

    image = cv2.circle(img_o, (int(result[1][0]), int(result[0][0])), res_r, (255, 0, 0), 2)
    return image


img = hough(img_o, min_r[0], max_r[0])
for circle in range(1, int(circle_count)):
    img = hough(img, min_r[circle], max_r[circle])
cv2.imshow("Final Result", img)
cv2.waitKey(0)
