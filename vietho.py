import random
from PIL import Image
import cv2
import numpy as np
from shutil import copyfile
import glob
import shutil
import os


def randomNoise(aug_img, param, param1):
    pass


def copy_org(image_org_filepath, param):
    pass


def aug_skew(aug_id, aug_num, org_image, image_org_filepath, image_aug_path):
    idx = 0
    # Copy image
    copy_org(image_org_filepath, image_aug_path + os.path.basename(image_org_filepath).split(".")[0] + ".jpg")

    # Copy label
    image_org_filename = os.path.basename(image_org_filepath).split(".")[0]

    if 1 == 1:  # random.random()>0.9:
        print("Aug")
        while idx < aug_num:
            value = 8
            # ----------------- Wrap image
            # Build org_corner
            org_corner = np.float32([[0, 0], [0, org_image.shape[0]], [org_image.shape[1], org_image.shape[0]],
                                     [org_image.shape[1], 0]])  # 4 point

            # Build new_corner
            top_left = [random.uniform(-value, value), random.uniform(-value, value)]
            top_right = [org_image.shape[1] - random.uniform(-value, value), random.uniform(-value, value)]
            bottom_left = [random.uniform(-value, value), org_image.shape[0] - random.uniform(-value, value)]
            bottom_right = [org_image.shape[1] + random.uniform(-value, value),
                            org_image.shape[0] + random.uniform(-value, value)]

            aug_corner = np.float32([top_left, bottom_left, bottom_right, top_right])

            # Wrap image
            M = cv2.getPerspectiveTransform(org_corner, aug_corner)
            aug_img = cv2.warpPerspective(org_image, M, (org_image.shape[1], org_image.shape[0]))

            # Set file name
            image_aug_filename = image_org_filename + "_" + str(aug_id) + "_" + str(idx) + ".jpg"

            # Ghi image aug
            aug_img = randomNoise(aug_img, 15, 10)
            ret, aug_img = cv2.threshold(aug_img, 127, 255, cv2.THRESH_BINARY)

            # aug_img = cv2.resize(aug_img, (256, 32))
            cv2.imwrite(image_aug_path + image_aug_filename, aug_img)
            idx = idx + 1
    return


def data_augment():
    # Em thay duong dan den ki tu hien tai vao day
    org_path = "D:\Đồ án HTN\LicensePlateDetector-master\train20X20\1"
    image_org_path = org_path + "D:\Đồ án HTN\LicensePlateDetector-master\train20X20\1"

    aug_path = "D:\Đồ án HTN\LicensePlateDetector-master\train20X20_aug\1"
    image_aug_path = aug_path + "D:\Đồ án HTN\LicensePlateDetector-master\train20X20_aug\1"

    shutil.rmtree(image_aug_path)

    if not os.path.exists(image_aug_path):
        os.mkdir(image_aug_path)

    count = 0
    # Lap qua folder data
    for img_org_path in glob.iglob(image_org_path + '*.jpg'):
        # print("Aug")
        org_img = cv2.imread(img_org_path, 0)

        # print("Start Skew")
        # 3. Skew
        aug_id = "skew"
        aug_num = 50
        aug_skew(aug_id, aug_num, org_img, img_org_path, image_aug_path)

        count += 1
        print(count)
        # if count==2:
        #   return

    return
#data_augment()
