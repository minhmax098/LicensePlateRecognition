import numpy as np
from skimage.transform import resize
from skimage import measure
from skimage.measure import regionprops
# import matplotlib.patches as patches
# import matplotlib.pyplot as plt
import DetectPlate2
import cv2
import imutils


def segmentCharacters(plate_like_objects, roiPlate):
    plate_like_objects1 = plate_like_objects[0].copy()
    license_plate = np.invert(plate_like_objects1)
    labelled_plate = measure.label(license_plate)
    character_dimensions = (0.3 * license_plate.shape[0], 0.60 * license_plate.shape[0], 0.07 * license_plate.shape[1],
                            0.15 * license_plate.shape[1])
    min_height, max_height, min_width, max_width = character_dimensions

    characters = []
    column_list = []
    for regions in regionprops(labelled_plate):
        y0, x0, y1, x1 = regions.bbox
        region_height = y1 - y0
        region_width = x1 - x0

        if region_height > min_height and region_height < max_height and region_width > min_width and region_width < max_width:
            roi = license_plate[y0:y1, x0:x1]

            # draw a red bordered rectangle over the character: vẽ một hình chữ nhật viền đỏ trên kí tự
            cv2.rectangle(roiPlate, (x0,y0), (x1, y1), (0,0,255), thickness= 1)

            #  resize the characters to 20X20 and then append each character into the characters list
            # thay đổi kích thước thành 20*20 và sau đó nối từng kí tự vào danh sách các kí tự
            resized_char = resize(roi, (20, 20))
            characters.append(resized_char)

            # this is just to keep track of the arrangement of the characters:
            # đây chỉ để theo dõi sự sắp xếp các kí tự
            column_list.append(x0)
    return  roiPlate, characters
# The invert was done so as to convert the black pixel to white pixel and vice versa
# việc đảo ngược được thực hiện để chuyển đổi pixel đen thành pixel trắng và ngược lại
# license_plate = np.invert(DetectPlate.plate_like_objects[0])
#
# labelled_plate = measure.label(license_plate)
#
# fig, ax1 = plt.subplots(1)
# ax1.imshow(license_plate, cmap="gray")
# # the next two lines is based on the assumptions that the width of :hai dòng tiếp theo giả định rằng chiều rộng của một biển số xe
# # a license plate should be between 5% and 15% of the license plate, : nằm trong khoảng 5-15% biển số xe, và chiều cao nằm trong khoảng
# # and height should be between 35% and 60% : 35 đến 60%
# # this will eliminate some : điều này sẽ loại bỏ một số giả định
# character_dimensions = (0.35*license_plate.shape[0], 0.60*license_plate.shape[0], 0.05*license_plate.shape[1], 0.15*license_plate.shape[1])
# min_height, max_height, min_width, max_width = character_dimensions
#
# characters = []
# counter=0
# column_list = []
# for regions in regionprops(labelled_plate):
#     y0, x0, y1, x1 = regions.bbox
#     region_height = y1 - y0
#     region_width = x1 - x0
#
#     if region_height > min_height and region_height < max_height and region_width > min_width and region_width < max_width:
#         roi = license_plate[y0:y1, x0:x1]
#
#         # draw a red bordered rectangle over the character: vẽ một hình chữ nhật viền đỏ trên kí tự
#         rect_border = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor="red",
#                                        linewidth=2, fill=False)
#         ax1.add_patch(rect_border)
#
#         #  resize the characters to 20X20 and then append each character into the characters list
#         # thay đổi kích thước thành 20*20 và sau đó nối từng kí tự vào danh sách các kí tự
#         resized_char = resize(roi, (20, 20))
#         characters.append(resized_char)
#
#         # this is just to keep track of the arrangement of the characters:
#         # đây chỉ để theo dõi sự sắp xếp các kí tự
#         column_list.append(x0)
# # print(characters)
# plt.show()
def segmentPlate2(roi):
    characters = []
    column_list = []
    mser = cv2.MSER_create()
    area = roi.shape[0]* roi.shape[1]
    # Resize the image so that MSER can work better
    img = cv2.resize(roi, (roi.shape[1] * 2, roi.shape[0] * 2))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11,2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    Mor = cv2.morphologyEx(Threshold, cv2.MORPH_CLOSE, kernel)
    vis = img.copy()

    regions = mser.detectRegions(Mor)
    for p in regions[0]:
        if cv2.contourArea(p) > 50:
            [x,y,w,h] = cv2.boundingRect(p)
            cv2.rectangle(vis, (x,y), (x+w, y+h), color=(0,0,255), thickness=2)
            roiCharacter = Mor[y:y+h, x:x+w]/255
            roiCharacter = resize(roiCharacter, (20,20))
            characters.append(roiCharacter)
            column_list.append(x)

    # hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions[0]]
    # cv2.polylines(vis, hulls, 1, (0, 255, 0))
    return vis, characters, column_list