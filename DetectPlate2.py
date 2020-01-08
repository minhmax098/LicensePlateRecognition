from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage import measure
from skimage.measure import regionprops

# import matplotlib.pyplot as plt
import imutils
import cv2

def detectPlate(image):
    image_out = image.copy()
    gray_car_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # chuyển về grayscale để dể dàng xử lí
    threshold_value = threshold_otsu(gray_car_image)
    binary_car_image = gray_car_image > threshold_value
    label_image = measure.label(binary_car_image)
    plate_dimensions = (0.03 * label_image.shape[0], 0.08 * label_image.shape[0], 0.15 * label_image.shape[1], 0.3 * label_image.shape[1])
    plate_dimensions2 = (0.08 * label_image.shape[0], 0.2 * label_image.shape[0], 0.15 * label_image.shape[1], 0.4 * label_image.shape[1])
    min_height, max_height, min_width, max_width = plate_dimensions # kích thước biển số
    plate_objects_cordinates = []    #tọa độ của biển
    plate_like_objects = []
    flag = 0 # gắn cờ bằng 0
    # regionprops creates a list of properties of all the labelled regions
    for region in regionprops(label_image):
        # print(region)
        if region.area < 50:
            # if the region is so small then it's likely not a license plate
            continue
            # the bounding box coordinates
        min_row, min_col, max_row, max_col = region.bbox

        region_height = max_row - min_row
        region_width = max_col - min_col
        if region_height >= min_height and region_height <= max_height and region_width >= min_width and region_width <= max_width and region_width > region_height:
            flag = 1
            plate_like_objects.append(binary_car_image[min_row:max_row,min_col:max_col])

            plate_objects_cordinates.append((min_row, min_col,max_row, max_col))
            image_out = cv2.rectangle(image_out,(min_col,min_row),(max_col,max_row), (0,0,255),1)

    if (flag == 1):
        return image_out, plate_objects_cordinates, plate_like_objects

    if (flag == 0):
        min_height, max_height, min_width, max_width = plate_dimensions2
        plate_objects_cordinates = []
        plate_like_objects = []


        # regionprops creates a list of properties of all the labelled regions
        for region in regionprops(label_image):
            if region.area < 50:
                # if the region is so small then it's likely not a license plate
                continue
                # the bounding box coordinates
            min_row, min_col, max_row, max_col = region.bbox
            region_height = max_row - min_row
            region_width = max_col - min_col
            if region_height >= min_height and region_height <= max_height and region_width >= min_width and region_width <= max_width and region_width > region_height:
                # print("hello")
                plate_like_objects.append(binary_car_image[min_row:max_row,
                                          min_col:max_col])
                plate_objects_cordinates.append((min_row, min_col,
                                                 max_row, max_col))
                image_out = cv2.rectangle(image_out, (min_col, min_row), (max_col, max_row), (0, 0, 255), 1)
        return image_out, plate_objects_cordinates, plate_like_objects


# filename = 'xe09.mp4'
# cap = cv2.VideoCapture(filename)
# # cap = cv2.VideoCapture(0)
# count = 0
# while cap.isOpened():
#     ret, frame = cap.read()
#
#     if ret == True:
#         frame1 = imutils.rotate(frame,270)
#         print((frame1.shape))
#         #frame1 = frame.copy
#         image, plate = detectPlate(frame1)
#         plate = plate[0]
#         print(count, plate)
#         roi = frame1[plate[0]:plate[2], plate[1]:plate[3]]
#         # print(roi.shape)
#         if roi.shape[0] !=0 and roi.shape[1] != 0:
#             cv2.imshow('roi', roi)
#         cv2.imshow('detect Plate', image)
#         count +=2
#         if cv2.waitKey(10) & 0xFF == ord('q'):
#             break
#     else:
#         break
# cap.release()
