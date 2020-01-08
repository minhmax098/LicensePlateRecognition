import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from skimage.io import imread
from skimage.filters import threshold_otsu

# from PredictCharacters import each

letters = [ '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D','E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X', 'Y', 'Z', ]

def read_training_data(training_directory):
    image_data = []
    target_data = []
    for each_letter in letters:  #

            image_path = os.path.join(training_directory, each_letter, each_letter + '_' + str(each) + '.jpg')
            # đọc mỗi ảnh của mỗi kí tự
            img_details = imread(image_path, as_gray = True)
            # chuyển mỗi hình kí tự  qua ảnh nhị phân
            binary_image = img_details < threshold_otsu(img_details)
            # mảng 2D của mỗi ảnh được làm mờ bởi máy học
            # phân loai yêu cầu của mỗi mẫu là mảng 1D
            # ảnh 20*20 trở thành 1*400
            #  trong thuật ngữ máy học có 400 tính năng trong mỗi pixel
            #  đại diện cho mỗi tính năng
            flat_bin_image = binary_image.reshape(-1)
            image_data.append(flat_bin_image) # hàm append nối một đối tượng vào cuối mảng
            target_data.append(each_letter)

    return (np.array(image_data), np.array(target_data))

def cross_validation(model, num_of_fold, train_data, train_label):
    # sử dụng khái niệm xác nhận chéo để đo lường độ chính xác sử dụng khái niệm xác nhận chéo để đo lường độ chính xác
    # nếu num_of_fold là 4, thực hiện xác thực chéo 4 lần
    # nó chia dữ liệu thành 4 phần và sử dụng 1/4 để kiểm tra
    # 3/4 còn lại cho việc training
    accuracy_result = cross_val_score(model, train_data, train_label,
                                      cv = num_of_fold)
    print("Cross Validation Result for ", str(num_of_fold), " -fold")

    print(accuracy_result * 100)


# current_dir = os.path.dirname(os.path.realpath(__file__))
#
# training_dataset_dir = os.path.join(current_dir, 'train')
print('reading data')  # đọc data
training_dataset_dir = './train20X20'
image_data, target_data = read_training_data(training_dataset_dir)
print('reading data completed')


# the probability was set to True so as to show : xác suất được đặt thành True để hiển thị
# how sure the model is of it's prediction : Làm thế nào chắc chắn mô hình là dự đoán của nó
svc_model = SVC(kernel='linear', probability=True, )

cross_validation(svc_model, 4, image_data, target_data)

print('training model')

# let's train the model with all the input data
svc_model.fit(image_data, target_data)

# we will use the joblib module to persist the model : sử dụng model joblib để duy trì mẫu vào file .
# into files. This means that the next time we need to
# predict, we don't need to train the model again
# save_directory = os.path.join(current_dir, 'models/svc/')
# if not os.path.exists(save_directory):
#     os.makedirs(save_directory)
# joblib.dump(svc_model, save_directory+'/svc.pkl')

import pickle
print("model trained.saving model..")
filename = './finalized_model.sav'
pickle.dump(svc_model, open(filename, 'wb'))
print("model saved")