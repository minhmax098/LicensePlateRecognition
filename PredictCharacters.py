import pickle
print("Loading model")
filename = './finalized_model.sav'
model = pickle.load(open(filename, 'rb'))

print('Model loaded. Predicting characters of number plate')
def predict(characters, column_list):
    classification_result = []
    try:
        for each_character in characters:
            # converts it to a 1D array : chuyển đổi nó thành mảng 1D
            each_character = each_character.reshape(1, -1)

            result = model.predict(each_character)
            classification_result.append(result)

        # print('Classification result:', classification_result)
        plate_string = ''
        for eachPredict in classification_result:
            plate_string += eachPredict[0]

        # print('Predicted license plate: ', plate_string)

        # it's possible the characters are wrongly arranged : có thể các kí tự được sắp xếp sai
        # since that's a possibility, the column_list will be : vì đó có một khả năng, column_list sẽ được sử dụng để sắp xếp các chữ cái theo đúng thứ tự
        # used to sort the letters in the right order

        column_list_copy = column_list[:].copy()
        column_list.sort()
        rightplate_string = ''
        for each in column_list:
            rightplate_string += plate_string[column_list_copy.index(each)]

        print('License plate:', rightplate_string)
        return  rightplate_string
    except:
        pass