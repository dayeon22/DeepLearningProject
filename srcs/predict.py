from keras.models import load_model
import sys
import os
from data_reader import img_convert
import numpy as np
import openpyxl


def predict_disease():
    label = ["covid", "normal", "pneumonia", "tuberculosis"]  # 코로나, 일반, 폐렴, 결핵
    test_images = []
    test_filenames = []

    try:
        model = load_model('../results/model.h5')
    except IOError:
        print("모델이 존재하지 않습니다. 먼저 학습을 진행해 주세요", file=sys.stderr)
        return

    for dirs in os.listdir("../photos"):
        if os.path.isdir("../photos" + "/" + dirs):
            for file in os.listdir("../photos" + "/" + dirs):
                path = "../photos" + "/" + dirs + "/" + file
                test_filenames.append(path)
                test_images.append(np.asarray(img_convert(path)))
        elif os.path.isfile("../photos" + "/" + dirs):
            path = "../photos" + "/" + dirs
            test_filenames.append(path)
            test_images.append(np.asarray(img_convert(path)))

    if len(test_filenames) == 0:
        print("photos 디렉토리 안에 X-ray 사진을 넣어주세요!")
        return

    test_images = np.asarray(test_images) / 255.0
    test_images = test_images.reshape(test_images.shape[0], 256, 256, 1)
    predictions = model.predict(test_images)


    # 표 출력
    print("\n\n============================================================================")
    print("%-50s| %7s | %10s" %("파일명", "확률", "결과"))
    print("============================================================================")
    for i in range(len(predictions)):
        print("%-50s | %7.3f%% | %10s" %(test_filenames[i], np.max(predictions[i]) * 100, label[np.argmax(predictions[i])]))
    print("============================================================================\n\n")

    # 엑셀 저장
    excel = openpyxl.Workbook()
    sheet = excel.active
    sheet.append(["파일명", "코로나19", "일반", "폐렴", "결핵", "결과"])
    for i in range(len(predictions)):
        sheet.append([test_filenames[i], predictions[i][0] * 100, predictions[i][1] * 100,
                      predictions[i][2] * 100, predictions[i][3] * 100, label[np.argmax(predictions[i])]])
    excel.save("../results/result.xlsx")
