from train import train_model
from predict import predict_disease


def print_menu():
    print("============= 메뉴 =============")
    print("1. 인공지능 모델 학습시키기")
    print("2. 흉부 X-ray 사진으로 질병 알아내기")
    print("3. 종료하기")
    print("===============================")


while True:
    print_menu()
    menu = input("입력 >> ")
    if menu == "1":
        train_model()
    elif menu == "2":
        predict_disease()
    elif menu == "3":
        break
    else:
        print("올바른 값을 입력해 주세요.")