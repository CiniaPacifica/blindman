import os
import numpy as np
import tensorflow as tf
import cv2
import winsound
from keras import models
from imutils.video import WebcamVideoStream  # Webcam영상 스트림을 위해 import

testPath = r'C:\Users\User\Desktop\Blind Man\test'
inputShare = (50, 66, 3)
numClass = 4



def read_image(path):
    gfile = tf.io.read_file(path)
    image = tf.io.decode_image(gfile, dtype=tf.float32)
    return image

# 모델 불러오기
model = models.load_model('Blind_Class.h5')

# 비프음 함수
def sound(frq, dur):
    winsound.Beep(frq, dur)



host = "{}:4747/video".format("http://192.168.251.44") # 유동 IP주소, 바뀔 수 있음
cam = WebcamVideoStream(src=host).start()    # 비디오 스트림 시작

while True:
    frame = cam.read()
    if cv2.waitKey(1) == ord('q'):
        break
    else:
        cv2.imshow("VideoFrame", frame)

        save = cv2.imwrite(testPath + "/temp.jpg", frame, params= None)
        img = read_image(testPath + "/temp.jpg")
        img = tf.image.resize(img, inputShare[:2])

        image = np.array(img)
        # print(image.shape)
        # plt.imshow(image)
        # plt.title('Check the Image and Predict Together!')
        # plt.show()

        testImage = image[tf.newaxis, ...]
        pred = model.predict(testImage)
        #print(pred)
        num = np.argmax(pred)

        if num == 0:    # 라인이 보이지 않음
            print("현관입니다.") 
            sound(1000, 50)

        elif num == 1:  # 라인이 영상 가운데에 보임
            print("계단입니다.") 
            sound(500, 70)

        elif num == 2:  # 라인이 왼쪽에 보임
            print("교실입니다.") 
            sound(1500, 50)

        elif num == 3:  # 라인이 오른쪽에 보임
            print("쉼터입니다.")
            sound(2000, 50) 
        else:
            pass
            
cv2.destroyAllWindows()
os._exit(True)