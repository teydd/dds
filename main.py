import cv2
import pygame
from cvzone.FaceMeshModule import FaceMeshDetector  # Finds face landmarks in BGR Image:param img: Image to find the face landmarks in.

#initializing pygame
pygame.init()

# Loading the audio file
audio_path = r'C:\Users\squidward\New folder\free-sound-effects-ALARM_BE.mp3'
pygame.mixer.music.load(audio_path)

video = cv2.VideoCapture(0)  # captures live feed video from the laptop camera

detect = FaceMeshDetector(maxFaces=1)  # detects the number of faces

eyeList = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243, 190, 230,
           359, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390,
           398, ]# points to identified on the eyes from the face mesh detector module

while True:
    success, img = video.read()
    img, faces = detect.findFaceMesh(img)

    if faces:
        face = faces[0]  # array on the number of maximum faces detected
        for eye in eyeList:
            cv2.circle(img, face[eye], 5, (255, 255, 0))

        # right eye
        rightUP = face[159]  # highest point on the eye
        rightDown = face[23]  # lowest point of the eye
        rightDistance, _ = detect.findDistance(rightDown, rightUP)
        print(f'right eye distance is {int(rightDistance)}')

        # left eye
        leftUp = face[386]
        leftDown = face[373]
        leftDistance, _ = detect.findDistance(leftUp, leftDown)
        print(f'left eye distance {int(leftDistance)}')
        ratio = (leftDistance + rightDistance) / 2

        if ratio< 14:
            cv2.putText(img, "DROWSY", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(img, "YOU ARE SLEEPY", (20, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            print('Drowsy')
            pygame.mixer.music.play(3)


        print(ratio)

    img = cv2.resize(img, (500, 360))
    cv2.imshow("video", img)
    cv2.waitKey(1)
