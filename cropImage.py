import dlib
import cv2


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("extra_file/shape_predictor_68_face_landmarks.dat")

image = cv2.imread("test_image/test2.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
rects = detector(gray, 1)

for i, d in enumerate(rects):
    crop = image[d.top()-20:d.bottom()+20, d.left()-30:d.right()]

cv2.imwrite("output/cropped.jpg", crop)

crop = image[0:1024, 0:768]

cv2.imwrite("seesize.jpg", crop)