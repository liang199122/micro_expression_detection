from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
from imutils.face_utils import FACIAL_LANDMARKS_IDXS

'''
def get_landmark_area(list, image):

    for x, y in enumerate(list):
        #cv2.rectangle(image, (x-8, y-7), (x + 9, y + 8), (0, 255, 0), 2)
        cropped = image[x-8:x+9, y-7:y+8]
        cv2.imshow("T-Rex Face", cropped)
        cv2.waitKey(0)
'''

count = 0
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("extra_file/shape_predictor_68_face_landmarks.dat")

image = cv2.imread("test_image/test2.jpg")
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

rects = detector(gray, 1)
maskList = []

for (i, rect) in enumerate(rects):

    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

# create mask for image
    for (m, name) in enumerate(FACIAL_LANDMARKS_IDXS.keys()):
        (j, k) = FACIAL_LANDMARKS_IDXS[name]
        pts = shape[j:k]
        mpts = shape[j:k]

        #get_landmark_area(shape, gray)
        if name == "left_eyebrow":
            mpts[:, 1] = mpts[:, 1] - 15
            leyeblowpts = mpts
        if name == "right_eyebrow":
            mpts[:, 1] = mpts[:, 1] - 15
            reyeblowpts = mpts
        if name =="jaw":
            jawpts = mpts

eyeblowdata = np.append(leyeblowpts, reyeblowpts, 0)
maskdata = np.append(eyeblowdata, jawpts, 0)
hull = cv2.convexHull(maskdata)

mask = np.zeros(gray.shape, dtype=np.uint8)
cv2.drawContours(mask, [hull], -1, 255, -1)
masked = cv2.bitwise_and(gray, gray, mask=mask)
cv2.imwrite('output/image_masked.jpg', masked)

for x, y in (shape):
    count = count+1
    maskimg = cv2.imread("output/image_masked.jpg")
    #print("x====", x)
    #print("y====", y)
    cv2.rectangle(gray, (x - 8, y - 7), (x + 9, y + 8), (255, 255, 255), 2)
    cv2.imshow("show retangle", gray)
    cv2.waitKey(0)

    cropped = gray[x - 9:x + 9, y - 8:y + 8]
    cv2.imshow("Face crop", cropped)
    cv2.waitKey(0)
    cv2.imwrite("output/crop/"+str(count)+""+".jpg", cropped)