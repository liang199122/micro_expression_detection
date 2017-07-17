from sklearn.svm import LinearSVC
from imutils import paths
from matplotlib import pyplot as plt
from skimage import feature
import numpy as np
import cv2


count = 0
data = []
labels = []



for imagePath in paths.list_images("output/crop/"):
    plt.figure()
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    colors = ("b", "g", "r")
    count = count + 1
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    lbp = feature.local_binary_pattern(gray, 59, 3, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 59 + 3), range=(0, 3 + 2))
    plt.plot(hist)

    data.append(hist)
    #plt.show()
    plt.savefig("output/histgrapys/" + str(count) + "" + ".jpg")

    print(lbp)
#print(data)

#print("width: {} pixels".format(test.shape[1]))
#print("height: {} pixels".format(test.shape[0]))
#print("channels: {}".format(test.shape[2]))


'''
    # extract the label from the image path, then update the
    # label and data lists
    labels.append(imagePath.split("/")[-2])
    data.append(hist)

    # train a Linear SVM on the data
    model = LinearSVC(C=100.0, random_state=42)
    model.fit(data, labels)

    # loop over the testing images
    for imagePath in paths.list_images(args["testing"]):
        # load the image, convert it to grayscale, describe it,
        # and classify it
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = desc.describe(gray)
        prediction = model.predict(hist)[0]

        # display the image and the prediction
        cv2.putText(image, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 255), 3)
        cv2.imshow("Image", image)
        cv2.waitKey(0)
'''