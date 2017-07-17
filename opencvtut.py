import matplotlib.pyplot as plt
import numpy as np
import cv2


def plot_histogram(image, tilte, mask = None):
    chans = cv2.split(image)
    colors = ("b", "g", "r")
    plt.figure()
    plt.title(tilte)
    plt.xlabel("bins")
    plt.ylabel("# of Pixels")

    for(chan, color) in zip(chans, colors):
        hist = cv2.calcHist([chan], [0], mask, [256], [0, 256])
        plt.plot(hist, color)
        plt.xlim([0, 256])

image = cv2.imread("test_image/test2.jpg")
cv2.imshow("original", image)
plot_histogram(image, "Histogram for Original Image")

mask = np.zeros(image.shape[:2], dtype="uint8")
cv2.rectangle(mask, (15, 15), (130, 100), 255, -1)
cv2.imshow("Mask", mask)

masked = cv2.bitwise_and(image, image, mask = mask)
cv2.imshow("Masked", masked)
plot_histogram(image, "Histogram for Masked", mask = mask)
plt.show()