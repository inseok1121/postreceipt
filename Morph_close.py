import cv2
import numpy as np


image = cv2.imread('./dataset/original_images/original_6.jpg', cv2.IMREAD_GRAYSCALE)

(thresh, im_bw) = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

thresh = 127
image = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)[1]


image = cv2.GaussianBlur(image, (5, 5), 0)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
closing = cv2.morphologyEx(im_bw, cv2.MORPH_CLOSE, kernel)
erode = cv2.erode(image, kernel, 1)
dilation = cv2.dilate(erode, kernel, 1)

cv2.imshow("erode", erode)
cv2.imshow("dilation", dilation)


cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("./dataset/modify_images/test1.jpg", erode)
cv2.imwrite("./dataset/modify_images/test2.jpg", dilation)

