import cv2
import numpy as np

def contour():
    img = cv2.imread('./KakaoTalk_20190202_132452238.jpg')
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thr = cv2.threshold(imgray, 127, 255, 0)
    contours, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(img, contours, -1, (0, 0, 0), 2)
    cv2.imshow('thresh', thr)
    cv2.imshow('contour', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    x, y, width, height = cv2.boundingRect(imgray)
    roi = img[y:y+height, x:x+width]
    cv2.imwrite('test.jpg', roi)

contour()