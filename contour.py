import cv2
import numpy as np

def contour():
    img = cv2.imread('./dataset/original_images/original_1.jpg')
    img = cv2.resize(img, (1000, 1500))
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgray = cv2.medianBlur(imgray, 5)

    thresh = cv2.adaptiveThreshold(imgray, 255, 1, 1, 11, 2)
    thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 3))

    #thresh = cv2.dilate(thresh, kernel, 1)
    #thresh = cv2.erode(thresh, kernel, 1)

    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:

        x, y, w, h = cv2.boundingRect(cnt)
        if w > 220 :
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.rectangle(thresh_color, (x, y), (x+w, y+h), (0, 255, 0), 2)


    cv2.imshow('contour', img)
    cv2.imshow('thresh', thresh_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

contour()