import cv2
import numpy as np

cropdata = []

def contour(image):


    thresh = cv2.adaptiveThreshold(image, 255, 1, 1, 11, 2)
    thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 3))

    #thresh = cv2.dilate(thresh, kernel, 1)
    #thresh = cv2.erode(thresh, kernel, 1)

    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:

        x, y, w, h = cv2.boundingRect(cnt)
        if w > 220 and h < 100:
            img = image.copy()
            cropdata.append(img[y:y+h+10, x:x+w+10])
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.rectangle(thresh_color, (x, y), (x+w, y+h), (0, 255, 0), 2)


    for da in cropdata:
        cv2.imshow("da", da)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def shadowremoval(image):

    rbg_planes = cv2.split(image)
    result_planes = []
    result_norm_plane = []

    for plane in rbg_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8()))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_plane.append(norm_img)

    result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_plane)

    return result, result_norm

image = cv2.imread('./dataset/original_images/original_1.jpg', cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (1000, 1500))

b, image = shadowremoval(image)

(thresh, im_bw) = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


thresh = 127
image = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)[1]


image = cv2.GaussianBlur(image, (5, 5), 0)

contour(image)








