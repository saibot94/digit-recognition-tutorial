from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np

import cv2

clf = joblib.load('./model/digits_clf.pkl')
print '=> Clasificatorul a fost incarcat'

im = cv2.imread('./image/digit-reco.jpg')
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, ksize=(5, 5), sigmaX=0)

ret, image_threshold = cv2.threshold(im_gray, 91, 255, cv2.THRESH_BINARY_INV)
contours, hierarchy = cv2.findContours(image_threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

rectangles = [cv2.boundingRect(ctr) for ctr in contours]
for rect in rectangles:
    # Desenare dreptunghi
    cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)

    leng = int(rect[3] * 1.5)
    pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
    pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
    region_of_interest = image_threshold[pt1:pt1 + leng, pt2:pt2 + leng]
    roi = cv2.resize(region_of_interest, (28, 28), interpolation=cv2.INTER_AREA)
    roi = cv2.dilate(roi, (3, 3))

    fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    nbr = clf.predict(np.array([fd], 'float64'))
    cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)

cv2.imshow("Final classification", im)
cv2.waitKey()
