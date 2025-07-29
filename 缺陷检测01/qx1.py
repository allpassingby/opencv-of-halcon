import numpy as np
import cv2

font = cv2.FONT_HERSHEY_SIMPLEX

img = cv2.imread('plastic_mesh_06.png')
cv2.imshow('src',img)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (3,3), 1.0)

#ret,thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
_,thresh1 = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY_INV )
_,thresh2 = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV )
thresh = thresh1 - thresh2
#cv2.imshow('thresh',thresh)

contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

isNG = False
for cnt in contours:
  area = cv2.contourArea(cnt)
  if(area>350):
    cv2.drawContours(img,cnt,-1,(0,0,255),2)
    isNG = True
#  else:
#    cv2.drawContours(img,cnt,-1,(0,255,0),1)

if isNG:
  rect, basline = cv2.getTextSize('Mesh Not OK', font, 1.0, 2)
  cv2.rectangle(img, (10,10,int(rect[0]*0.7),rect[1]), (212, 233, 252), -1, 8)
  cv2.putText(img,'Mesh Not OK', (10,5+rect[1]), font, 0.7, (0,0,255), 2)
else:
  rect, basline = cv2.getTextSize('Mesh OK', font, 1.0, 2)
  cv2.rectangle(img, (10,10,int(rect[0]*0.7),rect[1]), (212, 233, 252), -1, 8)
  cv2.putText(img,'Mesh OK', (10,5+rect[1]), font, 0.7, (0,200,0), 2)
cv2.imshow('meshDefects', img)
cv2.waitKey(0)
cv2.destroyAllWindows()