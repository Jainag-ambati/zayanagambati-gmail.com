# -*- coding: utf-8 -*-
import cv2
import pytesseract

def predict(img):
 pytesseract.pytesseract.tesseract_cmd=r"C:\Program Files\Tesseract-OCR\tesseract.exe"
 imge=cv2.imread(img)
 text=pytesseract.image_to_string(imge)
 print(text)

predict("C:/Users/windows/Desktop/New/num.png")
