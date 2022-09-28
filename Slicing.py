import numpy as np
import cv2

image = np.zeros((20,10))

iheight, iwidth = image.shape  
print("HEIGHT:", iheight)
print("WIDTH:", iwidth)

kernel = np.zeros((5,7))
kheight, kwidth = kernel.shape
kh = kheight // 2
kw = kwidth // 2

padimage = cv2.copyMakeBorder(image,
                                kh, kh,
                                kw, kw,
                                borderType=cv2.BORDER_CONSTANT,
                                value=0)

for r in range(0, iheight): # iheight-kheight+1):
    for c in range(0, iwidth): # iwidth-kwidth+1):
        sub = padimage[r:(r+kheight), c:(c+kwidth)]
        print(sub.shape)
        multresult = kernel*sub
        print("Result:", multresult.shape)
        sumval = np.sum(multresult)
        print("Sum:", sumval)


