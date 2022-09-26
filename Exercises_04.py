# MIT LICENSE
#
# Copyright 2022 Michael J. Reale
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

###############################################################################
# IMPORTS
###############################################################################

import numpy as np
import tensorflow as tf 
import cv2
import pandas
import sklearn
import sys
from enum import Enum

class FilterType(Enum):
    BOX = 0
    GAUSS = 1
    MEDIAN = 2
    LAPLACE = 3
    SHARP_LAPLACE = 4
    SOBEL_X = 5
    SOBEL_Y = 6
    GRAD_MAG = 7

def filterMagic(image, filterSize, filterType):
    output = np.copy(image)
    # Magic here
    if filterType == FilterType.BOX:
        output = cv2.blur(output, (filterSize, filterSize))
    elif filterType == FilterType.GAUSS:
        output = cv2.GaussianBlur(output, (filterSize, filterSize), 0)
    elif filterType == FilterType.MEDIAN:
        output = cv2.medianBlur(output, filterSize)
    elif filterType == FilterType.LAPLACE:
        laplace = cv2.Laplacian(output, cv2.CV_32F, 
                                ksize=filterSize, scale=0.25)
        output = cv2.convertScaleAbs(laplace, alpha=0.5, beta=127.0)
    elif filterType == FilterType.SHARP_LAPLACE:
        laplace = cv2.Laplacian(output, cv2.CV_32F, 
                                ksize=filterSize, scale=0.25)
        fimage = output.astype("float32")
        fimage -= laplace
        output = cv2.convertScaleAbs(fimage)
    elif filterType == FilterType.SOBEL_X:
        sobelx = cv2.Sobel( output, cv2.CV_32F, 
                            dx=1, dy=0, ksize=filterSize,
                            scale=0.25)
        output = cv2.convertScaleAbs(sobelx, alpha=0.5, beta=127.0)
    elif filterType == FilterType.SOBEL_Y:
        sobely = cv2.Sobel( output, cv2.CV_32F, 
                            dx=0, dy=1, ksize=filterSize,
                            scale=0.25)
        output = cv2.convertScaleAbs(sobely, alpha=0.5, beta=127.0)
    elif filterType == FilterType.GRAD_MAG:
        sobelx = cv2.Sobel( output, cv2.CV_32F, 
                            dx=1, dy=0, ksize=filterSize,
                            scale=0.25)
        sobely = cv2.Sobel( output, cv2.CV_32F, 
                            dx=0, dy=1, ksize=filterSize,
                            scale=0.25)
        gradimage = np.absolute(sobelx) + np.absolute(sobely)
        output = cv2.convertScaleAbs(gradimage)

        
    return output

###############################################################################
# MAIN
###############################################################################

def main():
    ###############################################################################
    # TENSORFLOW
    ###############################################################################

    a = tf.constant("Hello Tensorflow!")
    tf.print(a)
    print(tf.config.list_physical_devices('GPU'))
    print(tf.reduce_sum(tf.random.normal([1000, 1000])))

    ###############################################################################
    # PRINT OUT VERSIONS
    ###############################################################################

    print("Tensorflow:", tf.__version__)    
    print("Numpy:", np.__version__)
    print("OpenCV:", cv2.__version__)
    print("Pandas:", pandas.__version__)
    print("Scikit-Learn:", sklearn.__version__)
    
    print("It's filtering time!")
    for item in list(FilterType):
        print(item.value, "-", item.name)
    filterType = FilterType(int(input("Enter filter type: ")))
    filterSize = int(input("Enter size: "))
    
    ###############################################################################
    # OPENCV
    ###############################################################################
    if len(sys.argv) <= 1:
        # Webcam
        print("Opening webcam...")

        camera = cv2.VideoCapture(1, cv2.CAP_DSHOW) # CAP_DSHOW recommended on Windows

        # Did we get it?
        if not camera.isOpened():
            print("ERROR: Cannot open camera!")
            exit(1)

        # Create window ahead of time
        windowName = "Webcam"
        cv2.namedWindow(windowName)

        # While not closed...
        key = -1
        while key == -1:
            # Get next frame from camera
            ret, frame = camera.read()

            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            processed = filterMagic(frame, filterSize, filterType)

            # Show the image
            cv2.imshow(windowName, frame)
            cv2.imshow("FILTERED!", processed)

            # Wait 30 milliseconds, and grab any key presses
            key = cv2.waitKey(30)

        # Release the camera and destroy the window
        camera.release()
        cv2.destroyAllWindows()

        # Close down...
        print("Closing application...")

    else:
        # Trying to load image from argument

        # Get filename
        filename = sys.argv[1]

        # Load image
        print("Loading image:", filename)
        image = cv2.imread(filename) # For grayscale: cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

        # Check if data is invalid
        if image is None:
            print("ERROR: Could not open or find the image!")
            exit(1)

        # Show our image (with the filename as the window title)
        windowTitle = "PYTHON: " + filename
        cv2.imshow(windowTitle, image)

        # Wait for a keystroke to close the window
        cv2.waitKey(-1)

        # Cleanup this window
        cv2.destroyAllWindows()

if __name__ == "__main__": 
    main()