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

def pick_closest(data, centroid, dist):
    centroid = np.array(centroid, dtype="float32")
    centroid = np.reshape(centroid, (1,3))
    data = data.astype("float32")
    print("data:", data.shape)
    print("centroid:", centroid.shape)
    data = data - centroid
    print("data after subtraction:", data.shape)
    data = np.square(data)
    data = np.sum(data, axis=1)
    data = np.sqrt(data)
    print("data after sum:", data.shape)

    data = np.where(data < dist, 
                    np.ones(data.shape), 
                    np.zeros(data.shape))
    print("final data:", data.shape)

    return data

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
            pixel_list = np.reshape(frame, (-1, 3))
            print("pixel_list:", pixel_list.shape)

            # BGR
            chosen_ones = pick_closest(pixel_list, 
                                        (0,0,255),
                                        200)

            chosen_ones = np.reshape(chosen_ones, frame.shape[0:2])
            
            # Show the image
            cv2.imshow(windowName, frame)
            cv2.imshow("CHOSEN", chosen_ones)

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