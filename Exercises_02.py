import cv2
import numpy as np

counter = 0
MAX_COUNTER = 30
last_image = None

def process(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    global counter
    global last_image

    counter += 1
    if last_image is None:
        last_image = np.copy(image).astype("float32")

    if counter >= MAX_COUNTER:
        counter = 0
        last_image = np.copy(image).astype("float32")

    image = cv2.convertScaleAbs(image.astype("float32")*0.5 + last_image*0.5)
    return image

    '''
    minVal = 100
    maxVal = 200
    image = np.where(image <= minVal, minVal, image)
    image = np.where(image >= maxVal, maxVal, image)
    '''
    
    
'''
myimage = cv2.imread("test.jpg")
myimage = process(myimage)

print("HELLLLOOOOOOOO!")

a = [0,10,20,30,40,50]

print(a[1])
print(len(a))

print(a[1:3])

print(a[:3])
print(a[1:])
print(a[:])

image = np.zeros((480, 640, 3), dtype="uint8")
image[:100,:] = (0,0,255)
# image[:,:,:] = 128
# image[:,:,0:1] = 128
image[:,:,0] = 128
print(image[:,:,0:1].shape)
print(image[:,:,0].shape)

myimage = cv2.resize(myimage, dsize=(0,0), fx=0.1, fy=0.1)
myimage = cv2.resize(myimage, dsize=(0,0), fx=10.0, fy=10.0, 
                        interpolation=cv2.INTER_NEAREST)

cv2.imshow("IMAGE", image)
cv2.imshow("OTHER IMAGE", myimage)
cv2.waitKey(-1)
cv2.destroyAllWindows()
'''


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

    frame = process(frame)

    # Show the image
    cv2.imshow(windowName, frame)

    # Wait 30 milliseconds, and grab any key presses
    key = cv2.waitKey(30)

# Release the camera and destroy the window
camera.release()
cv2.destroyAllWindows()

# Close down...
print("Closing application...")






