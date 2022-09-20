import sys
import cv2

def print_hello():
    print("hello...it's me")

def main():
    print_hello()

    path = sys.argv[1]
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)



if __name__ == "__main__":
    main()
