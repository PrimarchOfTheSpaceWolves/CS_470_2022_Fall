from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Conv2D
import cv2
import numpy as np

def main():
    input_node = Input(shape=(None, None, 1))
    filter_layer = Conv2D(1, kernel_size=3, use_bias=False, padding="same")
    output_node = filter_layer(input_node)
    model = Model(inputs=input_node, outputs=output_node)
    model.compile(optimizer="adam", loss="mse", metrics=["mse"])
    model.summary()

    camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not camera.isOpened():
        print("HELP! NO CAMERA!")
        exit(1)

    key = -1

    while key == -1:
        ret, frame = camera.read()        
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #cv2.imshow("INPUT", grayscale)
        grayscale = np.expand_dims(grayscale, axis=-1)
        #print(grayscale.shape)

        sobelx = cv2.Sobel(grayscale, cv2.CV_32F, dx=1, dy=0, ksize=3,
                            scale=0.25)
        cv2.imshow("INPUT", 
                    cv2.convertScaleAbs(sobelx, alpha=0.5, beta=127.0))
        sobelx = np.expand_dims(sobelx, axis=-1)
                
        batch_input = np.expand_dims(grayscale, axis=0)
        batch_output = np.expand_dims(sobelx, axis=0)

        batch_input = batch_input.astype("float32")/255.0
        batch_output = batch_output.astype("float32")/255.0

        losses = model.train_on_batch(batch_input, batch_output)

        print("LOSS:", losses)
        print("WEIGHTS:\n", filter_layer.weights[0].numpy())

        pred_image = model.predict(batch_input)
        pred_image = np.squeeze(pred_image, axis=0)
        pred_image *= 255.0        
        pred_image = cv2.convertScaleAbs(pred_image, alpha=0.5, beta=127.0)

        cv2.imshow("PREDICTION", pred_image)

        key = cv2.waitKey(30)

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

