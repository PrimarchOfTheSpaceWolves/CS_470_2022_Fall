import cv2
import numpy as np
import tensorflow as tf
import sklearn
import pandas
from tensorflow.keras.datasets import mnist
import tensorflow.keras.utils as utils
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import (accuracy_score, f1_score, 
                            roc_auc_score, confusion_matrix)

def compute_metrics(ground, pred, class_cnt):
    scores = {}
    scores["accuracy"] = accuracy_score(y_true=ground, y_pred=pred)
    scores["f1"] = f1_score(y_true=ground, y_pred=pred, average="macro")

    one_ground = utils.to_categorical(ground, num_classes=class_cnt)
    one_pred = utils.to_categorical(pred, num_classes=class_cnt)

    scores["auc"] = roc_auc_score(y_true=one_ground, 
                                    y_score=one_pred,
                                    multi_class="ovr")
    return scores

def normalize_images(images):
    images = images.astype("float32")
    images /= 255.0
    # [0,1] --> [-1,1]
    images -= 0.5
    # [-0.5, 0.5]
    images *= 2.0
    # [-1, 1]
    images = np.expand_dims(images, axis=-1)
    return images

def convert_to_display_images(images):
    images /= 2.0
    images += 0.5
    images *= 255.0
    images = images.astype("uint8")
    return images

def display_image(image):
    image = convert_to_display_images(image)
    cv2.imshow("IMAGE", image)
    cv2.waitKey(-1)
    cv2.destroyAllWindows()

def main():
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train = normalize_images(x_train)
    x_test = normalize_images(x_test)
    print("x_train:", x_train.shape)
    print("y_train:", y_train.shape)
    print("x_test:", x_test.shape)
    print("y_test:", y_test.shape)

    #display_image(x_train[0])
    
    x_train = np.reshape(x_train, (x_train.shape[0], -1))
    x_test = np.reshape(x_test, (x_test.shape[0], -1))

    print("AFTER FLATTEN:")
    print("x_train:", x_train.shape)    
    print("x_test:", x_test.shape)

    # K Nearest Neighbors
    #classifier = KNeighborsClassifier(n_neighbors=3, 
    #                                    weights="uniform")
    #
    # Minimum Distance Classifier
    #classifier = NearestCentroid()
    #
    # AdaBoost
    classifier = AdaBoostClassifier(n_estimators=100, random_state=0)

    print("TRAINING...")
    classifier.fit(x_train, y_train)
    print("YOUR TRAINING IS COMPLETE!")

    print("PREDICTING...")
    pred_train = classifier.predict(x_train)
    pred_test = classifier.predict(x_test)
    print("PREDICTION ACHIEVED!")

    class_cnt = 10
    train_scores = compute_metrics(y_train, pred_train, class_cnt)
    test_scores = compute_metrics(y_test, pred_test, class_cnt)

    print("TRAIN:", train_scores)
    print("TEST:", test_scores)

    print("TRAIN CONFUSION:\n", confusion_matrix(y_train, pred_train))
    print("TEST CONFUSION:\n", confusion_matrix(y_test, pred_test))




if __name__ == "__main__":
    main()

