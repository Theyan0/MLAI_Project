from PIL import Image, ImageOps
import tensorflow as tf

import cv2
import numpy as np
import os
import sys



label = ''

frame = None

from PIL import ImageOps

def import_and_predict(image_data, model):
    size = (150, 150)  # Use the size your model expects
    image = ImageOps.fit(image_data, size, Image.LANCZOS)
    image = image.convert('RGB')
    image = np.asarray(image)
    image = image.astype(np.float32) / 255.0  # Normalize as done during training
    img_reshape = np.expand_dims(image, axis=0)  # Reshape to (1, 150, 150, 3)

    prediction = model.predict(img_reshape)
    return prediction




model = tf.keras.models.load_model('C:/Users/Aidan/Important/MLAI_Project/MLAI_Project/Models/model03_theyan.h5')

    
cap = cv2.VideoCapture(0)

if (cap.isOpened()):
    print("Camera OK")
else:
    cap.open()

while (True):
    ret, original = cap.read()

    frame = cv2.resize(original, (224, 224))
    cv2.imwrite(filename='img.jpg', img=original)
    image = Image.open('img.jpg')

    # Display the predictions
    # print("ImageNet ID: {}, Label: {}".format(inID, label))
    prediction = import_and_predict(image, model)
    #print(prediction)

    if np.argmax(prediction) == 0:
        predict="It is bao!"
    elif np.argmax(prediction) == 1:
        predict="It is neither!"
    else:
        predict="It is uni sushi!"
    
    cv2.putText(original, predict, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow("Classification", original)

    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break;

cap.release()
frame = None
cv2.destroyAllWindows()
sys.exit()
