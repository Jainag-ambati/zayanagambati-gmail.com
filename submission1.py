def predict(img_path):
  import tensorflow as tf
  import pandas as pd
  import numpy as np
  import mnist
  from mnist import MNIST
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
  from tensorflow.keras.layers import Conv2D, MaxPooling2D
  data=tf.keras.datasets.mnist
  (x_train,y_train),(x_test,y_test)=data.load_data()
  x_train, x_test = x_train / 255.0, x_test / 255.0
  x_train = np.expand_dims(x_train, axis=3)



  model=tf.keras.models.Sequential()


 
  model.add(Conv2D(20, (5, 5), padding="same", input_shape=(x_train.shape[1:]), activation="relu"))
  model.add(MaxPooling2D(pool_size=(2, 2) , strides=(2,2)))
  model.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
  model.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
  model.add(Flatten())

  model.add(tf.keras.layers.Dense(500, activation="relu"))
  model.add(tf.keras.layers.Dense(10, activation="softmax"))
  model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
  model.fit(x_train,y_train,epochs=10,batch_size=32)
  model.save("captcha.model")



  x_test = np.expand_dims(x_test, axis=3)
  v_acc,v_loss=model.evaluate(x_test,y_test)


  import os
  import os.path
  import glob
  import cv2

  captcha_image_file=img_path
  OUTPUT_FOLDER = "C:\\Users\\windows\\Desktop\\my trials\\output"
  os.mkdir(OUTPUT_FOLDER)
  image = cv2.imread(captcha_image_file)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)

  thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
  contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


  contours = contours[0] 
  letter_image_regions = []
  for contour in contours:

      (x, y, w, h) = cv2.boundingRect(contour)
      if w / h > 1.25:
        half_width = int(w / 2)
        letter_image_regions.append((x, y, half_width, h))
        letter_image_regions.append((x + half_width, y, half_width, h))
      else:
        letter_image_regions.append((x, y, w, h))
  letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])
  n=0
  for i,letter_bounding_box in enumerate(letter_image_regions):
      (x, y, w, h) = letter_bounding_box
      letter_image = gray[y - 2:y + h + 2, x - 2:x + w + 2]
      save_path = OUTPUT_FOLDER
      p = os.path.join(save_path, "{}.png".format(i+1))
      letter_image=cv2.resize(letter_image,(28,28))
      cv2.imwrite(p, letter_image)
      n=n+1

  import shutil

  for i in range(n):
    img=cv2.imread("C:\\Users\\windows\\Desktop\\my trials\\output\\{}.png".format(i+1))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=3)
    predict=new_model.predict([img])
    if i==0:
        p=str(np.argmax(predict))
    else:
        p=p + str(np.argmax(predict))
  print(p)
  shutil.rmtree(OUTPUT_FOLDER)


predict("C:\\Users\\windows\\Desktop\\solving_captchas_code_examples\\New\\hand.png")

























    
