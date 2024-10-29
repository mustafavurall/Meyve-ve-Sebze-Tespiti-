import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
from PIL import Image

# Modeli yükle
model = load_model('C:\\Users\\1must\\Desktop\\MeyveSebze\\Image_classify2.keras')

# Sınıf adları
data_cat = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 
            'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 
            'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 
            'orange', 'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 
            'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 
            'turnip', 'watermelon']

img_height = 180
img_width = 180

# Görüentü yolu
image_path = st.text_input('Enter Image name','C:\\Users\\1must\\Desktop\\MeyveSebze\\Apple.jpg')

# Görüntüyü yükleyin ve yeniden boyutlandırın
image = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))
img_array = tf.keras.utils.img_to_array(image)
img_array = np.expand_dims(img_array, axis=0)  # Batch boyutu ekleyin

# Model ile tahmin yapın
predict = model.predict(img_array)

# Tahmin sonuçlarını işleyin
score = tf.nn.softmax(predict[0])
predicted_class = data_cat[np.argmax(score)]
accuracy = np.max(score) * 100

# Streamlit ile görselleştirme
st.image(image, caption='Uploaded Image')
st.write('Veg/Fruit in image is {} with accuracy of {:0.2f}%'.format(predicted_class, accuracy))
