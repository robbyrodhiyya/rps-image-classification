import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files


# Load Dataset TFDS
(ds_train, ds_test), ds_info = tfds.load(
    'rock_paper_scissors',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)


# Preprocessing
IMG_SIZE = 150

def format_image(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0
    return image, label

train_ds = ds_train.map(format_image).cache().shuffle(500).batch(32).prefetch(tf.data.AUTOTUNE)
test_ds  = ds_test.map(format_image).cache().batch(32).prefetch(tf.data.AUTOTUNE)


# Model CNN
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(150,150,3)),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


# Training
history = model.fit(
    train_ds,
    epochs=15,
    validation_data=test_ds
)


# Upload Gambar dan Prediksi
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(150,150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    label = np.argmax(pred)

    classes = ["rock", "paper", "scissors"]

    plt.imshow(img)
    plt.axis("off")
    plt.show()

    print("Prediksi:", classes[label])


# Upload
uploaded = files.upload()

for fn in uploaded.keys():
    predict_image(fn)