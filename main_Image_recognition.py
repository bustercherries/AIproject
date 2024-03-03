import tensorflow as tf
from matplotlib import pyplot as plt
import os
plt.rcParams['figure.figsize'] = [20, 10]

# względna ścieżka do projektu
project_path = os.path.dirname(os.path.abspath(__file__))
# Dołączanie folderu do ścieżki projektu
all_dir = os.path.join(project_path, 'images dataset')
test_dir = os.path.join(project_path, 'extracted_frames_10')

resolution = 200
our_batch_size = 32
classes_no = 3

#rozbicie zbioru danych na treningowy i validacyjny
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2) # set validation split

train_flow = train_datagen.flow_from_directory(
  directory = all_dir, # Path for images folder
  color_mode = "rgb", # Images are in color
  target_size = (resolution, resolution), # Scale all images to given resolution
  batch_size = our_batch_size, # Batch size
  class_mode = "categorical" # Classification task
)

validation_flow = train_datagen.flow_from_directory(
  directory = all_dir,
  color_mode = "rgb",
  target_size = (resolution, resolution),
  batch_size = our_batch_size,
  class_mode = "categorical"
)

sample_batch = train_flow.next()


conv_base = tf.keras.applications.vgg16.VGG16(
  weights = "imagenet", # Weights trained on 'imagenet'
  include_top = False, # Without dense layers on top - we will add them later
  input_shape = (resolution, resolution, classes_no) # Same shape as in our generators
)

conv_base.summary()

for l in range(1, 7):
    conv_base.layers[l].trainable = False
conv_base.summary()


inputs = tf.keras.Input(shape=(resolution, resolution, 3))
outputs = conv_base(inputs, training=False)
outputs = tf.keras.layers.Flatten()(outputs)
outputs = tf.keras.layers.Dense(units = 256, activation = "relu")(outputs)
outputs = tf.keras.layers.Dense(units = 3, activation = "softmax")(outputs)

animals_model = tf.keras.Model(inputs, outputs)

animals_model.compile(
    optimizer = tf.keras.optimizers.RMSprop(learning_rate = 1e-5),
    loss = "categorical_crossentropy",
    metrics = ("accuracy"))

animals_model.summary()

history = animals_model.fit(
        train_flow,
        steps_per_epoch=22, # ceiling(694 / 32)
        epochs=15,
        validation_data=validation_flow,
        validation_steps=6) # ceiling(184 / 32)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale = 1 / 255
)

test_flow = train_datagen.flow_from_directory(
  directory = test_dir, # Path for train images folder
  color_mode = "rgb", # Images are in color
  target_size = (resolution, resolution), # Scale all images to 150x150
  batch_size = our_batch_size, # Batch size
  class_mode = "categorical" # Classification task
)

animals_model.evaluate(test_flow, steps = 18)
