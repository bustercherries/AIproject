import tensorflow as tf
from matplotlib import pyplot as plt
import os
import numpy as np
plt.rcParams['figure.figsize'] = [20, 10]

# relative path to the project
project_path = os.path.dirname(os.path.abspath(__file__))
# Appending folder names to the project path
all_dir = os.path.join(project_path, 'train_and_val')
test_dir = os.path.join(project_path, 'Test dataset')

resolution_y = 300
resolution_x = 300
our_batch_size = 32
classes_no = 3
epochs_no = 10
continue_learning = 1 #if 1 it starts learning from previously trained model
trained_model_filename = "models/animals_test_resnet300x300.hdf5"


#splitting the data to train and validation
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2) # set validation split

train_flow = train_datagen.flow_from_directory(
  directory = all_dir, # Path for images folder
  color_mode = "rgb", # Images are in color
  target_size = (resolution_y, resolution_x), # Scale all images to given resolution
  batch_size = our_batch_size, # Batch size
  class_mode = "categorical", # Classification task
  subset='training'
)

validation_flow = train_datagen.flow_from_directory(
  directory = all_dir,
  color_mode = "rgb",
  target_size = (resolution_y, resolution_x),
  batch_size = our_batch_size,
  class_mode = "categorical",
  subset='validation'
)

"""#vgg16
conv_base = tf.keras.applications.vgg16.VGG16(
  weights = "imagenet", # Weights trained on 'imagenet'
  include_top = False, # Without dense layers on top - we will add them later
  input_shape = (resolution_y, resolution_x, classes_no) # Same shape as in our generators
)"""
#resnet
conv_base = tf.keras.applications.ResNet50(
  weights = "imagenet", # Weights trained on 'imagenet'
  include_top = False, # Without dense layers on top - we will add them later
  input_shape = (resolution_y, resolution_x, classes_no) # Same shape as in our generators
)

for l in range(1, 7):
    conv_base.layers[l].trainable = False
conv_base.summary()

inputs = tf.keras.Input(shape=(resolution_y, resolution_x, 3))
outputs = conv_base(inputs, training=False)
outputs = tf.keras.layers.Flatten()(outputs)
outputs = tf.keras.layers.Dense(units = 256, activation = "relu")(outputs)
outputs = tf.keras.layers.Dense(units = 3, activation = "softmax")(outputs)

animals_model = tf.keras.Model(inputs, outputs)

animals_model.compile(
    optimizer = tf.keras.optimizers.RMSprop(learning_rate = 1e-5),
    loss = "categorical_crossentropy",
    metrics = ["accuracy"])

animals_model.summary()

if continue_learning == 1:
    animals_model = tf.keras.models.load_model(trained_model_filename)

steps_per_epoch = train_flow.samples/our_batch_size
validation_steps = validation_flow.samples/our_batch_size

history = animals_model.fit(
        train_flow,
        steps_per_epoch= steps_per_epoch,
        epochs= epochs_no,
        validation_data=validation_flow,
        validation_steps=validation_steps,
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
             tf.keras.callbacks.ModelCheckpoint(filepath=trained_model_filename, monitor="val_accuracy", save_best_only=True)])


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

test_flow = test_datagen.flow_from_directory(
  directory = test_dir, # Path for test images folder
  color_mode = "rgb", # Images are in color
  target_size = (resolution_y, resolution_x), # Scale all images to 150x150
  batch_size = our_batch_size, # Batch size
  class_mode = "categorical", # Classification task
  shuffle = False
)
print (test_flow.class_indices)
print (test_flow.filenames)
#animals_model.evaluate(test_flow, steps = 20)#test_flow.n // test_flow.batch_size)


#visualisation of guesses
class_names = list(test_flow.class_indices.keys())


def plot_value_img(i, predictions, true_labels, images, file_names, class_names):
    prediction, true_label, img = predictions[i], true_labels[i], images[0]
    predicted_label = np.argmax(prediction)
    true_value = true_label

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)

    plt.yticks(np.arange(len(class_names)), class_names)
    thisplot = plt.barh(range(len(class_names)), prediction, color="gray")
    thisplot[predicted_label].set_color('r')
    thisplot[true_value].set_color('g')

    plt.subplot(1, 2, 2)

    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    if predicted_label == true_value:
        color = 'green'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(prediction),
                                         class_names[true_value]),
               color=color)

    # Dodanie nazwy pliku obok obrazka
    plt.title("File Name: {}".format(file_names[i]))
    plt.show()


# Predykcje dla danych testowych
y_test_pred = animals_model.predict(test_flow)

# Wizualizacja dla pierwszych przykładów ze zbioru walidacyjnego
for i in range(12):
    plot_value_img(i, y_test_pred, test_flow.labels, test_flow[i][0], test_flow.filenames, class_names)