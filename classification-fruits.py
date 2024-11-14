import sys
sys.stdout.reconfigure(encoding='utf-8')
from tensorflow import keras
ImageDataGenerator = keras.preprocessing.image.ImageDataGenerator
image = keras.preprocessing.image
Sequential = keras.models.Sequential
Model = keras.models.Model
GlobalAveragePooling2D = keras.layers.GlobalAveragePooling2D
Dense = keras.layers.Dense
BatchNormalization = keras.layers.BatchNormalization
Dropout = keras.layers.Dropout
Flatten = keras.layers.Flatten
Conv2D = keras.layers.Conv2D
MaxPooling2D = keras.layers.MaxPooling2D
AveragePooling2D = keras.layers.AveragePooling2D
Adam = keras.optimizers.Adam
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
plt.ion()

im_shape = (250,250)

TRAINING_DIR = './input/ds_frutas_am/train'
TEST_DIR = './input/ds_frutas_am/test'
seed = 10
BATCH_SIZE = 16

data_generator = ImageDataGenerator(rescale=1./255, validation_split=0.2)
val_data_generator = ImageDataGenerator(rescale=1./255, validation_split=0.2)
test_generator = ImageDataGenerator(rescale=1./255)

train_generator = data_generator.flow_from_directory(TRAINING_DIR, target_size=im_shape, shuffle=True, seed=seed,
                                                     class_mode='categorical', batch_size=BATCH_SIZE, subset="training")
validation_generator = val_data_generator.flow_from_directory(TRAINING_DIR, target_size=im_shape, shuffle=False, seed=seed,
                                                     class_mode='categorical', batch_size=BATCH_SIZE, subset="validation")
test_generator = test_generator.flow_from_directory(TEST_DIR, target_size=im_shape, shuffle=False, seed=seed,
                                                     class_mode='categorical', batch_size=BATCH_SIZE)

nb_train_samples = train_generator.samples
nb_validation_samples = validation_generator.samples
nb_test_samples = test_generator.samples
classes = list(train_generator.class_indices.keys())
print('Classes: '+str(classes))
num_classes  = len(classes)

model = Sequential()

# Lớp tích chập đầu tiên, xác định input (im_shape[0], im_shape[1], 3)
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(im_shape[0], im_shape[1], 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Lớp tích chập thứ hai
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Lớp tích chập thứ ba
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(2, 2)))

# Lớp tích chập thứ tư
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(2, 2)))

# # Có thể xây thêm các Lớp tích chập khác ... 
# model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
# model.add(AveragePooling2D(pool_size=(2, 2)))

# Lớp Flatten để chuyển từ tensor 3D sang vector 1D
model.add(Flatten())

# Lớp fully connected đầu tiên
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))  # Để tránh overfitting

# Lớp fully connected thứ hai
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))   

# Lớp output, số lượng lớp tùy thuộc vào số lượng classes cần phân loại
model.add(Dense(num_classes, activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])


epochs = 50
callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='model.keras',  # Sử dụng đuôi '.keras' thay vì '.h5'
        monitor='val_loss', save_best_only=True, verbose=1),
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1)
]

# Training
history = model.fit(
    train_generator,
    steps_per_epoch=nb_train_samples // BATCH_SIZE,  # ví dụ: 75//16
    epochs=epochs,
    callbacks=callbacks_list,
    validation_data=validation_generator,
    verbose=1,
    validation_steps=nb_validation_samples // BATCH_SIZE
)

import matplotlib.pyplot as plt

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs_x = range(1, len(loss_values) + 1)
plt.figure(figsize=(10,10))
plt.subplot(2,1,1)
plt.plot(epochs_x, loss_values, 'r', label='Training loss')
plt.plot(epochs_x, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation Loss and Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.subplot(2,1,2)
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
plt.plot(epochs_x, acc_values, 'r', label='Training acc')
plt.plot(epochs_x, val_acc_values, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()
plt.show()

# Load the best saved model
load_model = keras.models.load_model
model = load_model('model.keras')
# Using the validation dataset
score = model.evaluate_generator(validation_generator)
print('Val loss:', score[0])
print('Val accuracy:', score[1])

# Using the test dataset
score = model.evaluate_generator(test_generator)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
import itertools

#Plot the confusion matrix. Set Normalize = True/False
def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(10,10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, decimals=2)
        cm[np.isnan(cm)] = 0.0
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
# Some reports
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

#On test dataset
Y_pred = model.predict_generator(test_generator)
y_pred = np.argmax(Y_pred, axis=1)
target_names = classes

#Confution Matrix
cm = confusion_matrix(test_generator.classes, y_pred)
plot_confusion_matrix(cm, target_names, normalize=False, title='Confusion Matrix')

#Classification Report
print('Classification Report')
print(classification_report(test_generator.classes, y_pred, target_names=target_names))