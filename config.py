from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *

image_size = (50, 50)
input_shape = (50, 50, 3)
whether_to_generator = True
train_split_proportion = 0.1
class_number = 10
steps_per_epoch = 32
batch_size = 32
epochs = 1000