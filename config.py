from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *

image_size = (250, 250)
input_shape = (250, 250, 3)
whether_to_generator = False
train_split_proportion = 0.9
class_number = 10
steps_per_epoch = 32
batch_size = 32
epochs = 100