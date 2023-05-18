
import keras
import tensorflow as tf
tf.test.gpu_device_name()
tf.config.list_physical_devices('GPU')

import torch
print(torch.cuda.is_available())
a=torch.cuda.FloatTensor()
print(torch.version.cuda)