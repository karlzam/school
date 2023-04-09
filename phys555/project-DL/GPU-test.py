import torch

#print(torch.cuda.is_available())
#a=torch.cuda.FloatTensor()

#import keras
import tensorflow as tf
tf.test.gpu_device_name()
print(tf.config.list_physical_devices('GPU'))
print(torch.version.cuda)
