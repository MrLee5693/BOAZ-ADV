import tensorflow as tf
from tensorflow.keras import backend as K

class BiasLayer(tf.keras.layers.Layer):
    def __init__(self, units, *args, **kwargs):
        super(BiasLayer, self).__init__(*args, **kwargs)
        self.units = units
        self.bias = self.add_weight('bias',
                                    shape=[self.units],
                                    initializer='zeros',
                                    trainable=True)

    def call(self,x):
        return self.bias

    def get_config(self):

        config = super().get_config().copy()
        config.update({
          'units': self.units
        })
        return config

def bias_dropout(input,drop_rate):
    training = K.learning_phase()
    if training is 1 or training is True:
        input *= (1 - drop_rate)    
    return input

def dropout(input,drop_rate):
    training = K.learning_phase()
    if training is 1 or training is True:
        input *= (1 - drop_rate)    
        input += (drop_rate * K.random_normal(K.shape(input), dtype='float32'))   
    
    return input