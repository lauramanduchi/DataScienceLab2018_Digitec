
""" Data Science Lab Project - FALL 2018
Mélanie Bernhardt - Mélanie Gaillochet - Laura Manduchi

This file defines the architecture of our neural network
for imitation learning. Helper file for dagger_train.py
"""

import sys
import os.path
# To import from sibling directory ../utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from tensorflow import keras
import tensorflow as tf



def create_model(number_filters, length_state, h1=2048, h2=1024, h3=512, h4=256):
        """ This functions defines initializes the model.
        Args:
            number_filters: number of questions available
                            i.e. number of output classes.
            length_state: size of the input one-hot state vector
            h1, h2, h3, h4: size of the hidden layers
        
        Returns:
            model: compiled keras model.
        """
        main_input = keras.layers.Input((length_state,), dtype='float32', name='main_input')
        mask_input = keras.layers.Input(shape=(number_filters,), dtype='float32', name='mask_input')
        x = keras.layers.Dense(h1, activation=tf.nn.relu, input_shape=(length_state,))(main_input)
        x = keras.layers.Dropout(rate=0.5)(x)
        if not h2 == 0:
            x = keras.layers.Dense(h2, activation=tf.nn.relu, input_shape=(length_state,))(x)
            x = keras.layers.Dropout(rate=0.3)(x)
        if not h3 == 0:
            x = keras.layers.Dense(h3, activation=tf.nn.relu, input_shape=(length_state,))(x)
            x = keras.layers.Dropout(rate=0.3)(x)
        if not h4 == 0:
            x = keras.layers.Dense(h4, activation=tf.nn.relu, input_shape=(length_state,))(x)
            x = keras.layers.Dropout(rate=0.3)(x)
        probas =keras.layers.Dense(number_filters, activation=tf.nn.softmax)(x)
        # Have to apply the mask AFTER softmax
        out = keras.layers.Lambda(lambda x: x[0]*x[1])([probas, mask_input]) 
        # Wrap everythin in Keras model
        model =  keras.Model(inputs=[main_input, mask_input], outputs=out)
        # Compile the model
        model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.001), 
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
        return model