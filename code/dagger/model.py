import sys
import os.path
# To import from sibling directory ../utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from tensorflow import keras
import tensorflow as tf


def create_model(number_filters, length_state, h1=2048, h2=1024, h3 =512, h4 = 256):
        # loss: 1.3718 - acc: 0.6029 - val_loss: 1.5231 - val_acc: 0.6466
        # epoch 28
        # best h1256_h2128_ts1543838439
        # Define architecture of the model
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
        out = keras.layers.Lambda(lambda x: x[0]*x[1])([probas, mask_input]) #have to apply the mask AFTER softmax
        # Wrap in Keras model
        model =  keras.Model(inputs=[main_input, mask_input], outputs=out)
        # Compile the model
        model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.001), 
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
        return model
