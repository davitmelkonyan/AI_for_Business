from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam

class Brain(object): #input, 2 hidden(first with 64 & second with 32 neurons) & output layers
    def __init__(self, learning_rate = 0.001, number_actions = 5):
        self.learning_rate = learning_rate
        states = Input(shape=(3,))#vector of 3rows and only 1 col
        x = Dense(units=64, activation= 'sigmoid')(states)
        x = Dropout(rate = 0.1)(x) #dropout is regularization technique to prevent overfitting-deactivates certain rate of random neurons during each step of forw/back propagation
        y = Dense(units=32, activation= 'sigmoid')(x)
        y = Dropout(rate = 0.1)(y) #i.e 10% of neurons will be randomly deactivated
        q_values = Dense(units=number_actions, activation = 'softmax')(y)#recommended for output layer
        self.model = Model(inputs = states, outputs = q_values)
        self.model.compile(loss= 'mse', optimizer= Adam(lr = learning_rate))#mean square errir because -> predition regression -> returning continous numerical outcome
