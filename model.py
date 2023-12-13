# Import necessary libraries
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam

# Function to create the model
def create_model(input_shape, action_space):

    # Define the input layer
    X_input = Input(input_shape)
    X = X_input
    
    # Convolutional layers with ReLU activation
    X = Conv2D(32, 8, strides=(4, 4),padding="valid", input_shape=input_shape, activation="relu", data_format="channels_first")(X)
    X = Conv2D(64, 4, strides=(2, 2),padding="valid", activation="relu", data_format="channels_first")(X)
    X = Conv2D(64, 3, strides=(1, 1),padding="valid", activation="relu", data_format="channels_first")(X)
    
    # Flatten the output for Dense layers
    X = Flatten()(X)
    
    # Dense layers with ReLU activation
    X = Dense(512, activation="relu", kernel_initializer='he_uniform')(X)
    X = Dense(256, activation="relu", kernel_initializer='he_uniform')(X)
    X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X)

    # Output Layer with # of actions: 2 nodes (left, right)
    X = Dense(action_space, activation="linear", kernel_initializer='he_uniform')(X)

    # Create the Keras model
    model = Model(inputs = X_input, outputs = X)

    # Compile the model with Adam optimizer and mean squared error loss
    model.compile(optimizer=Adam(lr=0.00005), loss='mean_squared_error')

    # Print model summary
    model.summary()

    # Return the model
    return model