from tensorflow.keras.models import load_model

# Load the model without compiling
model = load_model('../Models/valid_model.h5', compile=False)

# Get the input shape from the model's first layer
input_shape = model.input_shape
n_features = input_shape[1]
print(f'n_features: {n_features}')
