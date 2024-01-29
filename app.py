from flask import Flask, render_template, request, jsonify
# Replace with the actual import statement for your NLP model
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Parameters for padding and OOV tokens
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
# Number of examples to use for training
training_size = 20000

# Vocabulary size of the tokenizer
vocab_size = 10000

# Maximum length of the padded sequences
max_length = 32

# Output dimensions of the Embedding layer
embedding_dim = 16

app = Flask(__name__)

def load_model(model_path):
    """
    Loads a saved model from a specified path.
    """
    print(f"Loading saved model from{model_path}")
    model = tf.keras.models.load_model(model_path,
                                       custom_objects={"KerasLayer":hub.KerasLayer})
    return model

model = load_model('scrcasm.h5')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_data().decode('utf-8')
        # Now you can parse the data as needed
        # For example, if the data is in the form "input_string=value", you can do:
        input_string = data.split()
        #return jsonify({'result': input_string})
        return  input_string
        # input_string = "former versace store clerk sues over secret 'black code' for minority shoppers"
        #datax = input_string.split()
        # datax = [data]
        tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
        sequences = tokenizer.texts_to_sequences(input_string)
        pad = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

        # Assume your_nlp_model.predict() takes a list of titles and returns predictions
        predictions = model.predict(pad)
        binary_predictions = (predictions > 0.5).astype(int)
        # For simplicity, assuming predictions is a list of 0s and 1s
        result =  binary_predictions

        return render_template('index.html', prediction=result)
    except Exception as e:
        # Handle the error appropriately (log it, return an error response, etc.)
        return jsonify({'error': str(e)}), 40

if __name__ == '__main__':
    app.run(debug=True)
