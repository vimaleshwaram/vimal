import pickle
import json
from flask import Flask, request, jsonify
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from spellchecker import SpellChecker

app = Flask(_name_)

# Load your dialog dataset with 'latin-1' encoding
dataset = pd.read_csv('/home/gokul3002001/mean/ab.csv', encoding='latin-1')

# Preprocess the dataset
tokenizer = Tokenizer()
tokenizer.fit_on_texts(dataset['a'])
total_words = len(tokenizer.word_index) + 1

# Tokenize and pad the sequences
input_sequences = []
for line in dataset['a']:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        input_sequences.append(n_gram_sequence)

max_sequence_length = max([len(x) for x in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')

# Separate input and target sequences
X = input_sequences[:, :-1]
y = input_sequences[:, -1]

# Create and compile the model
model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_length - 1))
model add(LSTM(150))
model add(Dense(total_words, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Train the model with a specified number of epochs
model.fit(X, y, epochs=10)  # You can specify the number of epochs here

# Save the trained model to a pkl file
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Initialize a spell checker
spell = SpellChecker()

@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_input = request.json['user_input']  # Assuming you're receiving JSON input

    # Load the trained model from the pkl file
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    # Preprocess user input
    user_input = user_input.lower()  # Convert to lowercase
    user_input = spell.correction(user_input)  # Correct spelling
    input_sequence = tokenizer.texts_to_sequences([user_input])[0]
    input_sequence = pad_sequences([input_sequence], maxlen=max_sequence_length - 1, padding='pre')

    # Generate a response using the trained model
    response_sequence = []
    for _ in range(max_sequence_length - 1):
        predicted_word_index = model.predict_classes(input_sequence, verbose=0)
        predicted_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                predicted_word = word
                break
        input_sequence = pad_sequences([input_sequence.tolist() + [predicted_word_index]], maxlen=max_sequence_length - 1, padding='pre')
        response_sequence.append(predicted_word)

    response = ' '.join(response_sequence)

    return jsonify({'response': response})

if _name_ == '_main_':
    app.run(debug=True)