
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 🔹 Step 1: Load Dataset

data = pd.DataFrame({
    'text': [
        "I am so happy today",
        "I hate this weather",
        "It is an okay day",
        "I love my friends",
        "I am sad and tired"
    ],
    'sentiment': ['positive', 'negative', 'neutral', 'positive', 'negative']
})

print("Sample Data:")
print(data.head())


le = LabelEncoder()
data['label'] = le.fit_transform(data['sentiment'])
print("\nLabel Mapping:")
for i, class_name in enumerate(le.classes_):
    print(i, "->", class_name)

tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(data['text'])
sequences = tokenizer.texts_to_sequences(data['text'])
padded = pad_sequences(sequences, padding='post', maxlen=10)

#  Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    padded, data['label'], test_size=0.2, random_state=42
)

# Build Model
model = Sequential([
    Embedding(input_dim=1000, output_dim=16, input_length=10),
    LSTM(32),
    Dense(16, activation='relu'),
    Dense(3, activation='softmax')  
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 🔹 Step 6: Train Model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), verbose=2)

# 🔹 Step 7: Test Sample Messages
def predict_emoji(text):
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=10, padding='post')
    pred = np.argmax(model.predict(pad), axis=1)[0]
    sentiment = le.inverse_transform([pred])[0]
    emoji_map = {'positive': '😊', 'neutral': '😐', 'negative': '😢'}
    print(f"Text: {text}\nPredicted Sentiment: {sentiment}, Emoji: {emoji_map[sentiment]}\n")

#  prediction E.G
predict_emoji("I am so excited!")
predict_emoji("I feel terrible")
predict_emoji("It is just a normal day")
