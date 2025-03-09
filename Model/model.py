import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the extracted features from CSV
df = pd.read_csv('../audio_features.csv')


# Convert the features column (which contains lists) into a numpy array
# Combine all 40 MFCC features into a single array
df['features'] = df.iloc[:, :-1].values.tolist()



# Extract features and labels
X = np.vstack(df['features'].values)
y = df['label'].values

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Build the Deep Learning Model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(40,)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(2, activation='softmax')  # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Save the model
model.save('audio_classifier.h5')

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'✅ Model Accuracy: {accuracy * 100:.2f}%')
