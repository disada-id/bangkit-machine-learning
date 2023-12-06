import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import ModelCheckpoint
import os
import matplotlib.pyplot as plt

# Define a custom callback for model checkpointing
class MyCallback(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_path):
        super(MyCallback, self).__init__()
        self.checkpoint_path = checkpoint_path
        self.best_val_acc = 0.0

    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs.get('val_accuracy')
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.model.save(self.checkpoint_path)

# Function to preprocess the dataset
def preprocess_dataset(df):
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    return X, y

# Function to create the neural network model
def create_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Dense(512, activation='relu', input_shape=(input_shape,), kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to train the model
def train_model(X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
    input_dim = X_train.shape[1]
    output_dim = len(np.unique(y_train))

    model = create_model(input_dim, output_dim)

    checkpoint_path = 'save_model/checkpoint.h5'
    my_callback = MyCallback(checkpoint_path)
    checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True)

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_val, y_val), callbacks=[checkpoint, my_callback])

    return model, history

# Function to evaluate the model
def evaluate_model(model, X_test, y_test, label_names):
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = y_test

    accuracy = accuracy_score(y_true, y_pred)
    class_report = classification_report(y_true, y_pred, target_names=label_names)

    return accuracy, class_report

# Function to plot the training history
def plot_training_history(history):
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Function to save the trained model
def save_models(model, output_folder='saved_model'):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    model.save(os.path.join(output_folder, 'saved_model.h5'))

    model_json = model.to_json()
    with open(os.path.join(output_folder, 'saved_model.json'), 'w') as json_file:
        json_file.write(model_json)

# Main function to execute the training and evaluation
def main():
    # Read the dataset
    df = pd.read_csv('...')
    X, y = preprocess_dataset(df)

    # Split the dataset into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Train the model and get training history
    model, history = train_model(X_train, y_train, X_val, y_val, epochs=200)

    # Display model summary
    model.summary()

    # Load the best model from checkpoint and save it
    best_model = tf.keras.models.load_model('save_model/checkpoint.h5')
    save_models(best_model, output_folder='save_model')

    # Evaluate the model on the test set
    label_names = [str(i) for i in range(len(np.unique(y)))]
    accuracy, class_report = evaluate_model(best_model, X_test, y_test, label_names)
    print(f'Test Accuracy: {accuracy}')
    print('Classification Report:')
    print(class_report)

    # Plot training history
    plot_training_history(history)

if __name__ == "__main__":
    main()
