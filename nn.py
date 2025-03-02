# Note: This code is modified version of the original file provided in the assignment.
# I made the modifications to improve the performance of the model and f1 score.
# The changes made are mentioned below:

#### ********* Initially Used Fully Connected FeedForward Neural Network ********* ####

#1 - Decreased the batch size to 8 from 16 in the train function to improve the performance.
#2 - Added L2 regularization to the model to prevent overfitting.
#3 - Added Dropout and BatchNormalization to the model to improve the performance of the model.
#4 - Added Learning rate scheduler to improve learning.
#5 - Added more epochs with early stopping to improve the performance of the model.
#6 - Tried adding class weights to the model to handle class imbalance but later removed it.
#7 - Added more callbacks like ModelCheckpoint, EarlyStopping and ReduceLROnPlateau.
#8 - Added the compression parameter to the to_csv function to fix the pylint error.
#9 - Added the .keras to the model_path in the train function to fix the error.
#10 - Added the log_dir parameter to the TensorBoard callback to visualize the training process.
#11 - Added the initial_learning_rate, decay_steps, and decay_rate parameters.
#12 - Added the verbose parameter to the callbacks to display the progress of the training process.
#13 - Tried with additional hidden layers to improve the performance.
#14 - Played with learning rate and batch size values to improve the performance.

# After many Iterations of Training with diffrent permutation and combination of hyperparameters
# the best micro F1-Score I got is 0.82703
# The model was trained with 30 epochs and early stopping with patience 5.
# For the best F1-Score I got ,The best epoch was 20th and for next 5 epochs
# it did not improve the score and stopped early.
# I have submitted the prediction on codabench and got the score of 0.827 for the model



#### ********* Finally Used DistilBert pre-trained Model ********* ####

# NOte: I obtained permission from the instructord through e-mail to use the DistilBert model.

#1 - I Used the DistilBert pre-trained model to improve the performance of the model further.
#2 - I Increased the MAX_LENGTH from 64 to 128 to capture more context.
#3 - Added Learning rate scheduler to improve learning.
#4 - Added more callbacks like ModelCheckpoint, EarlyStopping and ReduceLROnPlateau.
#5 - Played with learning rate and batch size values to improve the performance.
#6 - Used the Adam optimizer with a learning rate scheduler to improve learning.
#7 - Tried with additional hidden layers to improve the performance.
#8 - Used the BatchNormalization and Dropout layers to improve the performance of the model.

# After many Iterations of Training with diffrent permutation and combination of hyperparameters
# the best micro F1-Score with DistilBert model I got is 0.85495 which I submitted on codabench.


"""
This module provides functions to train and predict using a simple neural network
model with the Hugging Face Transformers library and TensorFlow. 
"""
#importing the required libraries
import argparse
import gc
import datasets
import pandas
import transformers
import tensorflow as tf
import numpy as np
from transformers import TFDistilBertModel, DistilBertTokenizer


# To Clear any existing session and garbage collection
tf.keras.backend.clear_session()
gc.collect()

# Initializing the DistilBert tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
MAX_LENGTH = 128  # Increased MAX_LENGTH from 64 to 128 to capture more context

# I removed the to_bow function as it is not required for DistilBERT model.

# Creating a helper function to tokenize text.
def tokenize(examples):
    """Tokenize the text using DistilBERT tokenizer"""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_attention_mask=True
    )

# helper function to create a model using DistilBERT with custom classification layers
def create_model(num_labels):
    """Create a model using DistilBERT with custom classification layers"""

    # Loading the pre-trained DistilBERT model from transformers library.
    distilbert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')

    # Creating Input layers for the model to accept input data
    input_ids = tf.keras.layers.Input(shape=(MAX_LENGTH,), dtype=tf.int32, name='input_ids')
    attention_mask = tf.keras.layers.Input(shape=(MAX_LENGTH,),
                                           dtype=tf.int32, name='attention_mask')

    # Adding the DistilBERT layer to the model
    sequence_output = distilbert_model(input_ids=input_ids, attention_mask=attention_mask)[0]

    # Adding a Pooling layer - using mean pooling
    pooled_output = tf.keras.layers.GlobalAveragePooling1D()(sequence_output)

    # Adding a Dense layer with dropout and batch normalization to improve performance
    x = tf.keras.layers.Dense(512)(pooled_output)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    # Adding an Output layer
    outputs = tf.keras.layers.Dense(num_labels, activation='sigmoid')(x)
    #outputs = tf.keras.layers.Dense(num_labels, activation='sigmoid')(pooled_output)

    # Creating the model
    model = tf.keras.Model(
        inputs=[input_ids, attention_mask],
        outputs=outputs
    )
    return model


# added .keras to the model_path in the train function to fix the error.
# helper function to train the model on the specified dataset
def train(model_path="model.keras", train_path="train.csv", dev_path="dev.csv"):
    """Train a DistilBERT model on the specified dataset"""

    print("Loading and preparing data...")

    # Loading the datasets from train and dev CSV files
    hf_dataset = datasets.load_dataset("csv", data_files={
        "train": train_path,
        "validation": dev_path
    })

    # Getting the labels from the hugingface dataset
    labels = hf_dataset["train"].column_names[1:]

    def gather_labels(example):
        """Converts the label columns into a list of 0s and 1s"""
        # the float here is because F1Score requires floats
        return {"labels": [float(example[l]) for l in labels]}

    # Converting to Huggingface datasets and tokenizing it.
    hf_dataset = hf_dataset.map(gather_labels)
    hf_dataset = hf_dataset.map(tokenize, batched=True)

    # Converting to TensorFlow datasets
    train_dataset = hf_dataset["train"].to_tf_dataset(
        columns=['input_ids', 'attention_mask'],
        label_cols="labels",
        batch_size=32, # tried batch size of 16 and 64 also but 32 is best.
        shuffle=True
    )

    dev_dataset = hf_dataset["validation"].to_tf_dataset(
        columns=['input_ids', 'attention_mask'],
        label_cols="labels",
        batch_size=32   # tried batch size of 16 and 64 also but 32 is best.
    )

    print("Creating and compiling model...")

    # Creating the model
    model = create_model(len(labels))

    # adding Learning rate scheduler to improve learning
    initial_learning_rate = 2e-5
    decay_steps = 1000
    decay_rate = 0.9
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate
    )

    # Compiling model with additional metrics and learning rate schedular.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[
            tf.keras.metrics.F1Score(average="micro", threshold=0.5),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall()
        ]
    )

    # Enhancing the model with callbacks
    callbacks = [

        # adding model checkpoint to save only the best model based on F1 score.
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_f1_score",
            mode="max",
            save_best_only=True,
            verbose=1
        ),

        # adding early stopping to prevent overfitting
        tf.keras.callbacks.EarlyStopping(
            monitor='val_f1_score',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),

        # Reducing learning rate on plateau to improve learning
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,             # tried LEarning rate 0.1, 0.3, and 0.5 but 0.2 is best.
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # Training the  model
    print("Training model...")
    model.fit(
        train_dataset,
        epochs=5,
        validation_data=dev_dataset,
        # removed class_weights
        callbacks=callbacks,
        verbose=1
    )

    # Save the model with proper configuration
    #model.save(model_path, save_format='tf')
    #print(f"Model saved to {model_path}")

    print("Model trained successfully ......")




# helper function to generate predictions using the trained model.
def predict(model_path="model.keras", input_path="dev.csv"):
    """Generate predictions using a trained model and save them to a CSV file"""

    # Loading the trained model saved in the training step
    model = tf.keras.models.load_model(model_path, custom_objects={"TFDistilBertModel":
                                                                   transformers.TFDistilBertModel})

    # Loading the data read from csv file for prediction
    df = pandas.read_csv(input_path)

    # Creating Huggingface dataset from Pandas data frame and tokenizing it.
    hf_dataset = datasets.Dataset.from_pandas(df)
    hf_dataset = hf_dataset.map(tokenize, batched=True)

    # Creating tensorflow dataset from Huggingface dataset
    tf_dataset = hf_dataset.to_tf_dataset(
        columns=['input_ids', 'attention_mask'],
        batch_size=16
    )

    # Generating predictions from model
    predictions = np.where(model.predict(tf_dataset) > 0.5, 1, 0)

    # assigning predictions to label columns in Pandas data frame
    df.iloc[:, 1:] = predictions

    # Saving predictions to a zipped CSV file.
    df.to_csv("submission.zip", index=False, compression={'method': 'zip',
                                                          'archive_name': 'submission.csv'})

if __name__ == "__main__":
    # parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices={"train", "predict"})
    args = parser.parse_args()

    # call either train() or predict()
    globals()[args.command]()
