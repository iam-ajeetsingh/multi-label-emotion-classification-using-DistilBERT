# Emotion Detection in Text Using Neural Networks and DistilBERT: A Multi-Label Classification Approach

This project implements a multi-label text classification system to detect emotions in text. The system predicts whether any of the following emotions are present in a given text: *admiration*, *amusement*, *gratitude*, *love*, *pride*, *relief*, and *remorse*. The project leverages a neural network architecture enhanced with the pre-trained DistilBERT model and explores various techniques to improve performance.

## 1. Task

The goal of this project is to predict whether any of the following emotions are present in a given text:

- admiration
- amusement
- gratitude
- love
- pride
- relief
- remorse

For example, given the text:

> *Thanks for the reply! I appreciate your input. Please keep me in the loop, I’d love to be more active with this if possible.*

The system should predict the labels: **admiration, gratitude, love**.

## 2. Data Format

The data is provided in CSV format. The first row is the header, which includes the column names: "text" for the input text, followed by the names of the emotions to be detected. Each subsequent row contains a text sample in the first column and binary labels (0 or 1) in the remaining columns, indicating the presence or absence of each emotion.

### Example

The first four rows of `train.csv` look like this:

```csv
text,admiration,amusement,gratitude,love,pride,relief,remorse
My favourite food is anything I didn't have to cook myself.,0,0,0,0,0,0,0
"Now if he does off himself, everyone will think hes having a laugh screwing with people instead of actually dead",0,0,0,0,0,0,0
Yes I heard abt the f bombs! That has to be why. Thanks for your reply:) until then hubby and I will anxiously wait 😝,0,0,1,0,0,0,0
```

For example, the third entry in train dataset contains the text:

Yes I heard abt the f bombs! That has to be why. Thanks for your reply:) until then hubby and I will anxiously wait 😝

This text is labeled with the emotion: gratitude.


### Dataset Details
The dataset is divided into three parts:

- **Training data**: 80% of the dataset (25,196 examples)
- **Validation data**: 10% of the dataset (3,149 examples)
- **Test data**: 10% of the dataset (3,149 examples)
The test data is unlabeled, with all label columns containing 0s. The task is to predict the labels for the test data.

### Submission Format
submission should be a ZIP file containing a single CSV file. The CSV file should have the same format as the training data, with the text in the first column and your model's predictions (0 or 1) for each emotion in the remaining columns. The file should be named submission.zip.


## 3. Project Implementation

### Features

- **DistilBERT Integration**: Utilizes the pre-trained DistilBERT model for better contextual understanding of text.
- **Custom Neural Network Layers**: Includes additional dense layers with dropout and batch normalization for improved performance.
- **Learning Rate Scheduler**: Dynamically adjusts the learning rate during training for optimal convergence.
- **Advanced Callbacks**: Implements early stopping, model checkpointing, and learning rate reduction on plateau.
- **Micro F1-Score Optimization**: Achieved a best micro F1-Score of **0.85495** on the validation set.

### Results

- **Fully Connected Neural Network**: Achieved a micro F1-Score of **0.82703** after extensive hyperparameter tuning.
- **DistilBERT Model**: Improved the micro F1-Score to **0.85495** by leveraging the pre-trained DistilBERT model and additional optimizations.

## 4. Setup Instructions

1. Clone the repository and navigate to the project directory.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Prepare your dataset in CSV format with the following structure:
- The first column should contain the text data.
- Subsequent columns should contain binary labels (0 or 1) for each emotion.

## 5. Acknowledgments

- Hugging Face Transformers library for providing the DistilBERT model.
- TensorFlow for enabling the creation and training of the neural network.
- INFO557 instructor for providing the dataset and guidance.
   
