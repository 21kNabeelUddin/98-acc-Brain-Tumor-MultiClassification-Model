# Brain Tumor MRI Classification

This project is focused on classifying brain tumor images using a deep learning model. The dataset used for this project is the Brain Tumor MRI Dataset, which consists of four classes: Pituitary, No Tumor, Meningioma, and Glioma. The goal of this project is to predict whether a patient has one of these four types of brain tumors based on MRI scan images.

## Credits

This dataset was made available by Tom Beckert on Kaggle. A big thanks to him for providing this dataset and enabling us to explore and build a machine learning model for this important problem.

Dataset Description
The Brain Tumor MRI Dataset contains images of brain MRI scans categorized into four types of tumors:

Pituitary

No Tumor

Meningioma 

Glioma 

## 1. Data Preprocessing and Augmentation

Resizing: All images were resized to 299x299 pixels to match the input size required by the Xception model.
Data Augmentation: To increase dataset diversity and prevent overfitting, several augmentation techniques were applied:
Rotation: Images were rotated by up to 30 degrees.
Shifting: Images were shifted horizontally and vertically by 20%.
Zooming: Zoomed in on images by 20%.
Shearing: Applied slight image shearing.
Horizontal Flip: Randomly flipped images horizontally.
Reason: Augmentation helps the model generalize better, improving performance on new data and reducing overfitting.

## 2. Model Architecture

Base Model: The Xception model, pre-trained on ImageNet, was used as a feature extractor for better performance.
Pooling: Combined Global Average Pooling and Global Max Pooling for better feature extraction.
Custom Layers:
Dropout: Added to reduce overfitting.
Batch Normalization: Used to speed up training.
L2 Regularization: Applied to help generalize the model.
Reason: Xception is a powerful model for image classification, and using pre-trained weights improves accuracy.

## 3. Model Training

Learning Rate Scheduling: Gradually reduced the learning rate to improve convergence.
Optimizer: Adam optimizer was used to adjust model weights during training.
Callbacks:
Early Stopping: Stopped training early if validation loss stopped improving.
Model Checkpointing: Saved the best model weights for later use.
Reason: Learning rate scheduling helps the model converge better, and callbacks prevent overfitting while saving the best model.

## 4. Model Evaluation

Metrics: Accuracy, Precision, and Recall were used to evaluate the model’s performance.
Confusion Matrix: Used to visualize the model's predictions.
Reason: These metrics give a more complete view of model performance, especially for rare classes in medical images.

## 5. Testing and Final Evaluation

The best model, saved during training, was used to evaluate the test dataset and compared with the validation results to check generalization.
How to Run the Project

Clone the repository:
```bash
git clone https://github.com/your-username/brain-tumor-mri-classification.git
```

```bash
Install necessary dependencies: Create a virtual environment and install the required libraries using requirements.txt:
pip install -r requirements.txt
```

Run the Jupyter notebook: Open the notebook in Jupyter and execute the code cells step by step.

Use the trained model: The trained model is saved in the .keras format. You can load the model and use it for prediction on new MRI scans:

from tensorflow.keras.models import load_model
model = load_model('best_model.keras')

Requirements
Python 3.6+
TensorFlow 2.x
Keras
OpenCV
Matplotlib
Pandas
NumPy
Scikit-learn
