# ASL Fingerspelling Recognition with TensorFlow

Welcome to the ASL Fingerspelling Recognition project, an exploration into the realms of American Sign Language using TensorFlow. This GitHub repository houses the tools and insights needed to dive into the world of deep learning and the interpretation of ASL fingerspelling through advanced models.

## Overview

This project is centered around training a cutting-edge Transformer model to recognize and interpret American Sign Language fingerspelling from a specialized dataset. The primary goal is to predict and translate ASL fingerspelling captured in video frames into textual phrases, pushing the boundaries of machine learning in sign language interpretation.

## Project Structure

1. **Data Loading:**
   - Understand the dataset structure and learn how to load it efficiently for model training.

2. **Data Preprocessing:**
   - Convert raw data into TensorFlow Records (tfrecords) format for improved training speed and efficiency.

3. **Model Training:**
   - Dive into the training process of a Transformer model designed for ASL fingerspelling recognition. Explore the architecture and witness the model learning the intricacies of sign language.

4. **Hyperparameter Tuning:**
   - Fine-tune the model's hyperparameters to enhance its performance. Utilize techniques such as RandomSearch to search through the hyperparameter space and find the optimal configuration for your Transformer model.

## Getting Started

### Prerequisites

- [Python](https://www.python.org/) (>=3.6)
- [TensorFlow](https://www.tensorflow.org/) (>=2.0)

3. Navigate through the notebooks in the `main/` directory to understand and run the project.
## Output

After successfully training the ASL Fingerspelling Recognition model, you can use it to predict and translate ASL fingerspelling from new video frames. Here's a snippet of Output

![Screenshot](https://github.com/jaickeyminj/Building-a-Sign-Language-Model-/blob/main/final_training_validation_accuracy_plot.png)

![Screenshot](https://github.com/jaickeyminj/Building-a-Sign-Language-Model-/blob/main/final_training_validation_loss_plot.png)

## Accurany and Prediction
![Screenshot](https://github.com/jaickeyminj/Building-a-Sign-Language-Model-/blob/main/Prediction_Accuracy.png)
![Screenshot 2023-11-28 225722](https://github.com/jaickeyminj/Building-a-Sign-Language-Model-/assets/76790652/979bc92f-9b9c-4177-bccc-83a2b45b4726)

## Running
Before running the `python main.py` we need to change change path at line 31 and 33. We also need to create two folders named as `land` and `tf` to store our parquet files and tfrecords.`templates` folder contains the index.html to demonstrate a simple UI interface for user to interact. Sample image of UI is given below.
Link - `http://127.0.0.1:5000/`

![Screenshot](https://github.com/jaickeyminj/Building-a-Sign-Language-Model-/blob/main/UI_Image.png)


## Contribution
Feel free to contribute to the project by sharing insights, optimizations, or additional features. Create issues, provide feedback.

## Acknowledgments

Special thanks to [Google](https://www.google.com/) for providing the ASL Fingerspelling Recognition dataset and fostering innovation in machine learning.

Special thanks to the [Keras team](https://keras.io/examples/audio/transformer_asr/) for their valuable insights and resources on Transformer models in audio processing.




