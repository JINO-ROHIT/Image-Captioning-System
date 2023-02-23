# Image Captioning System

## Introduction

This project aims to generate captions for the images that users upload.
    
## How to run the code
Clone the Repository and extract the files

1. Go to the folder where app.py file is present
2. Type cmd in the file path 
3. Run the code with command `python app.py`
4. Upload your image
5. Select how many captions you want to generate
6. Now predict to see the captions generated!

## Training details

I have used a very simple resnet10t backbone and a 2 layered lstm to predict the captions. The architecture was trained on the flick30
dataset for 20 epochs,  and metrics logged in weights and biases. The application was deployed on flask.

