# Clothing Classifier

This program aims to identify the type of apparel shown in an image. 

## Description

This project utilizes a pre-trained model, the ResNet-18, to implement the Image Classification model on a custom dataset (https://www.kaggle.com/datasets/trolukovich/apparel-images-dataset). 
The program can identify 5 types of clothing: shirts, shorts, dresses, pants, and shoes. Given an image, the program predicts what type of clothing it is, and outputs it.

If further developed with more classes and a larger dataset, the program can be used for automatically categorizing catalogs in online stores, and or automatically sorting clothing.

## Getting Started

### Dependencies

* Jetson Nano
* python3, jetson-inference, and jetson-utils libraries

### Installing

* Ssh into your nano
* Clone this github repo onto your nano using the following command in terminal: ```git clone https://github.com/5ketch/clothing-classification.git```

### Executing program

* In your nano, cd into the "clothing-classification" directory
* Run the python file using ```python3 clothing_classifier.py INPUT_PATH``` where INPUT_PATH is replaced by the directory of the images you want to pass through the program (To use the test images in the project, use the directory "data/test")

### Output

* The program makes a new directory if it doesn't already exist called "outputs"
* In this directory, there will sub-directories corresponding to each clothing type (shirts, shorts, dress, shoes, and pants)
* The output images will automatically be sorted into the directory that the program classified it as

## Video Example

* [Video explanation on how to run the program](https://youtu.be/UzM6hOUiCYE)
