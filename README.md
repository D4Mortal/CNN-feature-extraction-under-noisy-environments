To run Main.py or Baseline.py you will need the following packages:
Python 3.6
numpy
pytorch
matplotlib
pandas

This is CNN network trained on the "Static Facial Expressions in the Wild" dataset, which contains captured images of actor experssions in movies.
It classifies the images to one of seven expression classes: angry, disgust, fear, happy, sad, surprise and neutral.

This repository does not contain the SFEW dataset.

Main.py contains the main program executing 10 fold cross validation on the SFEW dataset, to run Main.py there must be a folder named "Subset For SFEW" placed under the same directory containing all the image files sorted in sub folders by expressions.

CNN_Functions.py contains the implementation of a Convolutional neural network and k fold validation functions, it does not execute any actions.

Baseline.py contains the implementation of a standard multi layer neural network, there are no additional needed arguements for execution, however the data file "SFEW_processed.xlsx" must be under the same directory to run. 

More details are described in the report.


