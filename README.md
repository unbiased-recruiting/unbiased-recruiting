# Unbiased recruiting

## Context and goal

Machine Learning algorithms are more and more used to match job applicants with job offers. The question is: Do those new recruiting tools mean a less gender biased recruitment process?

This project aims at making sure that CV vectorial representation does not contain any gender component  stemming from linguistic characteristics or text syntax. In order to do so we use two neuronal networks fed with job applicants' CVs. Those CVs have been parsed into texts.

## High level explanation

The first neuronal network is an autoencoder. We train a network to reproduce the text that has been fed to it. The text is beeing first encoded into a vector of smaller size than the initial tokenized texts. The encoded vector is then decoded. The second neuronal network is a binary classifier whose goal is to determine the gender (male or female here) after encoding of CV. 

We design a custom loss including two variable:
* A reconstruction loss, which is the classical autoencoder loss comparing the input to the decoded output
* The classifier loss

$Adversarial_loss = Reconstruction_loss + /Beta Classifier_loss$

Here we want our networks to work together and establish a tradeof between reconstructing a CV and not learning the gender while learning. 

## Requirements

