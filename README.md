# Unbiased recruiting

## Context and goal

Machine Learning algorithms are more and more used to match job applicants with job offers. One of the arguments used by Machine Learning proponents is that algorithms are more objective than humans but is it really the case? We have seen recent scandals involving recruiting algorithms discriminating against women - Amazon 2018. The question is then how do we make recruiting algorithms genderless?

This project aims at making sure that CV s representations do not contain any gender component  stemming from linguistic characteristics or text syntax but do contain the information necessary to match it to a relevant job offer. 

## High level explanation

To do so we use two neuronal networks trained with job applicants' CVs. Those CVs have been parsed into texts.

The first neuronal network is an autoencoder. We train a network to reproduce the text that has been fed to it. The text is beeing first encoded into a vector of smaller size than the initial tokenized texts. The encoded vector is then decoded. The second neuronal network is a binary classifier whose goal is to determine the gender (male or female here) after encoding of CV. 

We design a custom loss including two variable:

* A reconstruction loss, which is the classical autoencoder loss comparing the input to the decoded output
* The classifier loss

$$ Loss_{Adversarial} = Loss_{Autoencoder} - \beta*  Loss_{Classifier} $$

Here we want our networks to work together. Finding out the gender 

## Requirements

