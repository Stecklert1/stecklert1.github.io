# Human Vs AI Text Model
## Introduction
In recent years, AI text-generation models have become increasingly powerful and complex, to the point where they generate textual structures more similar to those of humans. This can present several ethical concerns regarding education, work environments, impersonations, and misinformation. As these models advance, these problems will only be exacerbated. The ability to reliably identify and distinguish AI-generated text from human-written text is quickly becoming in demand to maintain authenticity and ensure AI will be used ethically.  

This project's objective is to develop a binary classification model that will be capable of distinguishing AI-generated essays from human-written essays. Specifically, it will leverage the DistilBERT transformer model, a smaller and faster version of BERT, to achieve high accuracy and efficiency in this binary text classification task. ~30000 essays will be introduced for the AI to train and test on. Then, an additional 9 essays of different origins will be introduced to demonstrate the trained model. 

## The Data
The data utilized came from a Kaggle dataset simply called 'Training_Essay_Data.csv.' The original data contained 29145 essays averaging ~500 words per essay. The essays were distributed with 17508 being human-written and 11637 being AI-generated. The essays were written based on multiple prompts, both from the writer and the AI. These essay batches ranged from a few dozen to hundreds from the same prompt. Making the essays have a slight lack of variability. As certain words would appear more often in comparison, the AI may be trained or fixated on these words. 

To demonstrate how well the model was trained, an additional 9 essays were introduced from a different origin. These essays contained 5 AI-generated essays and 4 human-written essays. One of the AI-generated essays was not a conventional essay in order to see if the model would be able to distinguish it. All the essays from this dataset contained ~500-1000 words.

## Architecture

