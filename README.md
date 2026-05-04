# Human Vs AI Text Model
## Introduction
In recent years, AI text-generation models have become increasingly powerful and complex, to the point where they generate textual structures more similar to those of humans. This can present several ethical concerns regarding education, work environments, impersonations, and misinformation. As these models advance, these problems will only be exacerbated. The ability to reliably identify and distinguish AI-generated text from human-written text is quickly becoming in demand to maintain authenticity and ensure AI will be used ethically.  

This project's primary objective is to develop a binary classification model that will be capable of distinguishing AI-generated essays from human-written essays. Specifically, it will leverage the DistilBERT transformer model, a smaller and faster version of BERT, to achieve high accuracy and efficiency in this binary text classification task. ~30000 essays will be introduced for the AI to train and test on. Then, an additional 9 essays of different origins will be introduced to demonstrate the trained model. 

## The Data
The data utilized came from a Kaggle dataset simply called 'Training_Essay_Data.csv.' The original data contained 29145 essays averaging ~500 words per essay. The essays were distributed with 17508 being human-written and 11637 being AI-generated. The essays were written based on multiple prompts, both from the writer and the AI. These essay batches ranged from a few dozen to hundreds from the same prompt. Making the essays have a slight lack of variability. As certain words would appear more often in comparison, the AI may be trained or fixated on these words. 

To demonstrate how well the model was trained, an additional 9 essays were introduced from a different origin. These essays contained 5 AI-generated essays and 4 human-written essays. One of the AI-generated essays was not a conventional essay in order to see if the model would be able to distinguish it. All the essays from this dataset contained ~500-1000 words.

## Architecture

This is a general visualization of the architecture of the model:
<img width="1099" height="158" alt="image" src="https://github.com/user-attachments/assets/5b5581d3-24dd-4db2-acf3-03e60ef34bb4" />

As above, this model is designed to perform a downstream task. Taking the input from the text of the essay and converting it into tokens,

<img width="992" height="432" alt="image" src="https://github.com/user-attachments/assets/c67cd33a-fdb1-4178-9668-f4d9f7d4c344" />

Every word in the input text is broken up into segments of words that are called tokens; those tokens are then given specific IDs and fed forward to be embedded into actual vectors. Let's expand to an example that gives the full process of tokenization:
<img width="1400" height="450" alt="image" src="https://github.com/user-attachments/assets/f7a87d6b-bfde-45c2-b22f-b1e1704f726a" />

It is from this point that all embeddings of the tokenized text will be fed forward to the DistilBERT transformer model. DistilBERT is a 'distilled' version of the BERT model. The original BERT model contains 12 layers and around 110 million parameters. It is trained to use two unsupervised prediction tasks: Mask Language Model (MLM) and Next Sentence Prediction (NSP). MLM is used to predict masked words in a sentence, while NSP is used to predict if two sentences follow each other. It is with this that BERT achieves maximum performance while doing various NLP tasks. 

In contrast, DistilBERT contains only 6 layers and around 60 million parameters. Rather than using unsupervised training, it uses distilled knowledge from the BERT model. The larger BERT model is used to supervise the training of the DistilBERT. The DistilBERT model learns to mimic the BERT model's behavior and probabilities, retaining most of its performance while being significantly smaller. DistilBERT also retains about 97% of BERT's language understanding capabilities while being 40% smaller and 60% faster at inference time. It also does not have the NSP objective during its training. It instead primarily focuses on the MLM loss and cosine embedding loss (to align hidden state representations) from the teacher model. This means it might be slightly less performant on tasks that heavily rely on understanding sentence relationships, but this is often a minor trade-off for its efficiency gains.

 <img width="850" height="412" alt="image" src="https://github.com/user-attachments/assets/78d157d4-bfe5-480e-af91-be56364d9a77" />

The visualization above gives an example of the internal mechanics of transformer layers, the structural difference between BERT and DistilBERT, and the distillation process that enables model compression while preserving performance. In this model, the Multi-Head attention layer means that the attention mechanism is run multiple times in parallel for each token in the input sequence. The self-attention mechanism allows the model to weigh the importance of all other tokens in the sequence when processing that specific token. It effectively helps the model understand the context and relationships between words in a sentence. Within the attention mechanism, there are 3 linear layers, as defined as q_lin, k_lin, and v_lin, that input the embeddings into, respectively, the Query, Key, and Value vectors. These are used to calculate the attention output. 

<img width="465" height="132" alt="image" src="https://github.com/user-attachments/assets/f8ad61e7-68f1-4909-8754-cd522f414e98" />

It then undergoes a layer normalization, and then proceeding that, there is a Feed-Forward Network (FFN) that follows the attention layer. It is applied independently to each position in the sequence. This FFN typically consists of two linear transformations with a non-linear activation function in between. Its purpose is to perform further transformations on the representation generated by the attention mechanism. The FFN linear layer expands from 768 to 3072 and then outputs back to 768 in the second layer within it. It then has a GELU activation. There is then another normalization layer that is applied after the residual connection. This layer stabilizes the training process and helps the model learn faster.

<img width="454" height="127" alt="image" src="https://github.com/user-attachments/assets/56b8abb7-66a4-4deb-9f17-88a4e28c4bd3" />

There is then a pre-classifier layer to act as an intermediate between the transformer layer and the classifier layer. It is a dense layer that is meant to pass through, specifically, the hidden state vector of the token after it has potentially passed through a dropout layer. It takes the 768 features from the token's hidden state and transforms them into another vector of 768 features. This transformation involves multiplying the input vector by a weight matrix and adding a bias vector. Its output will then serve as the classification layer's input. 


<img width="420" height="27" alt="image" src="https://github.com/user-attachments/assets/745ae238-3c0c-4312-8a00-72c26d1430f5" />

The classifier is the final layer in the model. It is a standard linear layer that maps the processes to the final output classes. The classifier will take the 768 inputs from the pre-classifier and will produce two outputs. One of those outputs will correspond with the classes: Either AI-Generated or Human-Written. The entire purpose of this layer is to convert the abstract essays that had been derived from the DistilBERT and refined by the pre-classifier layer into the prediction scores for each class. It directly learns to distinguish between the essay types based on the features it receives. The output of this layer will be a vector of two raw logits. To get meaningful probabilities for each class, these logits are typically passed through a softmax activation function. The softmax function converts the logits into a probability distribution, where each value is between 0 and 1, and all values sum up to 1. The class with the higher probability after softmax is the model's final predicted class for the essay.

## Prediction Probabilities

<img width="1389" height="590" alt="image" src="https://github.com/user-attachments/assets/2186dd91-a9f1-499d-a955-64bc64d87e1b" />

This is a given visualization that demonstrates that the model is highly confident that it will be able to predict and classify an AI-generated Essay. However, it also makes the same prediction for Human-Generated Essays; the model incorrectly classified this example as 'AI-generated'. Therefore, the bar for 'AI-generated' is high, and the bar for 'Human-written' is low for this model. While the model correctly identifies the pre-defined AI-generated example, it seems to misclassify the human-written example as AI-generated. This highlights that while the overall training metrics (accuracy, F1-score) are high, there might be specific types of human-written text that the model struggles with, or that the examples themselves are quite close to what the model has learned as 'AI-generated'.








