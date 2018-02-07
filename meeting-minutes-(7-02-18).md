### Meeting Minutes (07-02-18)

**What Was Discussed:**

* Potential proposed similarity metric: uses one-shot learning approach.
* We could state assumption about depth of network and its relation to the similarity of instances of data within each dataset. That is to say: similarity between faces is subtle and the similarity between clothes is obvious. Furthermore, we could abstract out the idea of using a methodology in order to provide an empirical similarity metric to prove how similar datasets are compared with one another to our future work section.
* In order to figure out whether we need a backup plan or not, we need to assess our methodology and perform a risk assessment. We need to state any changes to the objectives, the risks and whether or not we need a backup plan.
* Methodology: preprocess the data, such that any input from either dataset has the same dimensionality, use three conv layers, drop out, max pooling and ReLU activations.
* Document the loss function: cross-entropy error.
* Evaluation: classification accuracy (generalisation using K-fold validation).

**Data:**

* Two datasets: Faces and Clothes.
* Poses: front, left, right, up, down, 30k examples in total, unbalanced.
* Clothes: t-shirt, pants, flip-flops, shoes, bag, 30k examples in total, balanced.

**IMPORTANT: Make sure we state the exact number of training examples for each class.**

**IMPORTANT: State a risk: due to the unbalanced nature of one of our training sets (poses), this could be an issue.**

**Moving Forwards (for Friday and beyond):**

* Jonathon: Document suggested readings posted by Steve.
* Jonathon, Sebastian: Read suggested reading posted Steve.
* All: Continue working towards writing interim report.
* Sebastian: Finish up working with data, make analysis of number of parameters.
* Steve: Play around with CNN architectures with data on GitHub.