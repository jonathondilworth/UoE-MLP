### Draft Interim Report

**Abstract**

*THIS IS SOMETHING YOU TYPICALLY WRITE LAST*

**1. Introduction and Motivation**

*THIS IS SOMETHING YOU TYPICALLY WRITE LAST*

**1.1. Motivation**

*TODO: PERHAPS WRITE MOTIVATION BEFORE WRITING THE INTRODUCTION*

*Should these be two separate sections?*

*Taken from previous write up: An investigation into viable techniques to be employed in order to boost performance of smaller datasets within the domain of neural architectures is the primary motivation behind the aforementioned project proposals.*

*TODO: Add more information relating to our newest proposed idea: Investigating how similarity between instances of data in each proposed dataset can affect the performance of a CNN.*

*Note: Whilst writing introduction, embed semi-literature review into the introduction and motivation by citing relevant papers. i.e: neural network architectures typically need large amounts of data to perform well.*

**2. Research Questions**

Within this section two sets of research questions are presented. Firstly, a set of research questions that are addressed within this report is provided (§2.1). Secondly, a set of future research questions to be addressed within the concluding report is offered (§2.2). Thereafter, the aims and objectives of the proposed research questions are probed (§2.3). Finally, a set of hypotheses is given (§2.4).

**2.1. Interim Research Questions**

Using the methodology outlined within §4.x. the following research questions are investigated within this report:

1. How do differences in similarity **(NOTE: IT MAY BE BE GOOD TO LINK TO A SECTION THAT DOCUMENTS SOME KIND OF SIMILARITY METRIC BETWEEN DATA INSTANCES WITHIN EACH DATASET)** between datasets affect the performance (generalisation, accuracy and error) of fairly simple convolutional neural network architectures (as outlined in §4.x)?
2. How does reducing the size of a training dataset affect the performance of fairly simple convolutional neural network architectures?

Research question 2 has already been well researched within the literature [INSERT CITATIONS HERE]. However, the question links into our future research questions (§2.2) associated with using techniques {footnote: such as transfer learning and deep feature extraction} to improve the performance of small datasets on neural network architectures. For this reason, it is important that an investigation is conducted in order to establish a more conclusive understanding of the proposed datasets (§3) on our employed architecture, outlined in §4.x..

**2.2. Future Research Questions**

*TODO: write a chaining paragraph to connect 2.1. to 2.2. and introduce the following research questions:*

1. How does applying transfer learning on to deep convolutional neural networks improve the performance of classification tasks?
2. How does transfer learning perform on convolutional neural networks when the size of the dataset used to tune the network is greatly reduced in size?
3. How can deep feature extraction be used in order to improve the performance of small datasets on deep neural network architectures?
4. How does one shot learning perform on small datasets within classification tasks? (if time allows us to research this) *todo: clean up wording of research question*

**2.3. Aims and Objectives**

To try and find a good method of improving performance on the same task using a smaller version of the same dataset in combination with techniques.

*TODO: expand on this, note: it's important to contextualise your research questions in terms of aims and objectives.*

**2.4. Hypotheses**

* Datasets with subtle differences between classes will perform worse on classification tasks than datasets with obvious differences between classes using the proposed convolution neural network architecture (§4.x.).
* Reducing the size of the training dataset will result in worse generalisation using the proposed network architecture (§4.x.).
* Reducing the size of the training dataset will result in an overfitting of the network architecture to the subsampled dataset. **TODO: NOTE: THIS MAY NEED ADDITIONAL CLARIFICATION.**
* *TODO: ADD FURTHER HYPOTHESES*

**3. Data Set and Task**

*TODO: Look at potential similarity metrics that could be used as empirical measurements between instances of data within the same dataset, such that we can compare similarity averages between datasets.*

* http://cvit.iiit.ac.in/projects/IMFDB/ (Faces database)
* https://www.kaggle.com/zalando-research/fashionmnist (Clothes database)

*TODO: include details about how we're going to mop / clean the datasets before usage.*

**4. Methodology**

**TODO**

*Notes*

* Input Layer (are we going to pre-process the input data, such that the input layer is the same for both datasets? i.e: make the images the same dimensionality?)
* Three convolutional layers.
* Do we want to use batch norm, drop out, etc?
* Softmax output layer for multi-class classification.
* Probably a good idea to include a diagram of the architecture.

**5. Baseline Experiments**

*prerequisite: experimental results*

*Notes*

* Train basic CNN (using above methodology §4.) on entire dataset(s).
* Obtain accuracies for baseline results.
* Subsample original dataset, retrain the same network and obtain new accuracies using the smaller dataset(s).

**5.1. Further Experiments**

**TODO**

**6. Interim Conclusions**

*prerequisite: experimental results from §5.*

*Notes*

* Compare accuracies from dataset 1 and dataset 2, make some conclusions about the similarities between datasets.
* Evaluate hypothesis: smaller training datasets provide lower generalisation accuracies on our proposed CNN architecture.

**7. Future Work**

*Notes*

* Take subsamples of the datasets proposed in §3.x.
* Apply techniques such as deep feature extraction and transfer learning (we could try transferring from ImageNet, perhaps?)
* Produce an academic report documenting our findings.

**7.1. Backup Plans**

**TODO**