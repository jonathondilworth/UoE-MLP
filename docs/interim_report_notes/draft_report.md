### Draft Interim Report

(not totally sure which branch was the most recent, as this was a project from last year, created a pull request for what seemed like the most recent branch)

**Abstract**

*WRITE THIS LAST*

**1. Introduction and Motivation**


During the last six years there has been an increase in popularity of connectionist based approaches (specifically, variations on deep neural network architectures) to solving vision based pattern recognition problems [INSERT REFERENCE HERE]. This surge in popularity has been the result of numerous advancements, including an increase in the amount of available training data and compute power [INSERT REFERENCE HERE], as is demonstrated in [INSERT AlexNet, VGGNet, LeNet references here]. Although deep neural networks work well with large amounts of training data [INSERT REFERENCE HERE], the performance of these models typically drops off in situations where only small amounts of training data is available [INSERT REFERENCE TO LEARNING CURVE HERE]. This poses a problem to smaller businesses and organisations who may not have the appropriate amount of data to utilise these emergent technologies. Such a problem motivates the investigation presented within this (and the forthcoming) report that examines the application of fruitful techniques (specifically, transfer learning) to boost the performance of deep neural network architectures using small datasets.

In order to formalise the aforementioned problem, this paper investigates the effects of dramatically reducing the training set (§x.x) size and observes the effects on this has on the accuracy of the proposed deep neural network architecture (§x.x). In addition, this procedure is applied to two comparable datasets, one with subtle differences between classes and the other with apparent differences between classes. These datasets are compared in order to gauge a fuller understanding of how the data itself can also affect the performance of deep neural networks.

Within the remainder of this paper a set of research questions and associated hypotheses is initially presented (§2). After which, an overview of the selected datasets and the task is documented (§3). Subsequently, the methodology employed to address the aforementioned research questions and hypotheses are outlined (§4) and experimental results are documented (§5). The experimental results are then draw upon to derive a set of initial conclusions (§6). Finally, details of any associated risks, backup plans and further work are provided (§7).

*Notes*

*TODO: Maybe add more information relating to our newest proposed idea: Investigating how similarity between instances of data in each proposed dataset can affect the performance of a CNN.*

*TODO: Perhaps include further elaboration on how this project relates to things that have already been done in the literature.*

**2. Research Questions**

Within this section two sets of research questions are presented. Firstly, a set of research questions that are addressed within this report is provided (§2.1). Secondly, a set of future research questions to be addressed within the concluding report is offered (§2.2). Thereafter, the aims and objectives of the proposed research questions are probed (§2.3). Finally, a set of hypotheses is given (§2.4).

**2.1. Interim Research Questions**

Using the methodology outlined within §4.x. the following research questions are investigated within this report:

1. How do differences in similarity **(NOTE: IT MAY BE BE GOOD TO LINK TO A SECTION THAT DOCUMENTS SOME KIND OF SIMILARITY METRIC BETWEEN DATA INSTANCES WITHIN EACH DATASET .. we can include this within our future work section and state an assumption for now, that is to say that the faces dataset will underperform as compared to the clothes dataset)** between datasets affect the performance (generalisation, accuracy and error) of fairly simple convolutional neural network architectures (as outlined in §4.x)?
2. How does reducing the size of a training dataset affect the performance of fairly simple convolutional neural network architectures?

Research question 2 has already been well researched within the literature [INSERT CITATIONS HERE]. However, the question links into our future research questions (§2.2) associated with using techniques {footnote: such as transfer learning and deep feature extraction} to improve the performance of small datasets on neural network architectures. For this reason, it is important that an investigation is conducted in order to establish a more conclusive understanding of the proposed datasets (§3) on our employed architecture, outlined in §4.x..

*Notes*

*We need to clearly identify what our underlying assumption is as stated above and then abstract out the possibility of some kind of similarity metric to the further work section, essentially for someone else to build upon.*

**2.2. Future Research Questions**

1. How does applying transfer learning on to deep convolutional neural networks improve the performance of classification tasks?
2. How does transfer learning perform on convolutional neural networks when the size of the dataset used to tune the network is greatly reduced in size?
3. How can deep feature extraction be used in order to improve the performance of small datasets on deep neural network architectures?
4. How does one shot learning (Siamese Network) perform on small datasets within classification tasks? (if time allows us to research this) *todo: clean up wording of research question*

**2.3. Aims and Objectives**

To investigate whether subtle and obvious differences between classes given same no. of dataset would affect performance using convolution neural network.

To try and find a good method of improving performance on the same task using a smaller version of the same dataset in combination with techniques.

1. How does applying transfer learning on  deep convolutional neural networks improve the performance of classification tasks?
2. How does transfer learning perform when the size of the dataset used to tune the network is greatly reduced in size?
3. (not included?)How can deep feature extraction be used in order to improve the performance of small datasets on deep neural network architectures?
4. How does one shot learning methods such as Siamese Network perform on small datasets within classification tasks? (if time allows us to research this) *todo: clean up wording of research question*

**2.4. Hypotheses**

* Datasets with subtle differences between classes will perform worse on classification tasks than datasets with obvious differences between classes using the proposed convolution neural network architecture (§4.x.).
* Reducing the size of the training dataset will result in worse generalisation using the proposed network architecture (§4.x.).
* Reducing the size of the training dataset will result in an overfitting of the network architecture to the subsampled dataset. **TODO: NOTE: THIS MAY NEED ADDITIONAL CLARIFICATION.**
* *TODO: ADD FURTHER HYPOTHESES*

**3. Data Set and Task**

*TODO: Look at potential similarity metrics that could be used as empirical measurements between instances of data within the same dataset, such that we can compare similarity averages between datasets.*

*TODO: Document any preprocessing methods.*

* http://cvit.iiit.ac.in/projects/IMFDB/ (Faces database)
* https://www.kaggle.com/zalando-research/fashionmnist (Clothes database)
* https://grail.cs.washington.edu/projects/deepexpr/ferg-db.html (Facial expression database)

*TODO: include details about how we're going to mop / clean the datasets before usage.*

*TODO: The Task section: 'Describe how you will evaluate the task - i.e: error metric, accuracy, generalisation, etc.. Use citations where appropriate.*

**4. Methodology**

In the first phrase of our project, the two research questions are first examined to create a baseline system for further work in transfer learning.
Both of two image databases are subjected to pre-processing before using as inputs in neural network. In the facial expression dataset,

Original png files with 256x256 pixels are downscaled to 28x28 pixels so as to be comparable with the default pixels in clothes dataset.
To understand the effect of the subtle and obvious feature differences between classes on performances (prediction accuracy, error) for  distinctive tasks, 30k images from both clothes dataset and facial dataset are evaluated on convolutional netural network respectively by performing multi-class classification task. Firstly, the architecture consists of three convolutional layers with one max-pooling Relu layer in between.The final layer is then flattened to produce one numerical output with catagorical cross-entropy as loss function (maybe add one more softmax layer before flattening to increase stability as suggested by MLP lecture?). For the optimizer, Adam or RMSprop would be used. Weight and bias is also initialized using (gloro-bengio ini. ?, random ?). After inspecting the results from first experiment. Same task is performed on much smaller dataset to investigate the discrepency of size of dataset on classification performance. (1000 dataset maybe?). By implemening the two experiments, baseline systems could be set up to investigate possible strategies to perform prediction/classification task given very small dataset which is the main goal of our project.

In the second phrase of the project, two different transfer learning methods will be studied to examine potential methods to improve performances given very small dataset which is frequent in real-world scenario. Firstly, we transfer a very large pretrained network VGG16 on our aforementioned baseline system with pretrained weights on small dataset. Since VGG16 trains on 200 types of general objects. The generality of the model might be beneficial to train on common objects (clothes dataset). Apart from transferring model to domain-specific dataset (clothes dataset). We also transfer the model to dataset with unrelated and subtle differences between classes in the dataset (facial dataset), to test the effectiveness of pretrained model on task that shares little similarity with the pretrained model. 

Besides transferring pretrained model, we also wish to investigate the effect of one-shot learning on small dataset. To demonstrate a basic version of one-shot learning we will implement Siamese network on either one of the dataset( clothes/facial expression) with the help of existing models and our modification to these models , due to time constrain and taking potential difficulty of implementing one-shot learning architecture from scratch. As a backup plan, we will abandon this experiment and focus more on transferring models methods.

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

**NOTE: We now have baseline results, these need integrating into the report ASAP, ideally over the weekend.**

**5.1. Further Experiments**

*TODO*

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

**POTENTIAL Initial References**

*Notes*

*Probably won't be using web articles as references, but if we can find what they're referencing, we can delve into the literature a little more and figure out what exactly is appropriate to use.*

*End of Notes*

https://www.kdnuggets.com/2015/03/more-training-data-or-complex-models.html

https://arxiv.org/pdf/1211.1323.pdf

https://stats.stackexchange.com/questions/226672/how-few-training-examples-is-too-few-when-training-a-neural-network

https://www.quora.com/What-is-the-recommended-minimum-training-data-set-size-to-train-a-deep-neural-network

https://medium.com/nanonets/nanonets-how-to-use-deep-learning-when-you-have-limited-data-f68c0b512cab

https://www.researchgate.net/post/What_is_the_minimum_sample_size_required_to_train_a_Deep_Learning_model-CNN

https://arxiv.org/pdf/1511.06348.pdf

http://sci2s.ugr.es/keel/pdf/specific/articulo/raudys91.pdf

http://carlvondrick.com/bigdata.pdf

https://medium.com/@malay.haldar/how-much-training-data-do-you-need-da8ec091e956

https://stats.stackexchange.com/questions/51490/how-large-a-training-set-is-needed

http://people.idsia.ch/~ciresan/data/ijcnn2012_v9.pdf

https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/1472-6947-12-8

https://sorenbouma.github.io/blog/oneshot/

https://towardsdatascience.com/one-shot-learning-face-recognition-using-siamese-neural-network-a13dcf739e

http://cs231n.github.io/transfer-learning/

https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf

https://arxiv.org/pdf/1409.1556.pdf

https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf