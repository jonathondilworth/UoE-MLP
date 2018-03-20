## Preliminaries

We deal with small data and have found how the amount of data used to train a network affects its performance. Based on this we try to cope with this situation and take advantage of neura; networks. To do so. we try transfer learning. This technique allows us to use the knowledge that a given network already has. Obviously, there are several differences and coincides between the tasks and amount of data for the pre-trained model and the problem we try to solve.

We select the VGG16 pre-trained network. The decision is basde on the similarities of the datasets and tasks related to that network and what we have. First, VGG16 has been trained to solve a classification task. same as ours. Also, the data used to train VGG16 are images, we also use images. 

VGG16 has two main versions, the main difference between the versions are the dataset used to train the network. One of the versions used ImageNet, the other used CIFAR-100. ImageNet has 200 classes and XX number of samples. CIFAR-100 has 100 clasess and YY number of classes. We decide to use the version trained over ImageNet. Since ImageNet has a higher number of classes than CIFAR-100, we assume that it has more knowledge and can provide better results for our dataset. 

Now, the differences between the VGG16 ImageNet version and our problem. This is obviously related to the number of classes (100 vs 7) and the type of images. ImageNet uses color images of size width x height x 3, we have grayscale images of 64 x 64.

## Methodology

VGG16 has 19 layers (16 convolutional layers and 3 fully connected ones). We decide to use the knowledge from the fully connectes layers. Then, we modify the fully connected layers to adapt the model to our needs. This pipeline has a bottleneck between the convolutional layers and the fully connected ones. To reduce the impact of this bottleneck, we divide the process into two components. First, we simply use the fully connected layers to convert the raw images from the dataset to another representation. Tnis representation is the output of the last convolutional layer. In some sense, this can be seen as an encoding process, where we go from images that provide information about intensity, to images that provide other kind of information. In other words, we are converting the features from the raw (pixels) images to another feature space. 

That process is time consumming, however, one of the benefits of it is the dimensionality reduction, we go from 64x64=4096 features per image (pixel intensities) to 2048 (we can-t describe this) features per image. Once we have finished this process, we use the output to feed the fully connected layers. 

We need to adapt the fully connected layers to our needs. To do so, we define three alternatives. The main difference among them is the number of fully connected layers. In the first option, we use only one fully connected layer. This option has 205,607 parameters. The second alternative add another fully connected layer after the previous one. This option has 210,307 parameters. Finally, the third alternative adds another fully connected layer after the other two. This option has 211,407 parameters.

For the first fully connected layer we use a L2 regularization strategy. The main objective is to avoid overfitting, which is specially important for small data. Each layer uses a uniform parameter initialization strategy. This kind of initialization allows to have the activation done in the linear part of a sigmoid function. For the output layer, we use a softmax activation with a categorical cross entropy error loss. The learning rule is Adam.

## Experiments

At this stage we also use different sizes of the datasets, 100%, 10%, 1%, 0.1%, percentages that represent the amount of data from the original size of each dataset. In the previous stage we used the following percentages to run the experiments: 100%, 75%, 50%, 25%, 10%, 1%. Since we were able to find the relationship between the reduction of the performance of a network and the size of the dataset. In this project, out main focus is the small data. We expect to see the same affectation in this stage, however, we want to observe how big is this affectation with this approach, that is why we use these sizes: 100%, 10%, 1%, 0.1%

We also try two different activation functions for the fully connected layer. The first one is the sigmoid function, we use this function based on the initialization strategy. The second one is ELU, the reason for this activation function is to enable non-linearities between layers. Moreover, ELU has a gradients that is similar to the natural gradient (smooth).

Finally, we use different learning rates values: 0.01, 0.001, 0.0001

Based on the description of the components in the methodology for the transfer learning approach, we have the following values to configure the experiments for each dataset:

- Number of fully connected layers: 3
- Number of activation functions: 2
- Sizes of the dataset: 4
- Learning rates: 3

We combine all of the previous options and end up with 72 experiments for every dataset. Due to this number of experiments and the limited computational resources, we set a fixed number of epochs for every experiment(20), and a fixed size for the mini-batches used for the stochastic gradient descent calculation (50).

The objective of this first round of experiments is to find the configurations that gives better results in terms of validation accuracy for the small data. We also want compare the affectation on the performance with our previous experiments

After running the previous experiments, we analyse the results based on the validation accuracy. Then, we select the configurations that get the highest values for validation accuracy for dataset sizes of 1% and 0.1% and perform new experiments with higher number of epochs to analise the behaviour of the fully connected part of the architecture.

## Results

We analyse the first round of experiments based on the validation accuracy. Overall, there is still an evident reduction of the performance of the architecture relate to the size of the datasets. However, there are several things to highlight:

- For each dataset, the configuration that provides the highest accuracy for the smallest dataset size are quite different as seen in table 1. It is specially important to note the activation functions. It is expected that the Sigmoid activation function gives better results since the uniform initialization strategy is meant to benefit this activation function. However, in the case of expressions dataset, the best result is obtained with ELU. But the best accuracy using Sigmoid activations is not that far 0.30.

| Dataset | Accuracy | Fully connected layers | Activation | Learning rate |
| - | - | - | - | - |
| Clothes | 0.64 |2 | Sigmoid | 0.01 |
| Expressions | 0.31 |3 | Elu | 0.001 |

Table 1: Configuration that allows the highest accuracy for size of 0.1% for clothes and expressions datasets

- The proposed approach has provided more benefits to the clothes dataset than to the expressions one, as seen in figures accuracy_reduction_00.png and accuracy_reduction_01.png This gives us some clues about the domain of the images in every dataset. Based on these results, we can say that the "distance" between the clothes domain and ImageNet is smaller than the "distance" between the expressions domain and ImageNet.