# Carsification AI: A Machine Learning Model for Image Classification of Car Parts

**Alan Patricio González Bernal**  
**A01067546**

## Abstract
This project presents a detailed evaluation of a deep learning model for the
classification of car parts into 40 distict classes using a Convolutional Neural
Network (CNN) based on the ResNet50 architecture. The model employs transfer
learning by leveraging pretrained weights from ResNet50[8], which enhances its
performance in distinguishing between multiple classes of car parts.

The model is constructed with several layers, including GlobalAveragePooling2D,
Dropout, and multiple dense layers, culminating in a softmax output layer for
multiclass classification. Training is conducted using categorical crossentropy
as the loss function and the Adam optimizer, with early stopping and model
checkpointing implemented to prevent overfitting and ensure optimal performance.

Evaluation metrics, including precision, recall, and F1-score, are computed
from the confusion matrix to assess model performance. The results indicate
that the model achieves a validation accuracy of up to 90.5%, with a precision
of 89.4%, a recall of 87.5%, and an F1-score of 87.2%. This comprehensive
approach demonstrates the effectiveness of deep learning in automating the
classification of car parts, thereby facilitating enhanced workflows in
automotive industries.

<b>Keywords</b>: image classification, ResNet50, car parts, convolutional neural
network, transfer learning, car

## Introduction
The introduction of the chinnese automotive industry to our continent has led to an
increase in the number and quality of car parts available in the market and with 
more cars in circulation, the need for efficient and accurate classification of car 
parts has become paramount. This project aims to address this need partially by
developing a machine learning model capable of classifying images of car parts to
enhance the efficiency of automotive workflows. Not only for dealerships, but also
for repair shops, scavengers, and other automotive-related businesses[1].

To mitigate the challenges associated with manual classification, this project
employs a deep learning approach, specifically using a Convolutional Neural Network
(CNN) based on the ResNet50 architecture. CNNs are particularly well-suited for
image classification tasks due to their ability to automatically learn and extract
features from images, making them highly effective for tasks involving visual data. 
The ResNet50 architecture, with its deep residual learning framework, allows for the
training of very deep networks while addressing the vanishing gradient problem, which 
is a common issue in deep learning[9].

The model used model is based on the ResNet50 architecture, one of the most widely used models
for image classification tasks, along with transfer learning techniques to improve
the model's performance. Transfer learning allows the model to leverage
pretrained weights from a model trained on a large dataset, such as ImageNet, to
enhance its ability to classify images of car parts. This approach is particularly
effective when the dataset is limited, as it allows the model to learn from
features learned from a larger dataset, thereby improving its accuracy and
generalization capabilities[4].

## Material and Methods
### Dataset 
The dataset used in this study was sourced from Kaggle, specifically from a repository 
containing [images of various car parts](https://www.kaggle.com/datasets/gpiosenka/car-parts-40-classes).
This dataset is structured in folders representing different classes, facilitating its 
use in machine learning models.

The dataset followed the next structure:
```
- train
- test
- valid
- car parts.csv
```
and the labels are as follows:
| 1-10  | 11-20 | 21-30 | 31-40  |
|--------------------|-------------------|------------------|---------------------|
| AIR COMPRESSOR     | CYLINDER HEAD     | LOWER CONTROL ARM| RIM                 |
| ALTERNATOR         | DISTRIBUTOIR      | MUFFLER          | SPARK PLUG          |
| BATTERY            | ENGINE BLOCK      | OIL FILTER       | STARTER             |
| BRAKE CALIPER      | FUEL IJNECTOR     | OILF PAN         | TAILLIGHTS          |
| BRAKE PAD          | FUSE BOX          | OVERFLOW TANK    | THERMOSTAT          |
| BRAKE ROTOR        | GAS CAP           | OXYGEN SENSOR    | TORQUE CONVERTER    |
| CAMSHAFT           | HEADLIGHTS        | PISTON           | TRANSMISSION        |
| CARBERATOR         | IDLER ARM         | RADIATOR         | VACUUM BRAKE BOOSTER|
| COIL SPRING        | IGNITION COIL     | RADIATOR FAN     | VALVE LIFTER        |
| CRANKSHAFT         | LEAF SPRING       | RADIATOR HOSE    | WATER PUMP          |


## Methodology
# Preprocessing
The dataset is distributed as follow:
![Figure 1. Dataset Distribution (Train)](./images/numinstancestrain.png)
<p align="center"><em>Figure 1. Dataset Distribution (Train)</em></p>

each instance is from a fixed size (254x254 pixles) and they are already
separated in folders by class, so the preprocessing steps are minimal.

The Validation and test sets are also already separated as follows:

![Figure 2. Dataset Distribution (Validation)](./images/numinstancesvalid.png)
<p align="center"><em>igure 2. Dataset Distribution (Validation)</em></p>

![Figure 3. Dataset Distribution (Test)](./images/numinstancestest.png)
<p align="center"><em>Figure 3. Dataset Distribution (Test)</em></p>

so they also requiered almost no prepocessing. However since the validation
and Test data are not as sustantial as the training data, Data Augmentation
was applied to both Validation and Test sets to increase the number of
instances and improve the model's generalization capabilities. The augmentation
techniques are:
- rotation range
    - 10
- width shift range
    - 0.2
- height shift range
    - 0.2
- horizontal flip
    - True

Also as part of the preprocessing the `process_input` function from keras was
used, this is due to the normalization process requiered by ResNet50[7] which
requieres the images to be preprocessed in a specific way. This process consists
basically in inverting the RGB channels to BGR, and then subtracting the mean
pixel value from each channel.


#


### Model Architecture
ResNet50 (Residual Network 50) is a deep convolutional
neural network that has been pretrained on the ImageNet dataset, which contains millions
of images across thousands of categories. The Residual Network architecture works by
introducing skip connections, allowing gradients to flow through the network more 
effectively during training. This helps to mitigate the vanishing gradient problem, 
enabling the training of very deep networks.

In the specific case of ResNet50, the model consists of 50 layers, including
convolutional layers, batch normalization layers, and fully connected layers. The
model is designed to learn hierarchical features from images, starting from low-level
features such as edges and textures, to high-level features such as shapes and objects.
The model is trained using a large dataset of labeled images, allowing it to learn
the patterns and characteristics of different classes of images.


## References
1. A. Aldawsari, S. A. Yusuf, R. Souissi, and M. AL-Qurishi, "Real-Time Instance Segmentation Models for Identification of Vehicle Parts," Research Article, Elm Company, Riyadh, Saudi Arabia, Apr. 11, 2023. [Online]. Available: https://doi.org/10.1155/2023/6460639

2. K. Pasupa, P. Kittiworapanya, N. Hongngern, and K. Woraratpanya, "Evaluation of deep learning algorithms for semantic segmentation of car parts," Complex & Intelligent Systems, vol. 8, pp. 3613–3625, May 2021. [Online]. Available: https://doi.org/10.1007/s40747-021-00397-8

3. PyTorch, "torchvision.transforms — Torchvision 0.16 documentation," PyTorch.org, [Online]. Available: https://docs.pytorch.org/vision/stable/transforms.html.

4. S. Bechelli and J. Delhommelle, "Machine learning and deep learning algorithms for skin cancer classification from dermoscopic images," Bioengineering, vol. 9, no. 3, p. 97, Feb. 2022. [Online]. Available: https://doi.org/10.3390/bioengineering9030097

5. Y. Luo, M. Jiang, and Q. Zhao, "Visual Attention in Multi-Label Image Classification," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2019, [Online]. Available: https://openaccess.thecvf.com/content_CVPRW_2019/papers/MBCCV/Luo_Visual_Attention_in_Multi-Label_Image_Classification_CVPRW_2019_paper.pdf

6. M. Pektaş, "Performance Analysis of Efficient Deep Learning Models for Multi-Label Classification of Fundus Image," Artificial Intelligence Theory and Applications, vol. [Online]. Available: https://dergipark.org.tr/en/download/article-file/3202713?ref=https://git.chanpinqingbaoju.com

7. TensorFlow, "tf.keras.applications.resnet.preprocess_input," [Online]. Available: https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet/preprocess_input

8. V. K. Mahto, "Transfer Learning using MobileNetV2 (acc=99.08%)," Kaggle, [Online]. Available: https://www.kaggle.com/code/vaibhavkumarmahto/transfer-learning-using-mobilenetv2-acc-99-08#Draw-Learning-Curve.

9. K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Las Vegas, NV, USA, Jun. 2016. [Online]. Available: https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf


10. Gpiosenka, "Car Parts 40 Classes," Kaggle, 2021. [Online]. Available: https://www.kaggle.com/datasets/gpiosenka/car-parts-40-classes.

