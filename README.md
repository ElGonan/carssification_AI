# Carsification_AI
A machine learning model for image classification of car parts

Alan Patricio González Bernal
<br/>A01067546

## Selection of the dataset

### Introduction
There are many datasets out there but due to my limited knowledge on
AI, I decided to stick my machine learning model to be similar to the ones
we do on class. Checking different repositories of datasets, I encountered
Kaggle which is one of the most popular dataset repositories out there. 

Once inside, I decided to start looking to Classification Datasets, that way
I can apply what we see on classes here. However no dataset was interesting
enough for me. So I decided to ask my family for ideas. Then my brother
suggested the car parts classification and I liked it. So I just went to look
for it on Kaggle and bingo, found it.

### Separation
The dataset uploaded by gpiosenka is already separated in different folders:
First, we have 1 dataset smaller than the other one, which works more like a 
compliment to the first one.
- car parts
- car parts 50

The first car parts folder contains 40 different labels, and it's the one
I'll be using.

The main structure is:
- train
- test
- valid
- car parts.csv
- EfficientNetB2-40-(224 X 224)- 96.90.h5

The 40 labels i'll be working with are:
|   |   |   |   |
|--------------------|-------------------|------------------|-------------------|
| AIR COMPRESSOR     | CYLINDER HEAD     | FUEL INJECTOR    | IDLER ARM         |
| ALTERNATOR         | DISTRIBUTOIR      | FUSE BOX         | IGNITION COIL     |
| BATTERY            | ENGINE BLOCK      | GAS CAP          | LEAF SPRING       |
| BRAKE CALIPER      | HEADLIGHTS        | LOWER CONTROL ARM| MUFFLER           |
| BRAKE PAD          | IGNITION COIL     | MUFFLER          | OIL FILTER        |
| BRAKE ROTOR        | IDLER ARM         | OIL FILTER       | OIL PAN           |
| CAMSHAFT           | LEAF SPRING       | OIL PAN          | OVERFLOW TANK     |
| CARBERATOR         | LOWER CONTROL ARM | OVERFLOW TANK    | OXYGEN SENSOR     |
| COIL SPRING        | MUFFLER           | OXYGEN SENSOR    | PISTON            |
| CRANKSHAFT         | OIL FILTER        | PISTON           | RADIATOR          |
| RADIATOR FAN       | RADIATOR HOSE     | RIM              | SPARK PLUG        |
| STARTER            | TAILLIGHTS        | THERMOSTAT       | TORQUE CONVERTER  |
| TRANSMISSION       | VACUUM BRAKE BOOSTER | VALVE LIFTER  | WATER PUMP        |

### Augmentation
The dataset has for each label:
- train
    - 170-200 instances
- test
    - 5 instances
- validation
    - 5 instances

So I will do data augmentation to increase my dataset, specially on the test and
valdiation parts.

So, during the prepocess I applied the following augmentations:
- rotation_range=10
- width_shift_range=0.8
- height_shift_range=0.2
- horizontal_flip=True

## Preprocess

### Scalation
I decided to follow the preprocess of images the same way we've been doing, 
that means normalize images (scale them) from 0-255 to 0-1.

### Preprocess
I decided to resize them to 150x150 so I can visualize them
correctly and to ensure all images have the same size. Also I used the 
class_mode of "categorical", because I have multiple classes (40). The batch size 
will be the same as the one we used on classes (8) because it's one of the most 
common batch sizes. 

# Model Implementation and initial evaluation

## Model
As my pre-trained model I'll be using ResNet50, Many research papers[1][2][4][5] use it and
comparing it is one of the best, with the next metrics:
- Precision
    - 0.646
- Recall
    - 0.779
- F-Score
    - 0.579
[5]

I also have, just for comparison, the custom Model used in the paper[4]. Which consists of:
- 2D Convolutional Layer
    - 8 filters
    - 3x3 kernel
- Max Pooling Layer
- 2D Convolutional Layer
    - 16 filters
    - 3x3 kernel
- Max Pooling Layer
- Batch Normalization
- 2D Convolutional Layer
    - 32 filters
    - 3x3 kernel
2D Convolutional Layer
    - 64 filters
    - 3x3 kernel
- Max Pooling Layer
- Flatten Layer
- Dense Layer
    - 128
- Dense Layer
    - 2

It's important to mention that the custom model was trained with 2 classes
and not 40, so I expect the accuracy to be lower than the paper announced.

However, I list the custom model metrics:
- Precision
    - 0.94
- Recall
    - 0.76
- F-Score
    - 0.84
[4]

As my own model, I will be using the following layers:
- ResNet50 [3]
    - input_shape=(254, 254, 3)
    - include_top=False
    - weights="imagenet"
    - classifier_activation="softmax"
    - name="resnet50"
    - pooling="average"
- GlobalAveragePooling2D
- Dense
    - 2048
    - relu
- Dense
    - 256
    - relu
- Dense
    - 40
    - softmax

The compilation of the model is the same as used on classes:
- Optimizer
    - Adam
- Loss function
    - categorical_crossentropy[*]
- Metrics
    - accuracy

[*] As Benji mentioned in the first retro, the loss function that I should be using
is categorical_crossentropy. This is because I have multiple classes
and not binary classification. So I will be using that one.

## Training
For the training, I will be implementing checkpoints to save the
model every 5 epochs. I decided to have 25 epochs just to see what
we get.

## Evaluation
The model was trained for 25 epochs, with a batch size of 8. 
The model was trained with the Adam optimizer and the loss function
was categorical crossentropy.

According to the papers, My metrics will be:
- Precision
- Recall
- F-Score (or F1)

## First Results
The model was trained for 25 epochs, with a batch size of 8.
The model was trained with the Adam optimizer and the loss function
was categorical crossentropy.

The model gave the following results:
- Precision
    - 0.0163
- Recall
    - 0.0200
- F-Score (or F1)
    - 0.0170

<b>This tells me that the model is not performing well, it's basically
not learning anything. This can be because of what I meantioned about
the normalization. It's not the way the ResNet50 expects it to be
so it may not be able to detect anything.</b>

I also added the plots of accuracy and loss to be able to know if my model 
is overfitting, fitting :) or underfitting.

By now, my conclussion is that the model is <b>fitting :)</b>. However the rest
of the parameters indicate otherwise.


# Improvements
## Modifications
As stated on the notes, I improved my model and now looks like this:

- ResNet50 [3]
    - input_shape=(254, 254, 3)
    - include_top=False
    - weights="imagenet"
    - classifier_activation="softmax"
    - name="resnet50"
- GlobalAveragePooling2D
- Dense
    - 512
    - relu
- Dense
    - 128
    - relu
- Dense
    - 40
    - softmax

I also modified the image resize to maintain the sizes it should since ResNet50
expects images from 224x224 or greater.

## New Results
with this new modifications, in only 5 epochs the model reaches:

- Precision
    - 0.8977
- Recall
    - 0.8650
- F-Score (or F1)
    - 0.8645

## Conclusion
The model is now performing much better than before, the precision, recall and F-Score are much better. The model is now able to detect the car parts and I have achieved the almost equal the State of the Art[4] with only 5 epochs. I'll try other day with the ammount of epochs the paper[4] used and then compare them correctly.

Also I have a Kaggle notebook[8] with an accuracy of 99.08 with this dataset. That´s my new objective.

The [carssification_AI.ipynb](./carssification_AI.ipynb) file contains the code.

# Notes:
## 25/05/2025

### Improved model
Eureka! I managed to improve the model basically by doing the things the
way they are supposed to be done. The first issue was the Normalization, the
0-1 normalization, as mentioned, needs to be different. And keras has an import
that makes the calculation for me called "preprocess input"[7]. After that, instead
of using Flatten on the model, I decided to use GlobalAveragePooling, which
instead of flattening in 3 channels, makes an average to only 1 channel. This
is better because its easier on the processor or GPU and it's still pretty
efficient.

with this new modifications, in only 5 epochs the model reaches:

- Precision
    - 0.8949
- Recall
    - 0.8750
- F-Score (or F1)
    - 0.8728

## 24/05/2025

### New Paper
The new paper[5] indexed is referent to multi-label image classification,
I thas many models and combinations of pre trained models. The focus
point and maybe my baseline is the ResNet50 model Baseline as it's
mentioned in the paper[5].

I have done a lot of research and I decided to also use the custom model
from the paper[4]. I will be using it as a comparison to see if the
ResNet50 is really better than the custom model.


### Better model?
After some research and looking at kaggle, I noticed that other people
tried to use this dataset but didn't used ResNet, but EfficientNetB0, 
which is more complex than ResNet50 but better overall. I'll maybe try
It but first I want to see how the ResNet50 performs.[6] 


## 21/05/2025
After the first evaluation, Benji and Peblo allowed me to notice my gigantic
mistake: I confused terms and did object detection investigation but I need
<b>Image Classification</b> investigation. So I will re-investigate the models
to find one that fits. However Benji told me that ResNet is maybe the best
choice so I need to investigate that. It's my starting point.

### Model Selection!
As investigated before, ResNet50 is going to be very useful, on the paper[4]
they compare 3 models:
<br/>1. Xception
<br/>2. VGG16
<br/>3. ResNet50

and the one that had the best Accuracy was ResNet50. So I'll start using that
model.


## First Notes

### Model selection?
According to the 2 indexed research papers on this repo (and also referenced at 
the end of this file[1] [2]), the best models for object detection are:
<br/>1. HTC
<br/>2. YOLACT
<br/>3. Mask R-CNN

### ResNet Normalization
Both papers [1] [2] mention that the models where trained with ResNet as a 
backbone. Which doesn't have the same as the normalization 0-1, it's a bit 
more complex with the next values:

mean = 0.485, 0.456, 0.406
 <br/> std = 0.229, 0.224, 0.225
 [3]

This allows the input images to have the same distribution as the data the model 
was originally trained on, which helps maintain the expected performance of the 
pretrained model [3]. I have to consult this with Benji, however I think I'll 
try to do it the way I'm doing it rn and change if I need to.

### Dataset
The data set for this project was obtained online on Kaggle, uploaded by
the user gpiosenka:
```
https://www.kaggle.com/datasets/gpiosenka/car-parts-40-classes
```

# References
1. A. Aldawsari, S. A. Yusuf, R. Souissi, and M. AL-Qurishi, "Real-Time Instance Segmentation Models for Identification of Vehicle Parts," Research Article, Elm Company, Riyadh, Saudi Arabia, Apr. 11, 2023. [Online]. Available: https://doi.org/10.1155/2023/6460639 

2. K. Pasupa, P. Kittiworapanya, N. Hongngern, and K. Woraratpanya, "Evaluation of deep learning algorithms for semantic segmentation of car parts," Complex & Intelligent Systems, vol. 8, pp. 3613–3625, May 2021. [Online]. Available: https://doi.org/10.1007/s40747-021-00397-8

3. PyTorch, “torchvision.transforms — Torchvision 0.16 documentation,” PyTorch.org, [Online]. Available: https://docs.pytorch.org/vision/stable/transforms.html.

4. S. Bechelli and J. Delhommelle, "Machine learning and deep learning algorithms for skin cancer classification from dermoscopic images," Bioengineering, vol. 9, no. 3, p. 97, Feb. 2022. [Online]. Available: https://doi.org/10.3390/bioengineering9030097

5. Y. Luo, M. Jiang, and Q. Zhao, "Visual Attention in Multi-Label Image Classification," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2019, [Online]. Available: https://openaccess.thecvf.com/content_CVPRW_2019/papers/MBCCV/Luo_Visual_Attention_in_Multi-Label_Image_Classification_CVPRW_2019_paper.pdf

6. M. Pektaş, "Performance Analysis of Efficient Deep Learning Models for Multi-Label Classification of Fundus Image," Artificial Intelligence Theory and Applications, vol. [Online]. Available: 
https://dergipark.org.tr/en/download/article-file/3202713?ref=https://git.chanpinqingbaoju.com

7. TensorFlow. (2024). tf.keras.applications.resnet.preprocess_input. [Online]. Available: https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet/preprocess_input

8. V. K. Mahto, "Transfer Learning using MobileNetV2 (acc=99.08%)," Kaggle. [Online]. Available: https://www.kaggle.com/code/vaibhavkumarmahto/transfer-learning-using-mobilenetv2-acc-99-08#Draw-Learning-Curve. [Accessed: May 25, 2025].
