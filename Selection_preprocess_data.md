
Alan Patricio González Bernal
<br/>A01067546

## Selection of the dataset
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
The dataset already has a lot of instances for each label and is already
separated, so I decided I don't need to do augmentation to it.

## Preprocess
### Escalation
I decided to follow the preprocess of images the same way we've been doing, 
that means normalize images (scale them) from 0-255 to 0-1.

### Preprocess
I decided to resize them to 150x150 so I can visualize them
coorectly and to ensure all images have the same size. Also I used the 
class_mode of "categorical", because I have multiple classes (40). The batch size 
will be the same as the one we used on classes (8) because it's one of the most 
common batch sizes. 

The [carssification_AI.ipynb](./carssification_AI.ipynb) file contains the code
of the preprocess.

## Notes:

### Model selection?
According to the 2 indexed research papers on this repo (and also referenced at 
the end of this file[1] [2]), the best models for object detection are:
<br/>1. HTC
<br/>2. YOLACT
<br/>3. Mask R-CNN

### ImageNet Normalization
Both papers [1] [2] mention that the models where trained with ResNet as a backbone. After  ImageNet Normalization,
which isn't the same as the normalization 0-1, it's a bit more complex with
the next values:

mean = 0.485, 0.456, 0.406
 <br/> std = 0.229, 0.224, 0.225
 [3]

This allows the model to know better the color and texture of the images,
therefore returning a better result. I have to consult this with Benji,
however I think I'll try to do it the way I'm doing it rn and change if
I need to.

# References
1. A. Aldawsari, S. A. Yusuf, R. Souissi, and M. AL-Qurishi, "Real-Time Instance Segmentation Models for Identification of Vehicle Parts," Research Article, Elm Company, Riyadh, Saudi Arabia, Apr. 11, 2023. [Online]. Available: https://doi.org/10.1155/2023/6460639 

2. K. Pasupa, P. Kittiworapanya, N. Hongngern, and K. Woraratpanya, "Evaluation of deep learning algorithms for semantic segmentation of car parts," Complex & Intelligent Systems, vol. 8, pp. 3613–3625, May 2021. [Online]. Available: https://doi.org/10.1007/s40747-021-00397-8

3. PyTorch, “torchvision.transforms — Torchvision 0.16 documentation,” PyTorch.org, [Online]. Available: https://docs.pytorch.org/vision/stable/transforms.html.

