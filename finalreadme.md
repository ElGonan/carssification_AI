# Carsification AI: A Machine Learning Model for Image Classification of Car Parts

**Alan Patricio González Bernal**  
**A01067546**

## Abstract
This paper presents a machine learning model designed for the classification of images of car parts. Utilizing a dataset obtained from Kaggle, an approach based on the ResNet50 architecture was implemented along with a custom-model, which has proven effective in image classification tasks. Through a data preprocessing and augmentation process, the model's quality was enhanced, achieving significant metrics in precision, recall, and F-Score. The results indicate that the model can detect car parts with a high degree of accuracy, approaching state-of-the-art standards in image classification.

## Introduction
Image classification is a continuously evolving field within artificial intelligence, with applications ranging from object identification to image segmentation. This study focuses on the classification of car parts, an area that has received limited attention compared to other image classification applications. The choice of dataset was influenced by the search for a practical project that could be tackled with limited knowledge in artificial intelligence.

A dataset of car parts available on Kaggle was selected, containing images categorized into 40 different classes[9]. The objective of this work is to develop a robust model that not only accurately classifies these images but also evaluates its performance through standard metrics in the field.

## Material and Methods
### Dataset 
The dataset used in this study was sourced from Kaggle, specifically from a repository containing images of various car parts[9]. This dataset is structured in folders representing different classes, facilitating its use in machine learning models.

The dataset followed the next structure:
```
- train
- test
- valid
- car parts.csv
- EfficientNetB2-40-(224 X 224)- 96.90.h5
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


Each class contains a variety of images and are divided into training, validation, and test sets, distributed as follows:


![Figure 1. Dataset Distribution (Train)](./images/numinstancestest.png)






## References
1. A. Aldawsari, S. A. Yusuf, R. Souissi, and M. AL-Qurishi, "Real-Time Instance Segmentation Models for Identification of Vehicle Parts," Research Article, Elm Company, Riyadh, Saudi Arabia, Apr. 11, 2023. [Online]. Available: https://doi.org/10.1155/2023/6460639

2. K. Pasupa, P. Kittiworapanya, N. Hongngern, and K. Woraratpanya, "Evaluation of deep learning algorithms for semantic segmentation of car parts," Complex & Intelligent Systems, vol. 8, pp. 3613–3625, May 2021. [Online]. Available: https://doi.org/10.1007/s40747-021-00397-8

3. PyTorch, "torchvision.transforms — Torchvision 0.16 documentation," PyTorch.org, [Online]. Available: https://docs.pytorch.org/vision/stable/transforms.html.

4. S. Bechelli and J. Delhommelle, "Machine learning and deep learning algorithms for skin cancer classification from dermoscopic images," Bioengineering, vol. 9, no. 3, p. 97, Feb. 2022. [Online]. Available: https://doi.org/10.3390/bioengineering9030097

5. Y. Luo, M. Jiang, and Q. Zhao, "Visual Attention in Multi-Label Image Classification," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2019, [Online]. Available: https://openaccess.thecvf.com/content_CVPRW_2019/papers/MBCCV/Luo_Visual_Attention_in_Multi-Label_Image_Classification_CVPRW_2019_paper.pdf

6. M. Pektaş, "Performance Analysis of Efficient Deep Learning Models for Multi-Label Classification of Fundus Image," Artificial Intelligence Theory and Applications, vol. [Online]. Available: https://dergipark.org.tr/en/download/article-file/3202713?ref=https://git.chanpinqingbaoju.com

7. TensorFlow, "tf.keras.applications.resnet.preprocess_input," [Online]. Available: https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet/preprocess_input

8. V. K. Mahto, "Transfer Learning using MobileNetV2 (acc=99.08%)," Kaggle, [Online]. Available: https://www.kaggle.com/code/vaibhavkumarmahto/transfer-learning-using-mobilenetv2-acc-99-08#Draw-Learning-Curve.

9. Gpiosenka, "Car Parts 40 Classes," Kaggle, 2021. [Online]. Available: https://www.kaggle.com/datasets/gpiosenka/car-parts-40-classes.


