# Cornerstone-Project-2
This is the repositopry for the CPH 200 Cornerstone Project 2 by Hongzhou Luan and Li-Ching Chen
Due Date: 5PM PST Nov 16, 2023

## Dataset
PathMNIST
The PathMNIST is based on a prior study16,17 for predicting survival from colorectal cancer histology slides, providing a dataset (NCT-CRC-HE-100K) of 100,000 non-overlapping image patches from hematoxylin & eosin stained histological images, and a test dataset (CRC-VAL-HE-7K) of 7,180 image patches from a different clinical center. The dataset is comprised of 9 types of tissues, resulting in a multi-class classification task. We resize the source images of 3 × 224 × 224 into 3 × 28 × 28, and split NCT-CRC-HE-100K into training and validation set with a ratio of 9:1. The CRC-VAL-HE-7K is treated as the test set.

The data files of MedMNIST v2 dataset can be accessed at Zenodo38. It contains 12 pre-processed 2D datasets (MedMNIST2D) and 6 pre-processed 3D datasets (MedMNIST3D). Each subset is saved in NumPy39 npz format, named as <data> mnist.npz for MedMNIST2D and <data> mnist3d.npz for MedMNIST3D, and is comprised of 6 keys (“train_images”, “train_labels”, “val_images”, “val_labels”, “test_images”, “test_labels”). The data type of the dataset is uint8.

“{train,val,test}_images”: an array containing images, with a shape of N × 28 × 28 for 2D gray-scale datasets, of N × 28 × 28 × 3 for 2D RGB datasets, of N × 28 × 28 × 28 for 3D datasets. N denotes the number of samples in training/validation/test set.

“{train,val,test}_labels”: an array containing ground-truth labels, with a shape of N × 1 for multi-class/binary-class/ordinal regression datasets, of N × L for multi-lable binary-class datasets. N denotes the number of samples in training/validation/test set and L denotes the number of task labels in the multi-label dataset (i.e., 14 for the ChestMNIST).

## Tasks
### Part 0: Setup
For a refresher on how to access the CPH App nodes, setting up your development environment or using SGE, please refer to the Project 1 README.md]. In addition to the package requirement for project 1, make sure to install the packages listed in project 2s requirements.txt file.

As before, you can check your installation was succesful by running python check_installation.py from the project 2 directory.

The NLST dataset metadata is availale at: /wynton/protected/group/cph/cornerstone/nlst-metadata/

Preprocessed NLST scans (compatabile with the included data loaders) are included at: /scratch/datasets/nlst/preprocessed/

Note, the scan's themselves are saved on local NVMe storage to accelerate your experiments IO.

### Part 1: Build toy-cancer models with PathMNIST
In this part of the project, we'll leverage a toy dataset PathMNIST to introduce PyTorch, PyTorchLightning and Wandb. With these tools, you'll train a series of increasingly complex models to tiny (28x28 px) pathology images and study the impact of various design choices on model performance.

Note, there is a huge design space in neural network design, and so you may find extending your dispatcher.py from Project 1 to be a useful tool for managing your experiments. You may also find the starter code in main.py, lightning.py and dataset.py to be useful starting points for your experiments.

1.1: Training Simple Neural Networks with PyTorch Lightning (20 pts)
In this exercise, develop a simple neural network to classify pathology images from the PathMNIST dataset. Develop the following models:

Linear Model
MLP Model
Simple CNN Model
ResNet-18 Model (with and without ImageNet pretraining)
In doing so, explore the impact of model depth (i.e num layers), batch normalization, data augmentation and hidden dimensions on model performance. In the context of ResNet models, explore the impact of pretraining.

Your best model should be able to reach a validation accuracy of at least 99%. In your project report, include plots comparing the model variants and the impact of the design choices you explored.

#### Ideas: 
* Comparing Pydicom and SimpleITK:
  - Pydicom allows modification of metadata in dicom data such as patient information, study details, and other metadata without affecting the image data.
  - SimpleITK has built-ins for image processing tools such as filtering, segmentation and registration.
* Can we use autoencoder structure for MLP/CNN models?
* lightning has builtin Resnet-18 model. See [this](https://www.kaggle.com/code/stpeteishii/cifar10-resnet18-pytorch-lightning).


### Part 2: Build a cancer detection model with NLST
Now that you have experience developing deep learning models on toy datasets, it's time to apply these skills to a real world problem. In this part of the project, you will develop a deep learning model to predict lung cancer from low-dose CT scans from the National Lung Screening Trial (NLST). As before, you may find the project2 starter code helpful in getting started. Note, these experiments will be much more computationally intensive than the toy experiments in part 1, so you may find it useful to use the SGE cluster to run your experiments and to use multiple GPUs per experiment. You may also find it useful to use the torchio library for data augmentations.

2.1: Building cancer detection classifier (25 pts)
In this exercise, develop classifiers to predict if a patient will be diagnosed with cancer within 1 year of their CT scan. In src/dataset.py, you'll find the NLST LightningDataModule which will load a preprocessed version of the dataset where CT scans are downsampled to a resolution of 256x256x200 and stored on the fast NVME local storage.

Develop a lung cancer binary classifer and explore the impact of pretraining and model architectures on model performance. In your project report, please include experiments with the following models:

A simple 3D CNN model (extending your toy experiment)
A ResNet-18 model (with and without ImageNet pretraining) (adapted to 3D CT scans)
ResNet-3D models (with and without video pretrainig)
(Optional) Swin-T models (with Video pretraining)
In addition to these experiments, please also include an exploration of why pretraining helps model performance. To what extent is the performance boost driven by feature transfer as opposed a form of optimization preconditioning? Please design experiments to address this question and include your results in your project report.

#### Note: this means re-initialize randomly using mean and standard deviation from pretrained weights. (**extra credit**)

By the end of this part of the project, your validation 1-Year AUC should be at least 0.80.(**pooling strategy is important?**)

2.2: Building a better model with localization (25 pts)
In addition to cancer labels, our dataset also contains region annotations for each cancer CT scan. In this exercise, you will leverage this information to improve your cancer detection model. The bounding box data and the equivalent segementation masks are loaded for you in src/dataset.py.
#### note: 
Attention: cross attention between cancer images or cancer/non0cancer images
self-attention between patches
For the segmentation mask: we can have a separate pipeline that segments the nodules and classify whether the nodule will be maglinant within one year, concat this score with model prediction.
In your project report, please:

Introduce your method to incorporate localization information into your model
Note, there are many valid options here!
Add metrics to quantify the quality of your localizations (e.g. IoU, or a likelihood metric)
Add vizualizations of your generated localizations against the ground truth（**extra credit** plotting heatmap of attention and the provided annotation）
By the end of this part of the project, your validation 1-Year AUC should be at least 0.87.

#### Notes
IoU: Intersection over Union (IoU) is used when calculating mAP (mean average precision). It is a number from 0 to 1 that specifies the amount of overlap between the predicted and ground truth bounding box.
an IoU of 0 means that there is no overlap between the boxes
an IoU of 1 means that the union of the boxes is the same as their overlap indicating that they are completely overlapping
![image](https://github.com/hzluan/Cornerstone-Project-2/assets/66193810/10882652-d19c-4baa-836c-df88db0305ed)

2.3: Compare to LungRads criteria and simulate possible improvements to clinical workflow (10 pts)
Now that you've developed a lung cancer detection model, it's time to analyze its clinical implications. In this exercise, you will compare your model to the LungRads criteria (which are loaded in dataset.py) and simulate the impact of your model on the clinical workflow. In your report, please introduce a workflow of how your model could be used to ameliorate (improve) screening and provide quantitative estimates of your workflow's impact(**bayesian??**). Be sure to study the impact of your model across various subgroups, as you did in Project 1. Finally, please include a discussion of the limitations of your analyses and subsequent studies are needed to drive these tools to impact.

### Part 3: Extending your LDCT model to predict cancer risk
In this part of the project, you will extend your cancer detection model to predict cancer risk and compare your results to your best model from project 1.

Comparing to your best model from Project 1 (note not 100% overlapping questionares) (10pts)
Questionnaires from NLST are available in src/dataset.py. In your project report, please validate your PLCO model (from project 1) on the NLST dataset. Note, some of the information available in PLCO is not available in NLST, so you may need to simplify your project 1 model. Is there a meaningful performance difference between your PLCO model across the PLCO and NLST datasets? If so, why?

Extend risk model to predict cancer risk (20pts)
In this exercise, you will extend your cancer detection model to predict cancer risk. Specifically, you will predict the probability of a patient being diagnosed with cancer within [1,2,3,4,5,6] years of their CT scan. Note, there are many multiple ways to achieve this goal.

In your project report, please include the following:

Your approach to extending your classifier to predict risk over time
Detailed performance evaluation of your risk model across multiple time horizons (e.g. 1,3 and 6 years)
Comparison of your imaging-based risk model against your PLCO model
An approach for combining your imaging-based risk model with the clinical information in your PLCO model (**extra credit**)
Note, there are many valid options here!
Performance evaluation of this combined model across multiple time horizons.
Your image-based 6-year validation AUC should be at least 0.76 and your 1-year Validation should at least as good as your best detection model.

Explore clinical implications of this model (10pts)
Now that you've developed your risk model, it's time to analyze the clinical opportunities it enables. Please propose a workflow for how your model could be used to improve screening and quantify the potential impact of your workflow. Be sure to study the impact of your model across various subgroups, as you did in Project 1. Finally, please include a discussion of the limitations of your analyses and subsequent studies are needed to drive these tools to impact. (**add treatment as part of the input and compare treatment effect and survival rate, attention on image for where to pay attention to, who to screen how to screen, modality**) (**subgroup number required, writing optional**)
