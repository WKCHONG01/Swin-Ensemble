
# Swin-Ensemble

## PlantTraits2024 - FGVC11
## Descriptions
This competition aims to predict plant properties - so called plant traits - from citizen science plant photographs. Why are plant traits currently so relevant? Plant traits are plant properties that are used to describe how plants function how they interact with the environment. For instance, the trait of plant canopy height indicates how good a plant is at overshadowing its neightbors in the competition for sun light. Robust leaves (indicated by the leaf mass pear leaf area) indicate that plants optimize towards extreme conditions, such as heavy winds or droughts. Yet, environmental conditions are not static. Due to global change, the biosphere is being transformed at accelerating pace. Especially climate change is assumed to drastically impact the functioning of the ecosystems. This includes several processes, e.g. adaptions of plants and their traits to new conditions or even a altered plant species distribution with a resulting modification of the distribution of plant traits. However, we can hardly project on a global scale how plant traits and as such entire ecosystems will react to climate change because we do not have sufficient data on plant traits.

A data treasure in this regard may be the growing availability of citizen science photographs. Thousands of citizens around the globe photograph plants with species identification apps (examples are iNaturalist or Pl@ntNet). The species are identified using AI algorithms, and the prediction, photograph, and geolocation are curated in open databases. There are already more than 20 million plant photographs available, covering all ecosystem types and continents.

In its original form, this data initially only provides information on the species name of a plant and not its traits. However, a pioneering study showed that artificial intelligence can predict plant traits from such photographs using Convolutional Neural Networks (Schiller et al., 2021). To achieve this, we paired sample images from the iNaturalist database with plant trait data that scientists have been curating for decades for various species. The challenge was that the images and plant trait observations were not acquired for the same plant individuals or at the same time. Nevertheless, using a weakly supervised learning approach, we trained models that demonstrated the potential of this approach for a few plant traits. However, this potential was evident only for a limited number of plant traits and a couple of thousand images. This competition aims to further unlock the potential of predicting plant traits from plant photographs. To achieve this, we gathered more training data (over 30,000 images with labels).

## Objectives
The primary objective of this competition is to employ deep learning-based regression models, such as Convolutional Neural Networks (CNNs) like ConvNext or Transformers, to predict plant traits from photographs. These plant traits, although available for each image, may not yield exceptionally high accuracies due to the inherent heterogeneity of citizen science data. The various plant traits describe chemical tissue properties that are loosely related to the visible appearance of plants in images. Despite the anticipated moderate accuracies, the overarching goal is to explore the potential of this approach and gain insights into global changes affecting ecosystems. Your contribution to uncovering the wealth of data and the distribution of plant traits worldwide is invaluable.

## Evaluations
The models will be evaluated against the independent test data. The evaluation metric for this competition is the mean R2 over all 6 traits. The R2 is commonly used for evaluating regression models and is the ratio of the sum of squares the residuals (SSres) to the total sum of squares (SStot).

The R2 can result in large negative values. To prevent that we will only consider R2 values > 0.

## Proposed Method
Ensemble method with the combination of Swin Transformer and Self Attention Layers. Further details are shown in the diagram below:
![ML](https://github.com/WKCHONG01/Swin-Ensemble/assets/100023394/bdb07c2b-6d82-4874-b7f4-7b9195209068)

## How to run
### Please login your KAGGLE account to download the planttraits dataset (https://www.kaggle.com/competitions/planttraits2024/data)
Place the Swin-Ensemble.py files into the a folder with the dataset.
![Screenshot 2024-07-11 111735](https://github.com/WKCHONG01/Swin-Ensemble/assets/100023394/8d05aa5b-e833-4938-9a49-0e0ea24218da)


Before running the python file, make sure your file directory is in the folder that contains both the dataset and the python file.
--train for training the model with the train dataset
--test for evaluating the model with the test dataset

In the ./checkpoint folder, it contains all the trained models weights. 
The submission.csv file is the output from the test dataset.

