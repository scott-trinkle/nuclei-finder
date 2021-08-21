[![Streamlit
App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/scott-trinkle/nuclei-finder/main/app.py)

# Nuclei Finder

![](sample_imgs/demo.png)

Nuclei Finder is a [Streamlit](https://streamlit.io) app for nucleus
segmentation in optical cell imaging. It was built using a UNET deep learning
architecture trained from scratch on the [2018 Data Science
Bowl](https://www.kaggle.com/c/data-science-bowl-2018/overview) dataset, which
includes a diverse set of cell types, magnifications, and imaging modalities.

## Approach
- Used corrected training data from [here](https://github.com/lopuhin/kaggle-dsbowl-2018-dataset-fixes)
- Trained with both training and test data from stage1 of the original
  competition
- Stratified training and validation sets using image [class
  labels](https://www.kaggle.com/c/data-science-bowl-2018/discussion/48130)
- Decomposed training set images into overlapping tiles of size 128x128
- [UNET](https://en.wikipedia.org/wiki/U-Net) architecture
- Regularized with early stopping based on validation loss

Additional information about the model is found in `Train UNET.ipynb`.

## Planned future updates:
- I am currently working on retraining the model using ResNet101 trained on the ImageNet dataset as the encoder in the UNET architecture
- Experiment with a [Mask R-CNN](https://arxiv.org/abs/1703.06870) architecture
- Expand training with [additional datasets](https://www.nature.com/articles/s41597-020-00608-w)
- Implement additional post-processing steps

## Installation

### Web

Nuclei Finder is deployed with Streamlit at
[https://share.streamlit.io/scott-trinkle/nuclei-finder/main/app.py](https://share.streamlit.io/scott-trinkle/nuclei-finder/main/app.py)

### Local

Clone the repository: 

```
git clone https://github.com/scott-trinkle/nuclei-finder
```

Install dependencies (preferably within a virtual environment)

```
pip install -r requirements.txt
```

Run app with Streamlit:

```
cd nuclei-finder
streamlit run app.py
```
