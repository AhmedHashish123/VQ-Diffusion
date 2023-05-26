# Choosing The Model:<br />

The model chosen is a vector quantized (VQ) diffusion model based on these two papers:<br />
- [Vector Quantized Diffusion Model for Text-to-Image Synthesis](https://arxiv.org/pdf/2111.14822.pdf)
- [Improved Vector Quantized Diffusion Models](https://arxiv.org/pdf/2205.16007.pdf)

## Installing The Code Requirements:<br />

There are three ways to run the project:<br />
1. A .pynb file that was built to work with Google Colab.
2. A method through CLI.
3. A method through a web app.

Method number 1 only requires that the user uploads the file to Colab. Everything will run smoothly after installing the packages by simply running the first cell.<br />
Methods number 2 and 3 require downloading some packages. These packages are in "Requirements.txt" file. You simply need to create an environment (preferrably using Anaconda) and do the following:<br />
- Choose Python 3.10.11
- After the environement is created, open a terminal with this environment.
- Copy each command in "Requirements.txt" to the terminal and run it.

## Running The Training:<br />

There are two ways to train:<br />
1. Through Colab using the .ipynb file.
2. Through the given source code files.

To train using method 1:<br />
- Go to "configs/coco.yaml"
- You can control all the configurations for training in this file. Feel free to leave it as it is.
- Simply follow the steps in the notebook which include:
  - Installing the packages.
  - Cloning the repo.
  - Downloading the dataset.
  - Running the training file.

To train using method 2:<br />
- Create a folder called "datasets" in the root directory of the project.
- Create a folder called "MSCOCO_Caption" in "datasets".
- Follow the directory structure for Microsoft COCO Dataset in the "Data Preparing" section in "readme.md" file.
- Download the [dataset](https://cocodataset.org/#download), choose:
  - 2014 Train images
  - 2014 Val iamges
  - 2014 Train/Val annotations
- "2014 Train images" is a compressed file containing a folder called "train 2014".
- "2014 Val images" is a compressed file containing a folder called "val 2014".
- "2014 Train/Val annotations" is a compressed file containing .JSON files. You only need two:
  - "captions_train2014.json"
  - "captions_val2014.json"
- Go to "configs/coco.yaml"
- You can control all the configurations for training in this file. Feel free to leave it as it is.

**Note: Training requires a powerful machine with lost of VRAM.<br />

## Samples:<br />

Since training requires a very powerful system. I could not train using the original COCO 2014 dataset. I created a stripped down version.<br />
The reason for this was to check that the training works. I also trained for one epoch. So, it goes without saying that the model will not produce good results.<br />
Even the provided pretrained model was not trained for a lot of epochs.<br />
In this section, I'm going to compare the outputs of my trained model and the pretrained model using the same propmt: "A group of elephants walking in muddy water"<br />
There are six different inference methods which will also be shown.<br />

### Pretrained Inference VQ-Diffusion:<br />
![alt text](https://github.com/AhmedHashish123/VQ-Diffusion/blob/main/Samples/Pretrained/pretrained_1/000000.png)
![alt text](https://github.com/AhmedHashish123/VQ-Diffusion/blob/main/Samples/Pretrained/pretrained_1/000001.png)
![alt text](https://github.com/AhmedHashish123/VQ-Diffusion/blob/main/Samples/Pretrained/pretrained_1/000002.png)
![alt text](https://github.com/AhmedHashish123/VQ-Diffusion/blob/main/Samples/Pretrained/pretrained_1/000003.png)

### Pretrained Inference Improved VQ-Diffusion with learnable classifier-free sampling:<br />
![alt text](https://github.com/AhmedHashish123/VQ-Diffusion/blob/main/Samples/Pretrained/pretrained_2/000000.png)
![alt text](https://github.com/AhmedHashish123/VQ-Diffusion/blob/main/Samples/Pretrained/pretrained_2/000001.png)
![alt text](https://github.com/AhmedHashish123/VQ-Diffusion/blob/main/Samples/Pretrained/pretrained_2/000002.png)
![alt text](https://github.com/AhmedHashish123/VQ-Diffusion/blob/main/Samples/Pretrained/pretrained_2/000003.png)

### Pretrained Inference Improved VQ-Diffusion with high-quality inference:<br />
![alt text](https://github.com/AhmedHashish123/VQ-Diffusion/blob/main/Samples/Pretrained/pretrained_3/000000.png)
![alt text](https://github.com/AhmedHashish123/VQ-Diffusion/blob/main/Samples/Pretrained/pretrained_3/000001.png)
![alt text](https://github.com/AhmedHashish123/VQ-Diffusion/blob/main/Samples/Pretrained/pretrained_3/000002.png)
![alt text](https://github.com/AhmedHashish123/VQ-Diffusion/blob/main/Samples/Pretrained/pretrained_3/000003.png)

### Pretrained Inference Improved VQ-Diffusion with fast inference:<br />
![alt text](https://github.com/AhmedHashish123/VQ-Diffusion/blob/main/Samples/Pretrained/pretrained_4/000000.png)
![alt text](https://github.com/AhmedHashish123/VQ-Diffusion/blob/main/Samples/Pretrained/pretrained_4/000001.png)
![alt text](https://github.com/AhmedHashish123/VQ-Diffusion/blob/main/Samples/Pretrained/pretrained_4/000002.png)
![alt text](https://github.com/AhmedHashish123/VQ-Diffusion/blob/main/Samples/Pretrained/pretrained_4/000003.png)

### Pretrained Inference Improved VQ-Diffusion with purity sampling:<br />
![alt text](https://github.com/AhmedHashish123/VQ-Diffusion/blob/main/Samples/Pretrained/pretrained_5/000000.png)
![alt text](https://github.com/AhmedHashish123/VQ-Diffusion/blob/main/Samples/Pretrained/pretrained_5/000001.png)
![alt text](https://github.com/AhmedHashish123/VQ-Diffusion/blob/main/Samples/Pretrained/pretrained_5/000002.png)
![alt text](https://github.com/AhmedHashish123/VQ-Diffusion/blob/main/Samples/Pretrained/pretrained_5/000003.png)

### Pretrained Inference Improved VQ-Diffusion with both learnable classifier-free sampling and fast inference:<br />
![alt text](https://github.com/AhmedHashish123/VQ-Diffusion/blob/main/Samples/Pretrained/pretrained_6/000000.png)
![alt text](https://github.com/AhmedHashish123/VQ-Diffusion/blob/main/Samples/Pretrained/pretrained_6/000001.png)
![alt text](https://github.com/AhmedHashish123/VQ-Diffusion/blob/main/Samples/Pretrained/pretrained_6/000002.png)
![alt text](https://github.com/AhmedHashish123/VQ-Diffusion/blob/main/Samples/Pretrained/pretrained_6/000003.png)

### Custom Inference VQ-Diffusion:<br />
![alt text](https://github.com/AhmedHashish123/VQ-Diffusion/blob/main/Samples/Custom/custom_1/000000.png)
![alt text](https://github.com/AhmedHashish123/VQ-Diffusion/blob/main/Samples/Custom/custom_1/000001.png)
![alt text](https://github.com/AhmedHashish123/VQ-Diffusion/blob/main/Samples/Custom/custom_1/000002.png)
![alt text](https://github.com/AhmedHashish123/VQ-Diffusion/blob/main/Samples/Custom/custom_1/000003.png)

### Custom Inference Improved VQ-Diffusion with learnable classifier-free sampling:<br />
![alt text](https://github.com/AhmedHashish123/VQ-Diffusion/blob/main/Samples/Custom/custom_2/000000.png)
![alt text](https://github.com/AhmedHashish123/VQ-Diffusion/blob/main/Samples/Custom/custom_2/000001.png)
![alt text](https://github.com/AhmedHashish123/VQ-Diffusion/blob/main/Samples/Custom/custom_2/000002.png)
![alt text](https://github.com/AhmedHashish123/VQ-Diffusion/blob/main/Samples/Custom/custom_2/000003.png)

### Custom Inference Improved VQ-Diffusion with high-quality inference:<br />
![alt text](https://github.com/AhmedHashish123/VQ-Diffusion/blob/main/Samples/Custom/custom_3/000000.png)
![alt text](https://github.com/AhmedHashish123/VQ-Diffusion/blob/main/Samples/Custom/custom_3/000001.png)
![alt text](https://github.com/AhmedHashish123/VQ-Diffusion/blob/main/Samples/Custom/custom_3/000002.png)
![alt text](https://github.com/AhmedHashish123/VQ-Diffusion/blob/main/Samples/Custom/custom_3/000003.png)

### Custom Inference Improved VQ-Diffusion with fast inference:<br />
![alt text](https://github.com/AhmedHashish123/VQ-Diffusion/blob/main/Samples/Custom/custom_4/000000.png)
![alt text](https://github.com/AhmedHashish123/VQ-Diffusion/blob/main/Samples/Custom/custom_4/000001.png)
![alt text](https://github.com/AhmedHashish123/VQ-Diffusion/blob/main/Samples/Custom/custom_4/000002.png)
![alt text](https://github.com/AhmedHashish123/VQ-Diffusion/blob/main/Samples/Custom/custom_4/000003.png)

### Custom Inference Improved VQ-Diffusion with purity sampling:<br />
![alt text](https://github.com/AhmedHashish123/VQ-Diffusion/blob/main/Samples/Custom/custom_5/000000.png)
![alt text](https://github.com/AhmedHashish123/VQ-Diffusion/blob/main/Samples/Custom/custom_5/000001.png)
![alt text](https://github.com/AhmedHashish123/VQ-Diffusion/blob/main/Samples/Custom/custom_5/000002.png)
![alt text](https://github.com/AhmedHashish123/VQ-Diffusion/blob/main/Samples/Custom/custom_5/000003.png)

### Custom Inference Improved VQ-Diffusion with both learnable classifier-free sampling and fast inference:<br />
![alt text](https://github.com/AhmedHashish123/VQ-Diffusion/blob/main/Samples/Custom/custom_6/000000.png)
![alt text](https://github.com/AhmedHashish123/VQ-Diffusion/blob/main/Samples/Custom/custom_6/000001.png)
![alt text](https://github.com/AhmedHashish123/VQ-Diffusion/blob/main/Samples/Custom/custom_6/000002.png)
![alt text](https://github.com/AhmedHashish123/VQ-Diffusion/blob/main/Samples/Custom/custom_6/000003.png)

## Name & Link of The Training Set:<br />

### Name:<br />
**COCO 2014<br />

- Download the [dataset](https://cocodataset.org/#download), choose:
  - 2014 Train images
  - 2014 Val iamges
  - 2014 Train/Val annotations
- "2014 Train images" is a compressed file containing a folder called "train 2014".
- "2014 Val images" is a compressed file containing a folder called "val 2014".
- "2014 Train/Val annotations" is a compressed file containing .JSON files. You only need two:
  - "captions_train2014.json"
  - "captions_val2014.json"

## Number of Model Parameters:<br />

This project contains two main models:<br />
- VQ-VAE
- VQ-Diffusion

I trained the VQ-Diffusion model which contains:<br />
- content_codec: 65.8 million parameters
- condition_codec: 0
- transformer: 431.3 million parameters

These parameters add up to 497.1 million.<br />

## Model Evaluation Metric:<br />

Variational Bayes loss is used in this project. To get this loss, Kullback-Leibler (KL) divergence is calculated.

## Running:<br />

- To run the web app type:<br />
  streamlit run web_app.py<br />
  
- To run through CLI type:<br />
  python infer.py "your text decription" "number of images"<br />
  
  Example:<br />
  python infer.py "A group of elephants walking in muddy water" 4<br />

  
