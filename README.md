### Object Detection with a Transformer Architecture   

In this lesson you will explore a simple case of a farily state of the art object detection algorithm. This algorithm is based on a transformer architecture. 

Transformer architectures are starting to dominat several areas of computer visions (CV). However, massive resources are required to effectively train these models. Here we will work with small model trained on a relativeluy small dataset. Our focus is on developing understanding of how vision transformers work, rather than optimizing performance.   

The model we are working with uses 8 self-attention layers in the transformers. This model is based on the, now well know vision transformater model (ViT model), from [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale, Dosovitskiy, et.al., 2020](https://arxiv.org/abs/2010.11929) The model will be trained on the relatively small [CIFAR 100 dataset](https://www.cs.toronto.edu/~kriz/cifar.html), containing samples of 100 object types. Both of these conditions will limit the accuacy of the trained model.       

A number of [Keras-based vision transformer models from Google Research](https://github.com/google-research/vision_transformer) are available. These models have been trained on massive datasets, and can be fine-tuned for spedicic problems. But be warned, working with these sophisticated models can be difficult. 

The models in the Google GitHub repository include the model of [Dosovitskiy, et.al., 2020](https://arxiv.org/abs/2010.11929). We used for this example. The details of pretraining the model in the repository are outlined in Appendix B of the paper, and include:  
1. The semisupervised pretraning was performed for 1 million steps. 
2. The semisupervised pretraining was performed using two very large dataset, the [InageNet 21k dataset](https://github.com/Alibaba-MIIL/ImageNet21K/blob/main/dataset_preprocessing/processing_instructions.md), size 1.3 TB, and [JFT dataset](https://paperswithcode.com/dataset/jft-300m) of 300 million images. 
     
In these exercises, we will use a variation of the ViT architecture suitable for opject detection. No pre-training is available for this model. Dispite training from scratch on a relaively small dataset the results achieved are reasonably good, but not state-of-the-art.   

