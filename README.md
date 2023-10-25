# RetNet_ViT
# RMT paper: https://arxiv.org/pdf/2309.11523.pdf
## Precautions
I used is a data set of five kinds of flowers. You can download the specific data set at this URL: ___https://www.kaggle.com/datasets/alxmamaev/flowers-recognition___.
In addition, in my ___estimate_model.py___ script, the ___predict_single_image___ function. I randomly select an image，which is under the rose category. And ___rename___ it ___rose.jpg___. If you want to achieve prediction of a single image, ___remember to change the file name when using this function___.

### This code is mainly based on the PyTorch framework to reproduce the RMT model, which includes the RetNet model and the RMT attention module. With this implementation you can easily use it to train and validate your own datasets.
### There is still a lot of places for optimization in this code. If you have any good suggestions, you are welcome to give them to me.

## Train and evaluate the RMT-model
1. cd RetNet_ViT
2. python train_gpu.py
