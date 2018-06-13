Scripts for this project are divided in three categories:
A) Model building script: modelmlfn.py
This is the file that generates the actual model. The main method in this file that can be used to generate the file is MLFN. For example, to generate the mode with input size (256,128,3), depth 16, cardinality 16, width 16 and T_dimension 1000, we call as:
model = MLFN((256, 128, 3), depth=[3,4,6,3], cardinality=8, width=64, T_dimension=1000)

For details on these parameters, please refer to the paper:https://arxiv.org/abs/1803.09132

B) Training scripts: gpuTraining.py, training.py
This is the file that consumes the model generated above. In this file, we first generate the model with required dimensions and then load the input images into an array. Then, this array is passed in batches to parallel_model.fit_generator() method to train the model. As we will be using the gpu to train the model, we initialize our model as 'multi_gpu_model'. Then we save the trained model as <modelname.ckpt> file. Similarly, for training on cpu, use the training.py file. Its very similar to gpuTraining.py except that multi_gpu_model() function call is not used.
usage: python gpuTraining.py


C) Evaluation scripts: differentcheckEvaluate.py, samecheckEvaluate.py, AllEvaluate.py
These are the scripts that we used to evalute the trained model. After the model is loaded by load_model(), it is used to extract the features of query images in query.list file and test images in test.list file. These features are extracted using the method feature_extract(). Then, by using a loop, we identify the ground truth for query images. Then we calculate the mAP score and Rank1 score for queries. Note: please change the name of files in the script where you want to save your results. This script needs to be modified to take filename as command line argument.
usage: python xxxEvaluate.py