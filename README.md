# Tensorflow 2 Object Detection API Tutorial 

[![Python 3.6](https://img.shields.io/badge/Python-3.6-3776AB)](https://www.python.org/downloads/release/python-360/)
[![TensorFlow 2.2](https://img.shields.io/badge/TensorFlow-2.2-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v2.2.0)

## Introduction


With the [announcement](https://blog.tensorflow.org/2020/07/tensorflow-2-meets-object-detection-api.html) that [Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) is now compatible with Tensorflow 2,
I tried to test the new models published in the [TF2 model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md), and train them with my custom data.
However, I have faced some problems as the scripts I have for Tensorflow 1 is not working with Tensorflow 2 (which is not surprising!), in addition to having very poor documentation and tutorials from tensorflow models repo.
In this repo I am sharing my experience, in addition to providing clean codes to run the inference and training object detection models using Tensorflow 2.
 
This tutorial should be useful for those who have experience with the API but cannot find clear examples for the new changes to use it with Tensorflow 2.
However, I will add all the details and working examples for the new comers who are trying to use the object detection api for the first time, so hopefully this tutorial will make it easy for beginners to get started and run their object detection models easily.


## Roadmap

This tutorial will take you from installation, to running pre-trained detection model, then training your model with a custom dataset, and exporting your model for inference.

1. [Installation](#installation)
2. [Inference with pre-trained models](#inference-with-pre-trained-models)
3. [Preparing your custom dataset for training](#preparing-your-custom-dataset-for-training)
4. [Training object detection model with your custom dataset](#training-object-detection-model-with-your-custom-dataset)
5. [Exporting your trained model for inference](#exporting-your-trained-model-for-inference)


## Installation

The examples in this repo is tested with python 3.6 and Tensorflow 2.2.0, but it is expected to work with other Tensorflow 2.x versions with python version 3.5 or higher.

It is recommended to install [anaconda](https://www.anaconda.com/products/individual) and create new [environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) for your projects that use the installed packages:

```bash
# create new environment
conda create --name tf2 python=3.6

# activate your environment before installation or running your scripts 
conda activate tf2
``` 


PLEASE DO can check the compatible versions of any tensorflow version with cuda and cudnn versions from [here](https://www.tensorflow.org/install/source#tested_build_configurations).
If you have the matching CUDA and CUDNN versions in your system, install tensorflow-gpu as follows: 

```bash
# if you have NVIDIA GPU with cuda 11.2 and cudnn 8.1
pip install tensorflow-gpu==2.5.0
apt install --allow-change-held-packages libcudnn8=8.1.0.77-1+cuda11.2
```

Otherwise, a great feature of Anaconda is that it can automatically install a local version of cudatoolkit that is compatible with your tensorflow version (But you should have the proper nvidia gpu drivers installed).
But be aware that this cuda version will be only available to your python environment (or virtual environment if you created one), and will not be available for other python versions or other environments.



Clone the tensorflow models git repository & Install TensorFlow Object Detection API


# clone the tensorflow models on the colab cloud vm
!git clone --q https://github.com/tensorflow/models.git

Make sure you have [protobuf compiler](https://grpc.io/docs/protoc-installation/#install-using-a-package-manager) version >= 3.0, by typing `protoc --version`, or install it on Ubuntu by typing `apt install protobuf-compiler`.

Then proceed to the python package installation as follows:

```bash
#navigate to /models/research folder to compile protos
cd models/research

# Compile protos.
protoc object_detection/protos/*.proto --python_out=.

# Install TensorFlow Object Detection API.
cp object_detection/packages/tf2/setup.py .
python -m pip install .

```

The previous commands installs the object detection api as a python package that will be available in your python environment (or virtual environment if you created one),
and will automatically install all required dependencies if not already installed.

Finally, to test that your installation is correct, type the following command: 

```bash
# Test the model builder.
python object_detection/builders/model_builder_tf2_test.py
```

For more installation options, please refer to the original [installation guide](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md).

To run the examples in this repo, you will need some additional dependencies:

```bash
# install OpenCV python package
pip install opencv-python
pip install opencv-contrib-python
```
------------------------------------------------------------

## Inference with pre-trained models

To go through the tutorial, clone this tutorial repo and follow the instructions step by step. 
## Training object detection model with your custom dataset
```bash
git clone https://github.com/irwanmazlin/tf2-object-detection-api-tutorial.git
```

You can download any of the models from the table in the [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md), and place it in the [models/](models) directory. For example, let's download the EfficientDet D0 model.

you can use my gather data by using command below

 
```bash
cd tf2-object-detection-api-tutorial
mkdir /train/
cd /train/
curl -L "https://app.roboflow.com/ds/QgK44FPVFD?key=AV0JrcEmz5" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip
```



To get started with the Object Detection API with TF2, let's download one of the models pre-trained with coco dataset from the [tf2 detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md), and use it for inference.

```bash
cd ..
cd models/
# download the model
wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz
# extract the downloaded file
tar -xzvf ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz
```


The labelmap file and the modified configuration files are added to this repo. 
You can find them in [models/raccoon_labelmap.pbtxt](models/raccoon_labelmap.pbtxt) and [models/ssd_mobilenet_v2_raccoon.config](models/ssd_mobilenet_v2_raccoon.config).

Once you prepare the configuration file, you can start the training by typing the following commands: 

```bash
# you should run training scriptd from train_tf2/ directory
cd train_tf2/
bash start_train.sh
```

You can notice that it actually runs the [model_main_tf2.py](train_tf2/model_main_tf2.py), 
which I copied from the object detection api repo that we cloned at the beginning (directly in the object_detection folder), and you can also download it from [here](https://github.com/tensorflow/models/blob/master/research/object_detection/model_main_tf2.py).

It is also recommended to run the validation script along with the training scripts.
The training script saves a checkpoint every _n_ steps while training, and this value can be specified in the parameter `--checkpoint_every_n`.
While training is running, the validation script reads these checkpoints when they are available, and uses them to evaluate the model accuracy at this checkpoint using the validation set (from the validation tfrecord file).
This will help us to monitor the training progress by printing the validation mAP on the terminal, and by using a GUI monitoring package like [tensorboard](https://www.tensorflow.org/tensorboard/get_started) as we will see.  

To run the validation script along with training script, open another terminal and run:

```bash
bash start_eval.sh
```

Finally, it is time to see how our training is progressing, which is very easy task using [tensorboard](https://www.tensorflow.org/tensorboard/get_started). 
Tensorboard reads the training and evaluation log files written by tensorflow, and draws different curves showing the progress of the training loss values (lower is better), and validation accuracy or in our case mean average precision (higher is better).
To run the tensorboard, just open new terminal window and run the command:

```bash
tensorboard --logdir=models/training
```

the `--logdir` argument should point to the same directory as passed to the `--model_dir` argument used in training and validation scripts.
The training and validation scripts write their logs in separate folders inside this directory, then tensorboard reads these logs to plot the curves.

When you run the tensorboard command, it will not show any GUI, but will give you a link (something like `http://localhost:6006/ `) that you can copy and paste in your favourite internet browser.
Then you can see all the curves for training and validation. Here is how the training loss evolved with the steps: 

![train_loss](data/samples/docs/train_loss.png)

For the validation mAP (mean average precision) with the saved checkpoints (a checkpoint saved each 500 steps), you can see the next curve which represents mAP@0.5IoU. 
Note that here we have only one class, so we actually have the average precision (AP) for this class.
mAP@0.5IoU means that detection boxes are considered good detections (True positive) if their Intersection over Union (IoU) with the ground truth box is 0.5 or higher.
I recommend reading this [article](https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/), which explains the idea of Intersection over Union in object detection.

![mAP](data/samples/docs/val_precision.png)


The training took around 20 minutes on my laptop with NVIDIA GTX 1050Ti (4GB of GPU RAM), and intel Core i7 CPU.
The training runs for 3000 steps, which I found enough to get around 90% mAP on validation set.
If you are a beginner, I recommend that you play with the training parameters and try to figure out the effect of your parameter values on the results and training curves. 


The final mAP is around 92% which is pretty good. Now let's see how to use this trained model for inference to detect raccoons from images and videos.  


-------

## Exporting your trained model for inference

When the training is done, Tensorflow saves the trained model as a [checkpoint](https://www.tensorflow.org/guide/checkpoint).
Now we will see how to export the models to a format that can be used for inference, this final format usually called saved model or frozen model.

To generate the frozen model, we need to run [train_tf2/exporter_main_v2.py](train_tf2/exporter_main_v2.py),
which I just copied from the API code, and you can download it from [here](https://github.com/tensorflow/models/blob/master/research/object_detection/exporter_main_v2.py).
For convenience, I prepared the shell script [train_tf2/export_model.sh](train_tf2/export_model.sh) to run exporter code, and pass the required arguments.
So simply start the inference by running this shell script.

```bash
cd train_tf2/
bash export_model.sh
```  

The [export_model.sh](train_tf2/export_model.sh) file contains:

```bash
model_dir=../models/ssd_mobilenet_v2_raccoon
out_dir=$model_dir/exported_model
mkdir -p $out_dir

# start the exporter
python exporter_main_v2.py \
    --input_type="image_tensor" \
    --pipeline_config_path=$model_dir/pipeline.config \
    --trained_checkpoint_dir=$model_dir/ \
    --output_directory=$out_dir
```




