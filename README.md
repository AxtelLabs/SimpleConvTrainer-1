# SimpleConvTrainer

A simple CNN trainer optimized for vision purpousses: fast for prototyping
in a simple way.
What I use this program/utils for is to explore the hyperparameter tunning space
in a broad way,
then choose some prospective model and continue the hyperparameter adjustment 
in a more detailed manner.Of Course, after exploring the hyperparameters, we can change the structure of the
CNN architecture if necessary.


It can be also useful for learning purpouses: How do hyperparameters affect the
performance of a CNN model? Do I normalize the image? What about the batch size?
learning rate?
Simply, call the program with different argument
values and the plots, models and reports will be generated.


Available CNN architectures:
```
- lenet
- minivgg
- karpathynet
- alexnet
- ResNet50
- vgg16 (fixed size)
```

You can change the network architecture class in:
```visutils/neuralnets/conv/{architecture}.py``` file to better suit the project needs.


## Requirements

Required Python libraries:
+ numpy
+ TensorFlow
+ Keras
+ OpenCV
+ scikit-learn
+ scikit-image
+ imutils
+ imageio

## USAGE
First source the environment with
```
source env.sh
```
this will activate the vision environment and
initialize the `PYTHONPATH`.

To see program instructions:
```
python trainer/main.py --help
````
```
usage: python(3.6) trainer/main.py [-h] -a ARCHITECTURE -d DATASET
                                   [-n NORMALIZE] -mn MODEL_NAME
                                   [-g GRAYSCALE] [-i IMAGE_SIZE]
                                   [-o OPTIMIZER] [-etha ETHA] [-r REDUCE_LR]
                                   [-e EPOCHS] [-es EARLY_STOP]
                                   [-b BATCH_SIZE] [-tb TENSORBOARD]

Convolutional Neural Network trainer for a specific architecture using
TensorFlow as the backend engine and OpenCV for vision utilities.
                                        -msantosh@axtellabs.
The supported architectures are:
    - MiniVGG
    - LeNet
    - AlexNet
    - KarpathyNet
    - VGG16 (fixed size input {224, 224, 3} and 1000 classes.)

All architectures (until now) can be specified their input shapes
and their output dimension (number of classes), except for the VGG16.

optional arguments:
  -h, --help            show this help message and exit
  -a ARCHITECTURE, --architecture ARCHITECTURE
                        the architecture of the CNN to be trained: `alexnet`,
                        `lenet`, `minivgg`, `karpathynet`, `resnet50`,
                        `vgg16` and `custom` (defined by user in visutils/neuralnets/conv/customnet.py)
  -d DATASET, --dataset DATASET
                        path to input dataset.
  -n NORMALIZE, --normalize NORMALIZE
                        normalize the input images intensities or not (default
                        is True).
  -mn MODEL_NAME, --model-name MODEL_NAME
                        name [syntax] of output model file.
  -g GRAYSCALE, --grayscale GRAYSCALE
                        load images in grayscale or not (default is False).
  -i IMAGE_SIZE, --image-size IMAGE_SIZE
                        the size of the images in pixels (width and height)
                        for the input tensor.
  -o OPTIMIZER, --optimizer OPTIMIZER
                        the optimizer to use for the gradient descent variant
                        step for optimization. Supported optimizers: `adam`
                        and `sgd`.
  -etha ETHA, --etha ETHA
                        the learning rate for optmizer during the gradient
                        descent step.
  -r REDUCE_LR, --reduce-lr REDUCE_LR
                        reduce the learning rate on validation loss plateau.
                        `factor` = 0.2, check Callbacks if you want to modify
                        it.
  -e EPOCHS, --epochs EPOCHS
                        number of epochs to train the network.
  -es EARLY_STOP, --early-stop EARLY_STOP
                        Use early stopping callback while training (default is
                        0). The integerindicates the patience.
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        the size of the batch to train with stochastic
                        gradient descent.
  -tb TENSORBOARD, --tensorboard TENSORBOARD
                        enable weight histogram monitoring with tensorboard
                        visualizer
```

#

Training example (simple training):
```
python3 trainer/main.py \
    --architecture lenet \
    --dataset datasets/{the_dataset_name} \
    --model-name {the_model_name} \
    --etha 0.001 \
    --epochs 100
```

Training example (loop for sequential training):
this loop reads values, convert them to a string and assign them to variables
to use in the loop. Here we train 4 models with different etha (learning rate)
and we name them differently
```
for vars in 0.001,02 0.0001,hola 0.0005,05 0.00007,redneuronal
do
    IFS=',' read etha suffix <<< "${vars}";
    python3 trainer/main.py \
        --architecture lenet \
        --dataset datasets/NCWV2 \
        --model-name NCWV2_minivgg_ch3_it150_aug_${suffix}.h5 \
        --epochs 150 \
        --etha ${etha} \
        --optimizer adam
done
```

#

## OUTPUTS
The program will output three files
+ A {.h5} file in ```output/model/{filename}``` which represents the keras
compatible CNN model. This file is generated by a callback, which saves the model
if there is improvement in the validation loss metric with the same file name.
(Save best model only)

+ A {.txt} file in ```output/report/{filename}``` which represents the training model
report and the test validation information. This file is generated once at fhe final
stage of the program: after the testing of the model.

+ A {.png} file in ```output/plot/{filename}``` which represents the training plot
metrics monitor over the training phase. This file is generated each epoch
to monitor the training scalar metrics.


## Program structure
```
├── env.sh
├── datasets
│   └── {datasetname}
│       └── class1
│            └── image000.png
│            └── image001.png
│            └── image002.png
│            └── ...
│       └── class2
│       └── ...
│       └── classn
├── logs
├── output
│   ├── models
│   ├── plots
│   └── reports
├── simpleconv
│   └── main.py
│   └── trainutils.py
└── visutils
    ├── callbacks
    ├── datasets
    ├── neuralnets
    │   └── conv
    └── preprocessing
```


#
## Comments:
Until now I have to fix the creation of the log directory if it doens't exist
for the tensorboard option, that callback produces a log file to be read by tensorboard
in the path ```./logs/```.