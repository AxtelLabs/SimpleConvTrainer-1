# -*- encoding: utf-8 -*-
# trainutils.py

import argparse
import datetime

description = '''
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
'''

def get_argv():
    '''
    Get the program argument values.

    returns
    -------
        argv: an object (Namespace) with the parsed arguments
            as attributes.
    '''
    
    
    def string2bool(value):
        if isinstance(value, bool):
            return value
        if value.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif value.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')


    ap = argparse.ArgumentParser(
        prog = 'python(3.6) trainer/main.py', formatter_class = argparse.RawDescriptionHelpFormatter,
        description = description
    )
    ap.add_argument(
        '-a', '--architecture', required = True, type = str,
        help = 'the architecture of the CNN to be trained: `alexnet`, `lenet`, `minivgg`, ' \
            + '`karpathynet`, `resnet50`, `vgg16` and `custom` (defined by user in visutils/neuralnets/conv/customnet.py)'
    )
    ap.add_argument(
        '-d', '--dataset', required = True, type = str,
        help = 'path to input dataset.'
    )
    ap.add_argument(
        '-n', '--normalize', default = True, type = string2bool,
        help = 'normalize the input images intensities or not (default is True).'
    )
    ap.add_argument(
        '-mn', '--model-name', required = True, type = str,
        help = 'name [syntax] of output model file.'
    )
    ap.add_argument(
        '-g', '--grayscale', default = False, type = string2bool,
        help = 'load images in grayscale or not (default is False).'
    )
    ap.add_argument(
        '-i', '--image-size', default = 64, type = int,
        help = 'the size of the images in pixels (width and height) for the input tensor.'
    )
    ap.add_argument(
        '-o', '--optimizer', default = 'SGD', type = str,
        help = 'the optimizer to use for the gradient descent variant step \
        for optimization. Supported optimizers: `adam` and `sgd`.'
    )
    ap.add_argument(
        '-etha', '--etha', default = 0.001, type = float,
        help = 'the learning rate for optmizer during the gradient descent step.'
    )
    ap.add_argument(
        '-r', '--reduce-lr', default = False, type = string2bool,
        help = 'reduce the learning rate on validation loss plateau. `factor` = 0.2, \
        check Callbacks if you want to modify it.'
    )
    ap.add_argument(
        '-e', '--epochs', default = 30, type = int,
        help = 'number of epochs to train the network.'
    )
    ap.add_argument(
        '-es', '--early-stop', default = 0, type = int,
        help = 'Use early stopping callback while training (default is 0). The integer' \
            + 'indicates the patience.'
    )
    ap.add_argument(
        '-b', '--batch-size', default = 32, type = int,
        help = 'the size of the batch to train with stochastic gradient descent.'
    )
    ap.add_argument(
        '-tb', '--tensorboard', default = False, type = string2bool,
        help = 'enable weight histogram monitoring with tensorboard visualizer'
    )
    argv = ap.parse_args()

    return argv



def build_architecture(argv, width:int, height:int, depth:int, classes:int):
    '''
    Build the chosen architecture model.

    arguments
    ---------
        argv: the argument values the program was called with
        width: width of input images
        height: height of the input images
        depth: depth (no. of channels) of the input images
        classes: number of classes of the corresponding data
    returns
    -------
        model: the correspondant keras implementation model
            compatible with the given inputs.
    '''
    if argv.architecture == 'lenet':
        from visutils.neuralnets.conv import LeNet
        model = LeNet.build(
            width = width,
            height = height,
            depth = depth,
            classes = classes
        )
    
    elif argv.architecture == 'karpathynet':
        from visutils.neuralnets.conv import KarpathyNet
        model = KarpathyNet.build(
            width = width,
            height = height,
            depth = depth,
            classes = classes,
            dropout = True
        )
    
    elif argv.architecture == 'minivgg':
        from visutils.neuralnets.conv import MiniVGGNet
        model = MiniVGGNet.build(
            width = width,
            height = height,
            depth = depth,
            classes = classes
        )
    
    elif argv.architecture == 'alexnet':
        from visutils.neuralnets.conv import AlexNet
        model = AlexNet.build(
            width = width,
            height = height,
            depth = depth,
            classes = classes,
            reg = 0.0002
        )
    
    elif argv.architecture == 'vgg16':
        from visutils.neuralnets.conv import VGG16
        model = VGG16.build()
    
    elif argv.architecture == 'resnet50':
        from visutils.neuralnets.conv import ResNet50
        model = ResNet50.build(
            width = width,
            height = height,
            depth = depth,
            classes = classes
        )
    
    elif argv.architecture == 'custom':
        from visutils.neuralnets.conv import CustomNet
        model = CustomNet.build(
            width = width,
            height = height,
            depth = depth,
            classes = classes
        )
    
    return model



def build_callbacks(argv):
    '''
    '''
    from keras.callbacks import ModelCheckpoint
    from keras.callbacks import ReduceLROnPlateau
    from keras.callbacks import TensorBoard
    from keras.callbacks import EarlyStopping
    from visutils.callbacks import TrainingPlot
    from visutils.callbacks import  EpochCheckpoint

    modelpath = f'output/models/{argv.model_name}.h5'
    plotpath = f'output/plots/{argv.model_name}.png'
    callbacks = [
        ModelCheckpoint(
            filepath = modelpath,
            save_best_only = True,
            verbose = True
        ),
        TrainingPlot(
            figPath = plotpath
        )
    ]

    if argv.reduce_lr:
        callbacks.append(
            ReduceLROnPlateau(
                monitor = 'val_loss',
                factor = 0.2,
                patience = 10,
                cooldown = 3,
                verbose = True
            )
        )
    
    if argv.tensorboard:
        callbacks.append(
            TensorBoard(
                log_dir = './logs',
                histogram_freq = 5,
                verbose = True
            )
        )
    
    if argv.early_stop:
        callbacks.append(
            EarlyStopping(
                monitor = 'val_loss',
                patience = argv.early_stop,
                verbose = True
            )
        )

    return callbacks


def load_images_and_labels(argv, imagepaths, normalize = True):
    '''
    '''
    from visutils.preprocessing import ImageToArrayPreprocessor
    from visutils.preprocessing import AspectAwarePreprocessor
    from visutils.datasets import SimpleDatasetLoader
    # initializing the image preprocessors
    imagesize = argv.image_size
    aap = AspectAwarePreprocessor(imagesize, imagesize)
    iap = ImageToArrayPreprocessor()

    # load the dataset from disk to memory
    # then normalize raw pixel intensities
    print('[INFO] loading images...')
    sdl = SimpleDatasetLoader(
        preprocessors = [aap, iap], grayscale = argv.grayscale
    )
    (data, labels) = sdl.load(
        imagepaths, verbose = 500
    )

    if normalize:
        print('[INFO] normalizing data...')
        data = data.astype('float') / 255.0

    return data, labels



def write_training_report(argv, modelparams, report):
    '''
    '''
    imagesize = modelparams['imagesize']
    with open(f'output/reports/{argv.model_name}_report.txt', 'w') as f:
        f.write(f"REPORT FOR MODEL: {argv.model_name}\n\n")
        f.write(f"DATETIME: {datetime.datetime.now()}\n\n")
        
        f.write("             ========   PROGRAM ARGUMENTS   ========\n\n")
        for arg, val in argv._get_kwargs():
            f.write(f'{arg + ":":12}    {val}\n')
        f.write("\n\n")
        f.write("             ======== MODEL HYPERPARAMETERS ========\n\n")
        f.write(f"image size:\t{str(imagesize)} * {str(imagesize)}\n")
        f.write(f"epochs:\t{argv.epochs:10}\n")
        f.write(f"optimizer:\t{argv.optimizer:10}\n\n")
        
        optconfig = modelparams['optimizer'].get_config()
        for ft in optconfig:
            f.write(f"\t\t\t* {ft:10}: \t{optconfig[ft]:10}\n")
        f.write(f"\nloss function:\t{modelparams['loss_function']:10}\n")
        f.write(f"metrics:\t{str(modelparams['metrics']):10}\n\n\n")
        f.write("             ========    ELAPSED TRAINING TIME ========\n\n")
        f.write(f"elapsed training time:\t{modelparams['elapsed_time'] / 60:10.3f} minutes\n\n\n")
        f.write("             ========     MODEL SUMMARY  ========\n\n")
        modelparams['model_structure'].summary(print_fn=lambda x: f.write(x + '\n'))
        f.write("\n\n")
        f.write("             ======== CLASSIFICATION REPORT ========\n\n")
        f.write(report)
        f.write("\n\n")
