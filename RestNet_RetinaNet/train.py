#!/usr/bin/env python 
from __future__ import division

import argparse
import os
import sys
from pprint import pprint

import keras
from keras.utils import multi_gpu_model
import keras.preprocessing.image
from keras_retinanet import layers
from keras_retinanet import losses

import tensorflow as tf
import warnings
import keras_retinanet.losses
import keras_retinanet.layers
from keras_retinanet.callbacks import RedirectModel
from keras_retinanet.preprocessing.csv_generator import CSVGenerator
from keras_retinanet.models.resnet import resnet50_retinanet
from keras_retinanet.utils.keras_version import check_keras_version
from keras_retinanet.utils.model import freeze as freeze_model
from keras_retinanet.models.resnet import resnet_retinanet as retinanet


# parameters
batch_size = 1
steps_per_epoch = 10000
epochs = 50
steps_per_training_epoch = 1500
training_epochs = 10

# paths
boring_repository_dir = os.getcwd() # you can set this manually if you're not calling the script from the top repo directory
#boring_dataset_dir = os.path.join(boring_repository_dir, 'pandg-dataset')
boring_annotations_path = os.path.join(boring_repository_dir, 'consolidated_annotation.csv')
boring_classes_path = os.path.join(boring_repository_dir, 'consolidated_classes.csv')
boring_snapshots_dir = os.path.join(boring_repository_dir, 'snapshots')

assert os.path.isfile(boring_annotations_path), boring_annotations_path
assert os.path.isfile(boring_classes_path), boring_classes_path

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

def model_with_weights(model, weights, skip_mismatch):
    if weights is not None:
        model.load_weights(weights, by_name=True, skip_mismatch=skip_mismatch)
    return model


def download_imagenet():
        """ Downloads ImageNet weights and returns path to weights file.
        """
        resnet_filename = 'ResNet-{}-model.keras.h5'
        resnet_resource = 'https://github.com/fizyr/keras-models/releases/download/v0.0.1/{}'.format(resnet_filename)
        depth = int('50')

        filename = resnet_filename.format(depth)
        resource = resnet_resource.format(depth)
        if depth == 50:
            checksum = '3e9f4e4f77bbe2c9bec13b53ee1c2319'
        elif depth == 101:
            checksum = '05dc86924389e5b401a9ea0348a3213c'
        elif depth == 152:
            checksum = '6ee11ef2b135592f8031058820bb9e71'

        return keras.applications.imagenet_utils.get_file(
            filename,
            resource,
            cache_subdir='models',
            md5_hash=checksum
        )

#def create_models(num_classes):
#    # create "base" model (no NMS)
#    image = keras.layers.Input((None, None, 3))
#
#    model = resnet50_retinanet(inputs=image, num_classes=num_classes)
#    training_model = model
#
#    # append NMS for prediction only
#    classification   = model.outputs[1]
#    detections       = model.outputs[2]
#    boxes            = keras.layers.Lambda(lambda x: x[:, :, :4])(detections)
#    detections       = keras_retinanet.layers.NonMaximumSuppression(name='nms')([boxes, classification, detections])
#    prediction_model = keras.models.Model(inputs=model.inputs, outputs=model.outputs[:2] + [detections])
#
#    # compile model
#    training_model.compile(
#        loss={
#            'regression'    : keras_retinanet.losses.smooth_l1(),
#            'classification': keras_retinanet.losses.focal()
#        },
#        optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001)
#    )
#
#    return model, training_model, prediction_model


def create_models(backbone_retinanet, backbone, num_classes, weights, multi_gpu=0, freeze_backbone=False):
    modifier = freeze_model if freeze_backbone else None

    # Keras recommends initialising a multi-gpu model on the CPU to ease weight sharing, and to prevent OOM errors.
    # optionally wrap in a parallel model
    if multi_gpu > 1:
        with tf.device('/cpu:0'):
            model = model_with_weights(backbone_retinanet(num_classes, backbone=backbone, modifier=modifier), weights=weights, skip_mismatch=True)
        training_model = multi_gpu_model(model, gpus=multi_gpu)

        # append NMS for prediction only
        classification   = model.outputs[1]
        detections       = model.outputs[2]
        boxes            = keras.layers.Lambda(lambda x: x[:, :, :4])(detections)
        detections       = layers.NonMaximumSuppression(name='nms')([boxes, classification, detections])
        prediction_model = keras.models.Model(inputs=model.inputs, outputs=model.outputs[:2] + [detections])
    else:
        model            = model_with_weights(backbone_retinanet(num_classes, backbone=backbone, modifier=modifier), weights=weights, skip_mismatch=True)
        training_model   = model
        prediction_model = model

    # compile model
    training_model.compile(
        loss={
            'regression'    : losses.smooth_l1(),
            'classification': losses.focal()
        },
        optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001)
    )

    return model, training_model, prediction_model


def create_callbacks(model, training_model, prediction_model):
    callbacks = []

    # save the prediction model
    checkpoint = keras.callbacks.ModelCheckpoint(
        os.path.join(boring_snapshots_dir, 'boring_resnet50_{epoch:02d}.h5'),
        verbose=1
    )
    checkpoint = RedirectModel(checkpoint, prediction_model)
    callbacks.append(checkpoint)

    lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=2, verbose=1,
                                                     mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
    callbacks.append(lr_scheduler)

    return callbacks


def check_args(parsed_args):
    """
    Function to check for inherent contradictions within parsed arguments.
    For example, batch_size < num_gpus
    Intended to raise errors prior to backend initialisation.

    :param parsed_args: parser.parse_args()
    :return: parsed_args
    """

    if parsed_args.multi_gpu > 1 and parsed_args.batch_size < parsed_args.multi_gpu:
        raise ValueError(
            "Batch size ({}) must be equal to or higher than the number of GPUs ({})".format(parsed_args.batch_size,
                                                                                             parsed_args.multi_gpu))

    if parsed_args.multi_gpu > 1 and parsed_args.snapshot:
        raise ValueError(
            "Multi GPU training ({}) and resuming from snapshots ({}) is not supported.".format(parsed_args.multi_gpu,
                                                                                                parsed_args.snapshot))

    if parsed_args.multi_gpu > 1 and not parsed_args.multi_gpu_force:
        raise ValueError("Multi-GPU support is experimental, use at own risk! Run with --multi-gpu-force if you wish to continue.")

    if 'resnet' not in parsed_args.backbone:
        warnings.warn('Using experimental backbone {}. Only resnet50 has been properly tested.'.format(parsed_args.backbone))

    if 'resnet' in parsed_args.backbone:
        from keras_retinanet.models.resnet import validate_backbone
    elif 'mobilenet' in parsed_args.backbone:
        from keras_retinanet.models.mobilenet import validate_backbone
    elif 'vgg' in parsed_args.backbone:
        from keras_retinanet.models.vgg import validate_backbone
    elif 'densenet' in parsed_args.backbone:
        from keras_retinanet.models.densenet import validate_backbone
    else:
        raise NotImplementedError('Backbone \'{}\' not implemented.'.format(parsed_args.backbone))

    validate_backbone(parsed_args.backbone)

    return parsed_args

def parse_args(args):
    parser     = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True

    ts_parser = subparsers.add_parser('traffic_signs')
    ts_parser.add_argument('train_path', help='Path to folder containing files used for train. The rois.bin file should be there.')
    ts_parser.add_argument('val_path', help='Path to folder containing files used for validation (optional). The rois.bin file should be there.')

    def csv_list(string):
        return string.split(',')


    group = parser.add_mutually_exclusive_group()
    group.add_argument('--snapshot',          help='Resume training from a snapshot.')
    group.add_argument('--imagenet-weights',  help='Initialize the model with pretrained imagenet weights. This is the default behaviour.', action='store_const', const=True, default=True)
    group.add_argument('--weights',           help='Initialize the model with weights from a file.')
    group.add_argument('--no-weights',        help='Don\'t initialize the model with any weights.', dest='imagenet_weights', action='store_const', const=False)

    parser.add_argument('--backbone',        help='Backbone model used by retinanet.', default='resnet50', type=str)
    parser.add_argument('--batch-size',      help='Size of the batches.', default=1, type=int)
    parser.add_argument('--gpu',             help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--multi-gpu',       help='Number of GPUs to use for parallel processing.', type=int, default=2)
    parser.add_argument('--multi-gpu-force', help='Extra flag needed to enable (experimental) multi-gpu support.', action='store_true')
    parser.add_argument('--epochs',          help='Number of epochs to train.', type=int, default=50)
    parser.add_argument('--steps',           help='Number of steps per epoch.', type=int, default=10000)
    parser.add_argument('--snapshot-path',   help='Path to store snapshots of models during training (defaults to \'./snapshots\')', default='./snapshots')
    parser.add_argument('--tensorboard-dir', help='Log directory for Tensorboard output', default='./logs')
    parser.add_argument('--no-snapshots',    help='Disable saving snapshots.', dest='snapshots', action='store_false')
    parser.add_argument('--no-evaluation',   help='Disable per epoch evaluation.', dest='evaluation', action='store_false')
    parser.add_argument('--freeze-backbone', help='Freeze training of backbone layers.', action='store_true')
    parser.add_argument('--evaluate_score_threshold', help='Score thresholds to be used for all classes when evaluate.', default=0.5, type=float)

    return check_args(parser.parse_args(args))



def create_generator():
    # create image data generator objects
    train_image_data_generator = keras.preprocessing.image.ImageDataGenerator(
        horizontal_flip=True,
    )

    train_generator = CSVGenerator(
        boring_annotations_path,
        boring_classes_path,
#        train_image_data_generator,
        batch_size=batch_size
    )

    return train_generator

def main(args=None):
    # make sure keras is the minimum required version
    
#    if args is None:
#        args = sys.argv[1:]
#    
#    args = parse_args(args)
    
#    args.backbone = 'resnet'
    
    check_keras_version()

    keras.backend.tensorflow_backend.set_session(get_session())

    # create the generators
    train_generator = create_generator()
    
    weights=download_imagenet()
    
    
#    if 'resnet' in args.backbone:
#        from keras_retinanet.models.resnet import resnet_retinanet as retinanet, custom_objects, download_imagenet
#    elif 'mobilenet' in args.backbone:
#        from keras_retinanet.models.mobilenet import mobilenet_retinanet as retinanet, custom_objects, download_imagenet
#    elif 'vgg' in args.backbone:
#        from keras_retinanet.models.vgg import vgg_retinanet as retinanet, custom_objects, download_imagenet
#    elif 'densenet' in args.backbone:
#        from keras_retinanet.models.densenet import densenet_retinanet as retinanet, custom_objects, download_imagenet
#    else:
#        raise NotImplementedError('Backbone \'{}\' not implemented.'.format(args.backbone))

    # create the model
    print('Creating model, this may take a second...')
    print(train_generator.num_classes())
#    model, training_model, prediction_model = create_models(num_classes=train_generator.num_classes())

    model, training_model, prediction_model = create_models(
            backbone_retinanet=retinanet,
            backbone='resnet50',
            num_classes=train_generator.num_classes(),
            weights=weights,
            multi_gpu=1
#            freeze_backbone=args.freeze_backbone
        )
    # create the callbacks
    callbacks = create_callbacks(model, training_model, prediction_model)

    # start training
    training_model.fit_generator(
        generator = train_generator,
        steps_per_epoch = steps_per_training_epoch,
        epochs = training_epochs,
        verbose=1,
        callbacks=callbacks,
    )
    
    
    
    
    
if __name__ == '__main__':
    main()
