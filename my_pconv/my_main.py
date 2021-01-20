import os, sys
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]='0'
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from argparse import ArgumentParser

import tensorflow as tf
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.99)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# print('''+++ env info +++
# Python version : {},
# Tensorflow version : {},
# Keras version : {}
# '''.format(sys.version, tf.__version__, tf.keras.__version__))
# print(tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None))


# Change to root path
# if os.path.basename(os.getcwd()) != 'PConv-Keras-master':
#     os.chdir('..')

from libs.pconv_model import PConvUnet
from libs.util import ImageChunker
from libs.util import MaskGenerator

def parse_args():
    parser = ArgumentParser(description='Training script for PConv inpainting')

    parser.add_argument(
        '--image', 
        type=str, default='myDataset',
        help='Dataset name, e.g. \'imagenet\''
    )

    parser.add_argument(
        '--mask',
        type=str, default='./data/test_samples/',
        help='Where to output mask images'
    )

    parser.add_argument(
        '--savepath',
        type=str, default='./data/logs/',
        help='Where to output images'
    )
    
    parser.add_argument(
        '--weight',
        type=str, default='./data/logs/',
        help='Where to output weights'
    )

    parser.add_argument(
        '--gpu',
        type=bool, default=False
    )    

    return  parser.parse_args()


def plot_images_ts(images, s=5):
    _, axes = plt.subplots(1, len(images), figsize=(s*len(images), s))
    if len(images) == 1:
        axes = [axes]
    for img, ax in zip(images, axes):
        ax.imshow(img)
    plt.savefig(SAVE_PATH)
    plt.show()

def inpt_test_ts(model, img_path, mask_path, size=(512, 512), refer=True):

    im = Image.open(img_path).resize((size[0], size[1]))
    mask = Image.open(mask_path).resize((size[0], size[1]))

    im = tf.keras.preprocessing.image.img_to_array(im)/255
    mask = tf.keras.preprocessing.image.img_to_array(mask)/255
    mask = np.concatenate((mask,mask,mask), axis=-1)
    mask = 1-mask
    masked = deepcopy(im)
    masked[mask==0] = 1

    chunker = ImageChunker(512, 512, 3)
    chunked_images = chunker.dimension_preprocess(deepcopy(im))
    chunked_masks = chunker.dimension_preprocess(deepcopy(mask))
    chunked_masked = chunker.dimension_preprocess(deepcopy(masked))

#         start = time.time()
    pred_imgs = np.empty((0,512,512,3), float)
    for c_img_1, c_mask_1, c_ori_1 in zip(chunked_masked, chunked_masks, chunked_images):
        c_img = np.expand_dims(c_img_1, 0)
        c_mask = np.expand_dims(c_mask_1, 0)
        pred_img = model.predict([c_img, c_mask])

        pred_field = (1-c_mask)*pred_img
        real_field = (c_mask*c_img)

        recon_img = real_field + pred_field
        pred_imgs = np.append(pred_imgs, recon_img, axis=0)
#         print(time.time()-start)

    reconstructed_image = chunker.dimension_postprocess(pred_imgs, im)
    if refer:
        plot_images_ts([im, masked, reconstructed_image], s=10)
    else:
        plot_images_ts([im], s=10)
        
        
    
# Run script
if __name__ == '__main__':

#     Parse command-line arguments
    args = parse_args()
    
    SAMPLE_DIR = args.image
    MASK_DIR = args.mask
    SAVE_PATH = args.savepath
    WEIGHT_PATH = args.weight
    GPU = args.gpu

    ## GPU인지 CPU인지 확인
    if GPU:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.99)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        print(tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None))
    else:
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]=''
        print(tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None))


    # Load the model

    model = PConvUnet(vgg_weights=None, inference_only=True)
    model.load(WEIGHT_PATH, train_bn=False)
    
    IMG_SP = SAMPLE_DIR
    MSK_SP = MASK_DIR
    inpt_test_ts(model, IMG_SP, MSK_SP, size=(512,512))
    print('Finished!!!')