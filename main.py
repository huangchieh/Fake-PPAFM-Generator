'''
This is example to use trained model to do style translation
Usage: 
1. Login to a GPU node
2. Run this command:
python main.py --dataroot image_input --name HyperTest-resnet_6blocks-2-16-10-0.5 \
       --model test --netG resnet_6blocks  --ngf 16   --no_dropout --input_nc 1 \
       --output_nc 1 --num_test 100   --checkpoints_dir trained_models \
       --results_dir image_output
'''
import os
import sys
sys.path.append('./pytorch-CycleGAN-and-pix2pix') 
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options

    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling.
    opt.no_flip = True    # no flip
    opt.display_id = -1   # no visdom display

    # create a dataset given opt.dataset_mode and other options
    dataset = create_dataset(opt)  
    model = create_model(opt) # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers

    for i, data in enumerate(dataset):
        # Here A_paths is the path of input image, that is nothing to do with the type A. 
        # Just a path. 
        print('Process image... %s' % data['A_paths'])
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        
        real_B = model.real.cpu().numpy()
        real_B_size = real_B.shape

        # Print the size of real_A
        # print('Size of real_A:', real_A_size)

        # Print the range of real_A
        # print('Range of real_A:', np.min(real_A), np.max(real_A))
        fake_A = model.fake.cpu().numpy()
        image_fake_A = Image.fromarray(((fake_A[0, 0, :, :] \
                + 1) * 127.5).astype('uint8'), 'L')
        # Save style-transfered image in output folder
        image_fake_A.save('{}/{}'.format( opt.results_dir,\
                    data['A_paths'][0].split('/')[-1]))

