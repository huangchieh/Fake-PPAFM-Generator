import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# CycleGAN
import sys
sys.path.append('./pytorch-CycleGAN-and-pix2pix')
from models import create_model
from options.test_options_api import TestOptions

def afm2ppafm(image_array, hperameters=(10, 0.5)):
    # Convert to a PyTorch tensor
    image_tensor = torch.tensor(image_array, dtype=torch.float32)
    # Add batch and channel dimensions: (batch_size, channels, height, width)
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)
    # Ensure values are between 0 and 1
    image_tensor = image_tensor.clamp(0, 1)
    # Create data dict matching the expected format
    data = {'A': image_tensor, 'A_paths': ['']}  # Update path if needed
    
    opt = TestOptions().parse()  # get test options
    opt.name = "HyperTest-resnet_6blocks-2-16-{}-{}".format(hperameters[0], hperameters[1])
    opt.model = "test"
    opt.netG = "resnet_6blocks"
    opt.ngf = 16
    opt.no_dropout = True
    opt.input_nc = 1
    opt.output_nc = 1
    opt.checkpoints_dir = "trained_models"
    opt.load_size = 192
    opt.crop_size = 192
    
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling.
    opt.no_flip = True    # no flip
    opt.display_id = -1   # no visdom display
    
    model = create_model(opt) # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    
    model.set_input(data)  # unpack data from data loader
    model.test()           # run inference
    
    real_B = model.real.cpu().numpy()
    real_B_size = real_B.shape
    # The output 
    fake_A = model.fake.cpu().numpy().squeeze()  # Output image 
    return fake_A    

def visualize(real_B, fake_A):
    # Create a figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    # Plot the input image
    axes[0].imshow(real_B, cmap="gray", interpolation="nearest")
    axes[0].set_title("Input: AFM Image")
    axes[0].axis("off")  # Hide axis
    # Plot the output image
    axes[1].imshow(fake_A, cmap="gray", interpolation="nearest")
    axes[1].set_title("Output: Fake PPAFM Image")
    axes[1].axis("off")  # Hide axis
    # Adjust layout and show the plot
    plt.tight_layout()
    plt.savefig("AFM_and_FakePPAFM.png")
    plt.show()

if __name__ == "__main__":
    # Load the image as a grayscale NumPy array
    image_path = "image_input/testB/H2O Au111-20210107-059.sxm.png"  # Change this if needed
    image = Image.open(image_path).convert("L")  
    image_array = np.array(image)

    # Rescale pixel values to the range [-1, 1]: This range is expected by the CycleGAN model
    real_B = (image_array / 255.0) * 2.0 - 1.0
    
    # Style Transfer
    fake_A = afm2ppafm(real_B, hperameters=(50, 0.5)) # hperameters =  (Lambda1, Lambda2) 
    
    # Visualize the input and output images
    visualize(real_B, fake_A)
