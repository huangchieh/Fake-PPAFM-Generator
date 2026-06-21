import torch, os
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
    #image_tensor = image_tensor.clamp(0, 1)
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


def save_figure_multi_format(fig, base_path):
    fig.savefig(f"{base_path}.png", dpi=300, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(f"{base_path}.svg", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(f"{base_path}.pdf", bbox_inches="tight", pad_inches=0.01)


def visualize_sample(input_slices, output_slices, sample_name, save_dir):
    selected_indices = list(range(0, len(input_slices), 2))
    if not selected_indices:
        print("No slices available for visualization.")
        return

    num_slices = len(selected_indices)
    fig, axes = plt.subplots(2, num_slices, figsize=(2 * num_slices, 4))

    if num_slices == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    global_min = min(np.min(input_slices), np.min(output_slices))
    global_max = max(np.max(input_slices), np.max(output_slices))

    for plot_col, slice_idx in enumerate(selected_indices):
        im_input = axes[0, plot_col].imshow(
            input_slices[slice_idx], cmap="gray", interpolation="nearest", vmin=global_min, vmax=global_max
        )
        axes[0, plot_col].set_title(f"Exp. {slice_idx}", y=0.96)
        axes[0, plot_col].axis("off")

        im_output = axes[1, plot_col].imshow(
            output_slices[slice_idx], cmap="gray", interpolation="nearest", vmin=global_min, vmax=global_max
        )
        axes[1, plot_col].set_title(f"Sim. style {slice_idx}", y=0.96)
        axes[1, plot_col].axis("off")

    fig.subplots_adjust(left=0.01, right=0.90, bottom=0.02, top=0.95, wspace=0.03, hspace=0.03)
    cax = fig.add_axes([0.912, 0.12, 0.012, 0.76])
    cbar = fig.colorbar(im_output, cax=cax)
    cbar.set_label(r"$\Delta f$ Hz")

    save_figure_multi_format(fig, os.path.join(save_dir, f"AFM_and_FakePPAFM_{sample_name}"))
    plt.show()


def load_array_from_file(file_path):
    loaded = np.load(file_path)
    if isinstance(loaded, np.ndarray):
        return loaded
    return loaded['data']


def compare_saved_slices(original_file, new_file, sample_name, save_dir):
    original_data = load_array_from_file(original_file)
    new_data = load_array_from_file(new_file)

    if original_data.shape != new_data.shape:
        print("Shape mismatch:", original_data.shape, new_data.shape)
        return

    selected_indices = list(range(0, original_data.shape[2], 2))
    if not selected_indices:
        print("No slices available for comparison.")
        return

    num_slices = len(selected_indices)
    fig, axes = plt.subplots(3, num_slices, figsize=(2 * num_slices, 6))
    if num_slices == 1:
        axes = np.array([[axes[0]], [axes[1]], [axes[2]]])

    # Use a shared intensity range for fair visual comparison.
    global_min = min(np.min(original_data), np.min(new_data))
    global_max = max(np.max(original_data), np.max(new_data))

    for plot_col, slice_idx in enumerate(selected_indices):
        orig_slice = original_data[:, :, slice_idx]
        new_slice = new_data[:, :, slice_idx]
        diff_slice = new_slice - orig_slice

        im_orig = axes[0, plot_col].imshow(orig_slice, cmap="gray", interpolation="nearest", vmin=global_min, vmax=global_max)
        axes[0, plot_col].set_title(f"Exp. {slice_idx}", y=0.96)
        axes[0, plot_col].axis("off")

        im_new = axes[1, plot_col].imshow(new_slice, cmap="gray", interpolation="nearest", vmin=global_min, vmax=global_max)
        axes[1, plot_col].set_title(f"Sim. style {slice_idx}", y=0.96)
        axes[1, plot_col].axis("off")

        vmax_diff = np.max(np.abs(diff_slice))
        im_diff = axes[2, plot_col].imshow(diff_slice, cmap="seismic", interpolation="nearest", vmin=-vmax_diff, vmax=vmax_diff)
        axes[2, plot_col].set_title(f"Diff {slice_idx}", y=0.96)
        axes[2, plot_col].axis("off")

    fig.subplots_adjust(left=0.01, right=0.90, bottom=0.02, top=0.95, wspace=0.03, hspace=0.05)
    cax_gray = fig.add_axes([0.912, 0.40, 0.012, 0.45])
    cbar_gray = fig.colorbar(im_new, cax=cax_gray)
    cbar_gray.set_label(r"$\Delta f$ Hz")

    cax_diff = fig.add_axes([0.912, 0.13, 0.012, 0.22])
    cbar_diff = fig.colorbar(im_diff, cax=cax_diff)
    cbar_diff.set_label(r"$\Delta f$ Hz")

    save_figure_multi_format(fig, os.path.join(save_dir, f"compare_original_vs_new_{sample_name}"))
    plt.show()

if __name__ == "__main__":
    # Model parameters
    #L1, L2 = 20, 1
    L1, L2 = 50, 0.5
    # Load the image as a grayscale NumPy array
    image_path = "image_input/exp_data/"  # Change this if needed
    samples = ['Ying_Jiang_1.npz', 'Ying_Jiang_2_1.npz', 'Ying_Jiang_2_2.npz', 'Ying_Jiang_3.npz', 'Ying_Jiang_5.npz', 'Ying_Jiang_6.npz']
    combo_dir_name = f"L1_{L1}_L2_{L2}"
    output_path = os.path.join("results", combo_dir_name)
    os.makedirs(output_path, exist_ok=True)
    for s in samples:
        sample = np.load(image_path + s)
        #print(sample['data'])
        min_f, max_f = np.min(sample['data']), np.max(sample['data'])
        max_abs_f = max(abs(min_f), abs(max_f))
        print("Min value: {}, Max value: {}".format(min_f, max_f))
        print(sample['lengthX'], sample['lengthY'])
		# Rescale the pixel values from [-max_abs_f, max_abs_f] to [-1, 1], which is the expected input range for the CycleGAN model
        rescaled_data = ((sample['data'] + max_abs_f) / (2 * max_abs_f)) * 2.0 - 1.0
        print(rescaled_data.shape)
        print(np.min(rescaled_data), np.max(rescaled_data))
        new_sample = np.zeros_like(sample['data'])
        for i in range(rescaled_data.shape[2]):
            slice_img = rescaled_data[:, :, i]
            # Show image slice
            #plt.imshow(slice_img, cmap="gray", interpolation="nearest")
            #plt.show()
            original_h, original_w = slice_img.shape
            slice_192 = np.array(Image.fromarray(slice_img).resize((192, 192), resample=Image.BICUBIC))
            sim_style_slice_192 = afm2ppafm(slice_192, hperameters=(L1, L2))
            # Resize translated output back to the original slice size.
            print('Style translated', np.min(sim_style_slice_192), np.max(sim_style_slice_192))
            sim_style_slice_original = np.array(
                Image.fromarray(sim_style_slice_192.astype(np.float32), mode="F").resize(
                    (original_w, original_h), resample=Image.BICUBIC
                )
            )

            new_sample[:, :, i] = sim_style_slice_original
        # Rescale back to the original value range
        min_f_new, max_f_new = np.min(new_sample), np.max(new_sample)
        max_abs_f_new = max(abs(min_f_new), abs(max_f_new))
		# Map the pixel values of new_sample from [-max_abs_f_new, max_abs_f_new] back to the original range [-max_abs_f, max_abs_f]
        rescaled_new_sample = ((new_sample + max_abs_f_new) / (2 * max_abs_f_new)) * (2 * max_abs_f) - max_abs_f
        print("Min value before style transfer: {}, Max value before style transfer: {}".format(min_f, max_f))
        print("Min value after style transfer: {}, Max value after style transfer: {}".format(np.min(rescaled_new_sample), np.max(rescaled_new_sample)))
		# Save the new sample in .npz format
        output_file = output_path + s
        np.savez(output_file, data=rescaled_new_sample, lengthX=sample['lengthX'], lengthY=sample['lengthY'])

		# Save 'data', 'lengthX', and 'lengthY' into a new .npz file
        sample_name = s.rsplit('.', 1)[0]
        # First comparison should use original-value scale, not normalized working tensors.
        input_slices_for_plot = [sample['data'][:, :, i] for i in range(sample['data'].shape[2])]
        output_slices_for_plot = [rescaled_new_sample[:, :, i] for i in range(rescaled_new_sample.shape[2])]
        visualize_sample(input_slices_for_plot, output_slices_for_plot, sample_name, output_path)

        original_file = image_path + s
        compare_saved_slices(original_file, output_file, sample_name, output_path)

