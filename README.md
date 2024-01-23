# Blush Regularization Training Code Samples

This repository contains key parts of the code for the training procedure of Blush regularization. 
In particular, it contains the data augmentation function and the loss function that where used in the paper.
To train a model, one needs to prepare the training dataset and write the training loop.

The model definition used in the paper is available [here](https://github.com/3dem/relion-blush).
The cryo-EM maps used as the training dataset can be downloaded from the EMDB.
A list of entry IDs along with the manually curated masks can be downloaded from Zenodo: [10.5281/zenodo.10553452](https://zenodo.org/records/10553452)


## Data Augmentation
The `augment_data` function is designed to augment 3D cryo-electron microscopy (cryo-EM) maps for noise2noise training. It manipulates two half-maps (`half1` and `half2`) and a mask (`mask`), applying various transformations and noise augmentations to enhance training data diversity and quality.

### Parameters

- `half1`: numpy array
  - The first half of the 3D cryo-EM map.
- `half2`: numpy array
  - The second half of the 3D cryo-EM map.
- `mask`: numpy array
  - A mask indicating regions of interest in the cryo-EM maps.
- `box_size`: int
  - The size of the output.
- `voxel_size`: float
  - The size of each voxel in the cryo-EM map.
- `do_smooth_solvent`: bool, default `True`
  - Determines whether to apply smoothing in the solvent regions.
- `augment_orientational_bias`: bool, default `False`
  - Enables augmentation for orientational bias in the data.
- `augment_bfactor`: bool, default `False`
  - Enables augmentation simulating the effect of B-factor sharpening.
- `augment_noise`: bool, default `False`
  - Enables noise augmentation in the frequency domain.
- `device`: torch.device, default `torch.device("cpu")`
  - The device on which to perform the computations (CPU or GPU).
- `verbose`: bool, default `False`
  - Enables verbose output for debugging or informational purposes.

### Functionality

1. **Pre-processing**:
   - Stacks and converts the input half-maps into a 3D grid.
   - Computes the spectral amplitude of the mean grid.
   - Normalizes and Fourier-transforms the grids and masks.

2. **Noise Augmentation** (if `augment_noise` is `True`):
   - Applies noise in the frequency domain based on an exponential distribution.

3. **Input Filtering** (if `augment_bfactor` is `True`):
   - Applies B-factor sharpening using a low-pass filter.

4. **Real-space Processing**:
   - Transforms the Fourier-transformed grids back to real-space.
   - Adjusts the mean and standard deviation.

5. **Real-space Masked Smoothing** (if `do_smooth_solvent` is `True`):
   - Applies low-pass filtering and smoothing in solvent regions.

6. **Augment Orientational Bias** (if `augment_orientational_bias` is `True`):
   - Applies directional blur to simulate orientational bias.

7. **Final Processing**:
   - Applies a radial mask to the grids.
   - Resizes the grids to the specified `box_size`.
   - Normalizes the final output grids.

### Returns
- `grid1`, `grid2`: torch.Tensor
  - The augmented half-maps after all transformations.
- `gt_grid1`, `gt_grid2`: torch.Tensor
  - The ground truth half-maps after all transformations.
- `mask`: torch.Tensor
  - The transformed mask.

### Usage Notes
- Ensure that `half1`, `half2`, and `mask` are numpy arrays of the same size.
- The function supports both CPU and GPU processing, controlled by the `device` parameter.
- Enable `verbose` mode for detailed log output, useful for troubleshooting and understanding the internal processing steps.

### Example
```python
# Sample usage
augmented_half1, augmented_half2, gt_half1, gt_half2, updated_mask = augment_data(
    half1, half2, mask,
    box_size=64, voxel_size=1.0,
    do_smooth_solvent=True,
    augment_orientational_bias=False,
    augment_bfactor=False,
    augment_noise=True,
    device=torch.device("cuda"),
    verbose=True
)
```

## Acknowledgements

The work in this repository is based on the research conducted in the paper 
"Data-driven regularisation lowers the size barrier of cryo-EM structure determination" by Kimanius, Dari, et al. 

For the complete details of the research and methods that inspired this work, please refer to the following publication:

Kimanius, D., et al. (2023) Data-driven regularisation lowers the size barrier of cryo-EM structure determination. bioRxiv. 
DOI: [10.1101/2023.10.23.563586](https://doi.org/10.1101/2023.10.23.563586).


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.