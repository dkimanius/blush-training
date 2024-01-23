from util import *

EPS = 1e-12


def augment_data(
        half1, half2, mask,
        box_size, voxel_size,
        do_smooth_solvent=True,
        augment_orientational_bias=False,
        augment_bfactor=False,
        augment_noise=False,
        device=torch.device("cpu"),
        verbose=False
):
    grids = np.stack([half1, half2], axis=0).astype(np.float32)
    grids_spectral_amplitude = spectral_amplitude(np.fft.rfftn(np.mean(grids, 0)))

    gt_grids = grids.copy()

    gt_grids = torch.from_numpy(gt_grids).float()

    gt_grid1 = gt_grids[0]
    gt_grid2 = gt_grids[1]

    mask = torch.from_numpy(mask).float()
    bool_mask = mask > 0.5

    gt_grid1_mean = torch.mean(gt_grid1[bool_mask])
    gt_grid2_mean = torch.mean(gt_grid2[bool_mask])

    gt_grid1_std = torch.std(gt_grid1[bool_mask]) + EPS
    gt_grid2_std = torch.std(gt_grid2[bool_mask]) + EPS

    gt_grid1 = (gt_grid1 - gt_grid1_mean) / gt_grid1_std
    gt_grid2 = (gt_grid2 - gt_grid2_mean) / gt_grid2_std

    gt_grid1_ft = torch.fft.rfftn(torch.fft.fftshift(gt_grid1))
    gt_grid2_ft = torch.fft.rfftn(torch.fft.fftshift(gt_grid2))

    shape = np.array(gt_grid1.shape)
    shape_ft = np.array(gt_grid1_ft.shape)

    grids = torch.from_numpy(grids).float()

    grid1 = grids[0]
    grid2 = grids[1]

    grid1_mean = torch.mean(grid1[bool_mask])
    grid2_mean = torch.mean(grid2[bool_mask])

    grid1_std = torch.std(grid1[bool_mask]) + EPS
    grid2_std = torch.std(grid2[bool_mask]) + EPS

    grid1 = (grid1 - grid1_mean) / grid1_std
    grid2 = (grid2 - grid2_mean) / grid2_std

    grid1_ft = torch.fft.rfftn(torch.fft.fftshift(grid1))
    grid2_ft = torch.fft.rfftn(torch.fft.fftshift(grid2))

    # Augment noise ---------------------------------------------------

    if augment_noise:
        # Exponential distribution for noise augmentation
        spectral_size = len(grids_spectral_amplitude)

        coord_ft = torch.stack(torch.meshgrid(
            torch.linspace(-(shape_ft[0] // 2), shape_ft[0] // 2 - 1, shape_ft[0]),
            torch.linspace(-(shape_ft[1] // 2), shape_ft[1] // 2 - 1, shape_ft[1]),
            torch.linspace(0, shape_ft[2] - 1, shape_ft[2]),
            indexing="ij"
        ), -1)

        coord_ft = torch.fft.ifftshift(coord_ft, dim=(0, 1))
        spectral_index = torch.sqrt(torch.sum(torch.square(coord_ft), -1)).round()
        spectral_index[spectral_index > spectral_size - 1] = spectral_size - 1
        noise_dist_x = np.arange(0., 2., .1)
        noise_dist_y = np.exp(noise_dist_x) - 0.8
        lres_noise_power = sample_from_distribution(noise_dist_x, noise_dist_y)
        hres_noise_power = sample_from_distribution(noise_dist_x, noise_dist_y)
        noise_spectral_amp, cache_lowres_noise_power, cache_highres_noise_power = \
            colored_noise_spectrum(grids_spectral_amplitude, lres_noise_power, hres_noise_power)

        if verbose:
            print(f"cache_lowres_noise_power={cache_lowres_noise_power}")
            print(f"cache_highres_noise_power={cache_highres_noise_power}")

        noise_power_grid = spectra_to_grid_torch(
            torch.from_numpy(noise_spectral_amp).float().to(device),
            spectral_index
        )
        grid1_ft += torch.normal(mean=0, std=noise_power_grid)
        grid2_ft += torch.normal(mean=0, std=noise_power_grid)

    # Apply input filtering -----------------------------------------

    if augment_bfactor:
        bfac = np.random.uniform(-5, 0)
        if verbose:
            print(f"B-factor sharpening: {bfac}")
        filter_grid = get_lowpass_filter(
            size=shape_ft[0],
            bfac=bfac
        )
        gt_grid1_ft *= filter_grid
        gt_grid2_ft *= filter_grid
        grid1_ft *= filter_grid
        grid2_ft *= filter_grid

    # Back to real-space -------------------------------------------------

    grid1 = torch.fft.fftshift(torch.fft.irfftn(grid1_ft))
    grid2 = torch.fft.fftshift(torch.fft.irfftn(grid2_ft))

    grid1 = grid1 * grid1_std + grid1_mean
    grid2 = grid2 * grid2_std + grid2_mean

    gt_grid1 = torch.fft.fftshift(torch.fft.irfftn(gt_grid1_ft))
    gt_grid2 = torch.fft.fftshift(torch.fft.irfftn(gt_grid2_ft))

    gt_grid1 = gt_grid1 * gt_grid1_std + gt_grid1_mean
    gt_grid2 = gt_grid2 * gt_grid2_std + gt_grid2_mean

    fsc, _ = get_fsc_torch(grid1, grid2)

    # Apply real-space masked smoothing ----------------------------------

    if do_smooth_solvent:
        index = res_to_index(15, voxel_size, shape[0])
        filter_grid = get_lowpass_filter(
            size=shape_ft[0],
            ires_filter=index,
            filter_edge_width=5
        )
        gt_grid1_lp_ft = gt_grid1_ft * filter_grid
        gt_grid2_lp_ft = gt_grid2_ft * filter_grid
        gt_grid1_smooth = torch.fft.fftshift(torch.fft.irfftn(gt_grid1_lp_ft))
        gt_grid2_smooth = torch.fft.fftshift(torch.fft.irfftn(gt_grid2_lp_ft))

        gt_grid1_smooth = gt_grid1_smooth * gt_grid1_std + gt_grid1_mean
        gt_grid2_smooth = gt_grid2_smooth * gt_grid2_std + gt_grid2_mean
        
        gt_grid1 = gt_grid1 * mask + gt_grid1_smooth * (1 - mask)
        gt_grid2 = gt_grid2 * mask + gt_grid2_smooth * (1 - mask)

    # Augment orientational bias in real-space ----------------------------

    if augment_orientational_bias:
        N = np.random.choice([7, 9, 11])
        sx = np.random.uniform(0.5, 1.)
        theta = np.random.uniform(-np.pi, np.pi)
        axis_pair = np.random.randint(3)
        if verbose:
            print(f"augment_orientational_bias: N={N}, sx={sx}")
        grid1 = directional_blur_torch(grid1, N, sx, 1e-1, theta, axis_pair)
        grid2 = directional_blur_torch(grid2, N, sx, 1e-1, theta, axis_pair)

    # Final processing ----------------------------------------------------

    radial_dist = torch.from_numpy(box_radial_dist(shape)).to(device)
    max_r = np.min(shape) / 2
    rmask = radial_dist < max_r

    grid1 *= rmask
    grid2 *= rmask
    gt_grid1 *= rmask
    gt_grid2 *= rmask

    if np.any(np.array(shape) <= box_size):
        if verbose:
            print(f"resizing box from {shape} to {box_size}")
        grid1 = pad_to_box(grid1, box_size)
        grid2 = pad_to_box(grid2, box_size)
        gt_grid1 = pad_to_box(gt_grid1, box_size)
        gt_grid2 = pad_to_box(gt_grid2, box_size)
        mask = pad_to_box(mask, box_size)

    gt_grids = torch.stack([gt_grid1, gt_grid2], 0)
    mean = torch.mean(gt_grids)
    std = torch.std(gt_grids)

    grid1 = (grid1 - mean) / std
    grid2 = (grid2 - mean) / std

    gt_grid1 = (gt_grid1 - mean) / std
    gt_grid2 = (gt_grid2 - mean) / std

    return grid1, grid2, gt_grid1, gt_grid2, rmask
