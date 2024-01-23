import numpy as np
import torch
from scipy.stats import multivariate_normal


def box_radial_dist(shape):
    Z, Y, X = shape
    z, y, x = np.meshgrid(np.linspace(-Z // 2, Z // 2, Z),
                          np.linspace(-Y // 2, Y // 2, Y),
                          np.linspace(-X // 2, X // 2, X),
                          indexing="ij")
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    return r


def colored_noise_spectrum(spectral_amplitude, lres_noise_power=.8, hres_noise_power=.8):
    f = np.arange(len(spectral_amplitude)).astype(float) / float(len(spectral_amplitude))
    noise_spectral_amplitude_fraction = lres_noise_power + f * (hres_noise_power - lres_noise_power)
    noise_spectral_amplitude = noise_spectral_amplitude_fraction * spectral_amplitude
    return noise_spectral_amplitude, lres_noise_power, hres_noise_power


def sample_from_distribution(x, y):
    cdf = np.cumsum(y / np.sum(y))
    y_selected = np.random.uniform()  # Between zero and one
    x_selected = np.interp(y_selected, cdf, x)
    return x_selected

def get_spectral_indices(f):
    (z, y, x) = f.shape
    Z, Y, X = np.meshgrid(np.linspace(-z // 2, z // 2 - 1, z),
                          np.linspace(-y // 2, y // 2 - 1, y),
                          np.linspace(0, x - 1, x), indexing="ij")
    R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    R = np.round(np.fft.ifftshift(R, axes=(0, 1)))
    return R


def fourier_shell_avg(f, R=None):
    (fiz, fiy, fix) = f.shape

    if R is None:
        R = get_spectral_indices(f)

    R = np.round(R).astype(int)
    avg = np.zeros(fix)

    for r in range(fix):
        i = r == R
        avg[r] = np.sum(f[i])/np.sum(i)

    return avg


def spectral_amplitude(f, R=None):
    return fourier_shell_avg(np.abs(f), R)


def get_lowpass_filter(
    size,
    ires_filter: int = None,
    bfac: float = 0,
    filter_edge_width: int = 3,
    use_cosine_kernel: bool = True,
):
    ls = torch.linspace(-(size // 2), size // 2 - 1, size)
    lsx = torch.linspace(0, size // 2 + 1, size // 2 + 1)
    r = torch.stack(torch.meshgrid(ls, ls, lsx, indexing="ij"), 0)
    R = torch.sqrt(torch.sum(torch.square(r), 0))
    spectral_radius = torch.fft.ifftshift(R, dim=(0, 1))

    res = spectral_radius / (size / 2. + 1.)
    scale_spectrum = torch.zeros_like(res)

    if ires_filter is not None:
        filter_edge_halfwidth = filter_edge_width // 2

        edge_low  = np.clip((ires_filter - filter_edge_halfwidth) / size, 0, size / 2. + 1)
        edge_high = np.clip((ires_filter + filter_edge_halfwidth) / size, 0, size / 2. + 1)
        edge_width = edge_high - edge_low
        scale_spectrum[res < edge_low] = 1

        if use_cosine_kernel and edge_width > 0:
            scale_spectrum[(res >= edge_low) & (res <= edge_high)] = 0.5 + 0.5 * torch.cos(
                np.pi
                * (res[(res >= edge_low) & (res <= edge_high)] - edge_low)
                / edge_width
            )
    else:
        scale_spectrum += 1.

    if bfac != 0:
        scale_spectrum *= torch.exp(-bfac * res)

    return scale_spectrum


def res_to_index(resolution, voxel_size, box_size):
    """
    Calculates the Fourier shell index from resolution and voxel size and box size.

    i = index, b = box_size / 2, r = resolution, v = voxel_size
    i = 2 * b * v / r
    r -> inf    => i = 0
    r -> 2 * v  => i = b = box_size / 2
    """
    return int(round(box_size * voxel_size / resolution))


def get_fsc_torch(F1, F2, ang_pix=1):
    if F1.is_complex():
        pass
    else:
        F1 = torch.fft.fftn(F1, dim=[-3, -2, -1])
    if F2.is_complex():
        pass
    else:
        F2 = torch.fft.fftn(F2, dim=[-3, -2, -1])

    if F1.shape != F2.shape:
        print('The volumes have to be the same size')

    N = F1.shape[-1]
    ind = torch.linspace(-(N - 1) / 2, (N - 1) / 2 - 1, N)
    end_ind = torch.round(torch.tensor(N / 2)).long()
    X, Y, Z = torch.meshgrid(ind, ind, ind, indexing="ij")
    R = torch.fft.fftshift(torch.round(torch.pow(X ** 2 + Y ** 2 + Z ** 2, 0.5)).long())

    if len(F1.shape) == 3:
        num = torch.zeros(torch.max(R) + 1)
        den1 = torch.zeros(torch.max(R) + 1)
        den2 = torch.zeros(torch.max(R) + 1)
        num.scatter_add_(0, R.flatten(), torch.real(F1 * torch.conj(F2)).flatten())
        den = torch.pow(
            den1.scatter_add_(0, R.flatten(), torch.abs(F1.flatten(start_dim=-3)) ** 2) *
            den2.scatter_add_(0, R.flatten(), torch.abs(F2.flatten(start_dim=-3)) ** 2),
            0.5
        )
        FSC = num / den

    res = N * ang_pix / torch.arange(end_ind)
    FSC[0] = 1.

    return FSC[0:end_ind], res


def spectra_to_grid_torch(spectra, indices):
    if len(spectra.shape) == 1:  # Has no batch dimension
        grid = torch.gather(spectra, 0, indices.flatten().long())
    elif len(spectra.shape) == 2:  # Has batch dimension
        indices = indices.unsqueeze(0).expand([spectra.shape[0]] + list(indices.shape))
        grid = torch.gather(spectra.flatten(1), 1, indices.flatten(1).long())
    else:
        raise RuntimeError("Spectra must be at most two-dimensional (one batch dimension).")
    return grid.view(indices.shape)


def apply_filer_3d_torch(grid, filter):
    grid_ = grid.unsqueeze(0).unsqueeze(0)
    grid_ = torch.nn.functional.conv3d(grid_, filter, padding='same')
    return grid_.view(grid.shape)


def directional_blur_torch(grid, N, sx, sy, theta, axis_pair):
    ls = np.linspace(-1.5, 1.5, N)
    r = np.dstack(np.meshgrid(ls, ls, indexing="ij"))

    cov = np.array(
        [
            [sx * np.cos(theta), -sy * np.sin(theta)],
            [sx * np.sin(theta),  sy * np.cos(theta)]
        ]
    )
    cov = cov.dot(cov.T)
    kernel = multivariate_normal.pdf(r, mean=[0, 0], cov=cov).astype(np.float32)
    kernel /= np.sum(kernel)
    if axis_pair == 0:
        kernel = torch.from_numpy(kernel[None, None, :, :, None])
    elif axis_pair == 1:
        kernel = torch.from_numpy(kernel[None, None, :, None, :])
    elif axis_pair == 2:
        kernel = torch.from_numpy(kernel[None, None, None, :, :])

    return apply_filer_3d_torch(grid, kernel)



def pad_to_box(grid, box_size):
    shape = grid.shape
    shape_ = np.clip(shape, box_size + 1, None)

    si = np.array(shape) // 2
    so = np.array(shape_) // 2

    new_grid = torch.zeros(list(shape_)).to(grid.device)
    new_grid[
        so[0] - si[0]: so[0] + si[0],
        so[1] - si[1]: so[1] + si[1],
        so[2] - si[2]: so[2] + si[2]
    ] = grid
    return new_grid
