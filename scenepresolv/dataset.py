import os
from pathlib import Path
from datetime import datetime
import re
from functools import partial

from numba import jit
import numpy as np
from torch.utils.data import Dataset
from spectral import envi
from scipy.stats import gamma, norm, beta

from spectf.utils import envi_header
from isofit.core.common import resample_spectrum
from isofit.core.fileio import IO
from isofit.data import env


IRRFP = str(
    Path(env.examples) 
    / '20151026_SantaMonica/data/prism_optimized_irr.dat'
)


class ImageDataset(Dataset):
    def __init__(self, sp_paths: list, atm_paths: list,
                 obs_paths: list, nsamples: int, wl_grid: str,
                 dtype=np.float32, target_fun='gamma',
                 fixed_bins=np.linspace(0.2, 6, 30),
                 cache_cube=False, cache_root=None,
                 save_to_disk=False):
        """
        Options for target_fun are:
        mean_std: mean and standard deviation
        p95: 95% quantile interval
        IQR: IQR
        histogram: nbin histogram density
        """
        self.nsamples = nsamples
        self.index = [i for i in range(len(sp_paths))]
        self.dtype = dtype

        self.global_max = 0.1

        # Set up paths
        self.sp_paths = sp_paths
        self.atm_paths = atm_paths
        self.obs_paths = obs_paths

        self.irr = np.loadtxt(IRRFP)
        self.esd = IO.load_esd()

        rdn = envi.open(envi_header(self.sp_paths[0]))
        
        # Load wavelengths
        wl_grid = np.loadtxt(wl_grid) # in um
        self.wl = wl_grid[:, 0] * 1000 # in nm
        self.fwhm = wl_grid[:, 1] * 1000 # in nm

        self.irr = np.array(resample_spectrum(
            self.irr[:, 1], self.irr[:, 0],
            self.wl, self.fwhm
        ), dtype=float)

        # Constant indexes
        self.h2o_names = [
            'H2O (g cm-2)',
            'H2OSTR'
        ]
        self.solzen_names = [
            'To-sun zenith (0 to 90 degrees from zenith)',
        ]
        self.row_cols = []
        bad_idx = []
        for path_i, atm_path in enumerate(self.atm_paths):
            atm = envi.open(envi_header(atm_path))
            h2o_idx = [
                i for i, n in enumerate(atm.metadata['band names'])
                if n in self.h2o_names
            ]

            if 'Spacecraft Flag' in atm.metadata['band names']:
                spacecraft_idx = [
                    i for i in range(len(atm.metadata['band names']))
                    if atm.metadata['band names'][i] == 'Spacecraft Flag'
                ][0]
            else:
                spacecraft_idx = None

            atm_im = atm.open_memmap(interleave='bip')

            if np.std(atm_im[..., h2o_idx]) <= 2e-3:
                bad_idx.append(path_i)
                continue

            row, col, iters = self.sample_rowcol(
                atm_im, h2o_idx, spacecraft_idx
            )

            if iters < 10:
                self.row_cols.append([row, col])
            else:
                bad_idx.append(path_i)

        remove = lambda val_list, bad_idx: [
            v for i, v in enumerate(val_list)
            if i not in bad_idx
        ]
        self.index = remove(self.index, bad_idx)
        self.sp_paths = remove(self.sp_paths, bad_idx)
        self.atm_paths = remove(self.atm_paths, bad_idx)
        self.obs_paths = remove(self.obs_paths, bad_idx)

        if target_fun == 'mean_std':
            self.target_fun = self.calc_mean_std()
            self.target_dim = 2
        if target_fun == 'gamma':
            self.target_fun = self.calc_gamma()
            self.target_dim = 2
        if target_fun == 'histogram':
            self.target_fun = self.calc_histogram(fixed_bins)
            self.target_dim = len(fixed_bins)
        if target_fun == 'IQR':
            self.target_fun = self.calc_qrange(0.25, 0.75)
            self.target_dim = 2
        if target_fun == 'p90':
            self.target_fun = self.calc_qrange(0.1, 0.9)
            self.target_dim = 2
        if target_fun == 'p95':
            self.target_fun = self.calc_qrange(0.05, 0.95)
            self.target_dim = 2
        if target_fun == 'p99':
            self.target_fun = self.calc_qrange(0.01, 0.99)
            self.target_dim = 2

        self.get = self.get_with_calculations
        if cache_cube:
            self.get = self.get_cached
            if save_to_disk:
                toa_cube, atm_cube = self.build_cube(
                    cache_root,
                    True
                )
                self.toa_cube = toa_cube
                self.atm_cube = atm_cube
            else:
                toa_cube, atm_cube = self.build_cube(cache_root, False)
                self.toa_cube = toa_cube
                self.atm_cube = atm_cube

    def build_cube(self, cache_root, save_to_disk=True, units='TOA'):
        if cache_root and save_to_disk:
            cache_root = Path(cache_root)
            toa_name = f"{cache_root.stem}_toa.npy"
            toa_cache_path = cache_root.parent / toa_name

            atm_name = f"{cache_root.stem}_atm.npy"
            atm_cache_path = cache_root.parent / atm_name

            if toa_cache_path.is_file() and atm_cache_path.is_file():
                return (
                    np.load(toa_cache_path, mmap_mode='r'),
                    np.load(atm_cache_path, mmap_mode='r')
                )

        toa_cube = np.zeros((
            len(self.atm_paths),
            self.nsamples,
            len(self.wl)
        ), dtype=np.float32)
        atm_cube = np.zeros((
            len(self.atm_paths),
            self.target_dim
        ), dtype=np.float32)
        self.global_max = 0
        for idx in range(len(self.atm_paths)):
            dayofyear = get_dayofyear(self.sp_paths[idx])
            rdn = envi.open(envi_header(self.sp_paths[idx]))
            wl = np.array(rdn.metadata['wavelength']).astype(float)
            rdn = rdn.open_memmap(interleave='bip')
            atm = envi.open(envi_header(self.atm_paths[idx]))
            atm_idx = [
                i for i in range(len(atm.metadata['band names']))
                if atm.metadata['band names'][i] in self.h2o_names
            ][0]

            atm = atm.open_memmap(interleave='bip')
            obs = envi.open(envi_header(self.obs_paths[idx]))
            obs_idx = [
                i for i in range(len(obs.metadata['band names']))
                if obs.metadata['band names'][i] in self.solzen_names
            ][0]
            obs = obs.open_memmap(interleave='bip')

            row, col = self.row_cols[idx]
            rdn_sample = resample_spectrum(
                rdn[row, col, :].copy(),
                wl,
                self.wl, self.fwhm
            )

            if units == 'Radiance':
                toa_cube[idx, ...] = rdn_sample
            elif units == 'TOA':
                toa_cube[idx, ...] = rdn_to_toa(
                    rdn_sample,
                    obs[row, col, 4],
                    self.irr,
                    self.esd[int(dayofyear) - 1, 1],
                )
            else:
                raise ValueError("Unit mode not valid")

            atm_cube[idx, ...] = self.target_fun(atm[..., atm_idx])

            # get_maximum
            cube_max = np.max(toa_cube)
            if cube_max > self.global_max:
                self.global_max = cube_max

        toa_cube = toa_cube / self.global_max

        # Save cubes
        if save_to_disk:
            cache_root = Path(cache_root)
            os.makedirs(cache_root.parent, exist_ok=True)

            toa_name = f"{cache_root.stem}_toa.npy"
            np.save(cache_root.parent / toa_name, toa_cube)

            atm_name = f"{cache_root.stem}_atm.npy"
            np.save(cache_root.parent / atm_name, atm_cube)

            return (
                np.load(cache_root.parent / toa_name, mmap_mode='r'),
                np.load(cache_root.parent / atm_name, mmap_mode='r')
            )
        else:
            return toa_cube, atm_cube


    def sample_rowcol(self, atm_im, h2o_idx, spacecraft_idx=None, iter_limit=10):
        iters = 0
        row = np.random.randint(0, atm_im.shape[0], self.nsamples)
        col = np.random.randint(0, atm_im.shape[1], self.nsamples)
        rowcol = np.array([row, col]).T
        check = True
        while check:
            n = len(rowcol) - len(np.unique(rowcol, axis=0))
            row_add = np.random.randint(0, atm_im.shape[0], n)
            col_add = np.random.randint(0, atm_im.shape[1], n)
            rowcol = np.vstack([
                np.unique(rowcol, axis=0),
                np.array([row_add, col_add]).T
            ])
            row, col = rowcol[:, 0], rowcol[:, 1]

            row, col = replace_bad_rowcol(
                row,
                col,
                atm_im[..., h2o_idx],
                lambda x: ((x == -9999.) | np.isnan(x)) | np.isinf(x)
            )
            if spacecraft_idx:
                row, col = replace_bad_rowcol(
                    row,
                    col,
                    atm_im[..., spacecraft_idx],
                    lambda x: x == 1
                )
            rowcol = np.array([row, col]).T
            iters += 1

            if len(np.unique(rowcol, axis=0)) != len(rowcol):
                check = False

            if iters < iter_limit:
                break

        return row, col, iters

    @staticmethod
    def calc_histogram(fixed_bins):
        _hist = partial(np.histogram, bins=fixed_bins, density=True)
        return lambda data: _hist(data)[0]

    @staticmethod
    def calc_mean_std():
        return lambda data: np.array([
            np.nanmean(data),
            np.nanstd(data)
        ])

    @staticmethod
    def calc_gamma():
        def fit(data):
            data = data.flatten()
            mask = ~np.isnan(data) & ~np.isinf(data)
            data = data[mask]
            data = data[data > 0]

            shape, loc, scale = gamma.fit(data, floc=0)
            return np.array([shape, 1 / scale])

        return lambda data: fit(data)

    @staticmethod
    def calc_qrange(low, high):
        return partial(np.quantile, q=[low, high])

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        return self.get(idx)

    def get_with_calculations(self, idx):
        dayofyear = get_dayofyear(self.sp_paths[idx])
        rdn = envi.open(envi_header(self.sp_paths[idx]))

        wl = np.array(rdn.metadata['wavelength']).astype(float)
        fwhm = np.array(rdn.metadata['fwhm'], dtype=float)

        rdn = rdn.open_memmap(interleave='bip')
        atm = envi.open(envi_header(self.atm_paths[idx]))

        atm_idx = [
            i for i in range(len(atm.metadata['band names']))
            if atm.metadata['band names'][i] in self.h2o_names
        ][0]
        spacecraft_idx = [
            i for i in range(len(atm.metadata['band names']))
            if atm.metadata['band names'][i] == 'Spacecraft Flag'
        ][0]

        atm = atm.open_memmap(interleave='bip')
        obs = envi.open(envi_header(self.obs_paths[idx])).open_memmap(
            interleave='bip')

        # row, col = self.row_cols[idx]
        row, col, iters = self.sample_rowcol(
            atm, h2o_idx, spacecraft_idx
        )
        rdn_sample = resample_spectrum(
            rdn[row, col, :].copy(),
            wl,
            self.wl, self.fwhm
        )

        toa = rdn_to_toa(
            rdn_sample,
            obs[row, col, 4],
            self.irr,
            self.esd[int(dayofyear) - 1, 1],
        )

        toa = toa / self.global_max

        # Sample atm
        atm_sample = self.target_fun(atm[..., atm_idx])

        return {
            "toa": toa.astype(self.dtype),
            "atmosphere": atm_sample.astype(self.dtype),
        }

    def get_cached(self, idx):
        return {
            "toa": self.toa_cube[idx],
            "atmosphere": self.atm_cube[idx],
        }


@jit
def rdn_to_toa(rdn_sample, solzen_sample, irr, irr_factors):
    return (
        (np.pi / np.cos(np.deg2rad(solzen_sample)))[:, None]
        * (rdn_sample / (irr[None, :] / (irr_factors ** 2)))
    )


def get_dayofyear(path):
    pattern = r'(\d{8})t(\d{6})'
    match = re.search(pattern, path)
    return datetime.strptime(match[1], '%Y%m%d').timetuple().tm_yday


def replace_bad_rowcol(row, col, ar, criteria):
    while np.any(criteria(ar[row, col])):
        idx = np.where(criteria(ar[row, col]))[0]
        row_nan = np.random.randint(0, ar.shape[0], len(idx))
        col_nan = np.random.randint(0, ar.shape[1], len(idx))
        row[idx] = row_nan
        col[idx] = col_nan

    return row, col
