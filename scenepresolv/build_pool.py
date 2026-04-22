import os
from pathlib import Path
from datetime import datetime
import re
from joblib import Parallel, delayed

import click
import numpy as np
from spectral import envi
from tqdm import tqdm

from spectf.utils import envi_header
from isofit.core.common import resample_spectrum
from isofit.core.fileio import IO
from isofit.data import env

from scenepresolv.utils import file_to_list


IRRFP = str(
    Path(env.examples)
    / '20151026_SantaMonica/data/prism_optimized_irr.dat'
)


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


def sample_rowcol(
    atm_im, h2o_idx, nsamples,
    spacecraft_idx=None, iter_limit=20
):
    iters = 0
    row = np.random.randint(0, atm_im.shape[0], nsamples)
    col = np.random.randint(0, atm_im.shape[1], nsamples)
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


def process_single_scene(
    idx,
    sp_path,
    atm_path,
    obs_path,
    wl,
    fwhm,
    irr,
    esd,
    pool_size,
    quantiles=[0.25, 0.75]
):
    h2o_names = ['H2O (g cm-2)', 'H2OSTR']
    dayofyear = get_dayofyear(sp_path)

    # Open files
    rdn_obj = envi.open(envi_header(sp_path))
    rdn_wl = np.array(rdn_obj.metadata['wavelength']).astype(float)
    rdn_mm = rdn_obj.open_memmap(interleave='bip')

    atm_obj = envi.open(envi_header(atm_path))
    atm_mm = atm_obj.open_memmap(interleave='bip')

    obs_obj = envi.open(envi_header(obs_path))
    obs_mm = obs_obj.open_memmap(interleave='bip')

    atm_idx = [
        i for i, n in enumerate(atm_obj.metadata['band names'])
        if n in h2o_names
    ][0]

    spacecraft_idx = None
    if 'Spacecraft Flag' in atm_obj.metadata['band names']:
        spacecraft_idx = [
            i for i, n in enumerate(atm_obj.metadata['band names'])
            if n == 'Spacecraft Flag'
        ][0]

    # Sample row/cols
    row, col, iters = sample_rowcol(atm_mm, atm_idx, pool_size, spacecraft_idx)

    if iters >= 20:
        return None

    # Resample and TOA math
    rdn_sample = resample_spectrum(
        rdn_mm[row, col, :].copy(),
        rdn_wl,
        wl,
        fwhm
    )

    toa_data = rdn_to_toa(
        rdn_sample,
        obs_mm[row, col, 4],
        irr,
        esd[int(dayofyear) - 1, 1]
    )

    atm_data = np.quantile(atm_mm[..., atm_idx], q=quantiles)

    return idx, toa_data, atm_data


@click.command()
@click.argument('sp_paths')
@click.argument('atm_paths')
@click.argument('obs_paths')
@click.argument('wl_path')
@click.argument('cache_root')
@click.option('--pool_size', default=7500)
@click.option('--quantiles', '-q', multiple=True, default=[0.05, 0.95])
@click.option('--n_jobs', default=-1)
def build_cube(
    sp_paths,
    atm_paths,
    obs_paths,
    wl_path,
    cache_root,
    pool_size=7500,
    quantiles=[0.25, 0.75],
    n_jobs=-1
):
    sp_paths = file_to_list(sp_paths)
    atm_paths = file_to_list(atm_paths)
    obs_paths = file_to_list(obs_paths)

    wl_grid = np.loadtxt(wl_path)
    wl, fwhm = wl_grid[:, 0] * 1000, wl_grid[:, 1] * 1000
    irr = np.array(
        resample_spectrum(
            np.loadtxt(IRRFP)[:, 1],
            np.loadtxt(IRRFP)[:, 0],
            wl,
            fwhm
        )
    )
    esd = IO.load_esd()

    toa_cube = np.zeros((
        len(atm_paths),
        pool_size,
        len(wl)
    ), dtype=np.float32)

    atm_cube = np.zeros((
        len(atm_paths),
        len(quantiles)
    ), dtype=np.float32)

    results = Parallel(n_jobs=n_jobs)(
        delayed(process_single_scene)(
            i, sp_paths[i], atm_paths[i], obs_paths[i],
            wl, fwhm, irr, esd, pool_size, quantiles
        ) for i in tqdm(range(len(atm_paths)))
    )

    global_norm = 0
    valid_count = 0
    for res in results:
        if res is not None:
            idx, toa_data, atm_data = res
            toa_cube[idx] = toa_data
            atm_cube[idx] = atm_data

            local_max = np.max(toa_data)
            if local_max > global_norm:
                global_norm = local_max
            valid_count += 1

    toa_cube /= global_norm

    cache_root = Path(cache_root)
    os.makedirs(cache_root.parent, exist_ok=True)

    toa_name = f"{cache_root.stem}_toa.npy"
    np.save(cache_root.parent / toa_name, toa_cube)

    atm_name = f"{cache_root.stem}_atm.npy"
    np.save(cache_root.parent / atm_name, atm_cube)

    return 0


if __name__ == '__main__':
    build_cube()
