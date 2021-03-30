import os
import sys
import pickle
import joblib
import math
import functools
from scipy import spatial
import numpy as np
import scipy.sparse as ss
from tqdm.notebook import tqdm

sys.path.append('..')
import config
from cluster import cluster, spectrum

msclustering_tolerance = 0.05
hdvectors_min_mz, hdvectors_max_mz = 101, 1500
hdvectors_fragment_mz_tolerance = 0.05


# The function to process the spectra (same as for falcon)
process_spectrum = functools.partial(
        spectrum.process_spectrum,
        min_peaks=config.min_peaks,
        min_mz_range=config.min_mz_range,
        mz_min=config.min_mz,
        mz_max=config.max_mz,
        remove_precursor_tolerance=config.remove_precursor_tolerance,
        min_intensity=config.min_intensity,
        max_peaks_used=config.max_peaks_used,
        scaling=config.scaling)

"""
    IO functions
"""

def extract_mz_split(filename):
    ch, mz, extension = filename.replace('.', '_').split('_')
    return int(mz)


# Return True if this is a pkl file and it contains spectra of the indicated charge (based on the filename)
def is_valid(filename, charge):
    ch_mz, extension = filename.split('.')
    if extension != "pkl":
        return False

    ch, mz = map(int, ch_mz.split('_'))
    if ch != charge:
        return False

    return True


# Assume that all the .pkl in "path" contain spectra. Adapted from the falcon source code.
def read_spectra(charge, path, limit=None):
    filenames = os.listdir(path)
    pkl_filenames = [fn for fn in filenames if is_valid(fn, charge)]
    mz_splits = [extract_mz_split(fn) for fn in pkl_filenames]
    mz_splits.sort()
    cnt = 0

    for mz_split in mz_splits:
        with open(os.path.join(path, f'{charge}_{mz_split}.pkl'), 'rb') as f_in:
            for spec in pickle.load(f_in):
                cnt = cnt + 1
                if limit is not None and cnt > limit:
                    return
                yield spec


"""
    Functions related to the precursor mz tolerance
"""

# It seems that the precursor_tol_mass is applied on the precursor_mz, not the mass
# sp1 is the "current spectrum", sp2 is a potential neighbor
def respect_constraint(sp1, sp2, precursor_tol_mass):
    prec_mz1, prec_mz2 = (sp1.precursor_mz, sp2.precursor_mz)
    return math.abs(prec_mz1-prec_mz2)/prec_mz2 * 10**6 < precursor_tol_mass


# The formula used in falcon is abs(curr_mz - other_mz)/other_mz * 10^6 < tol
def window_precursor_mz(mz, tol):
    # For the begin of the window, abs(curr_mz - other_mz) = curr_mz - other_mz
    start_w = mz*10**6 / (tol+10**6)

    # For the end of the window, abs(curr_mz - other_mz) = other_mz - cyrr_mz
    end_w = mz*10**6 / (10**6-tol)

    return start_w, end_w

# Use a sliding window to extract all the potential neighbors
def update_window(queue_sp, currw_sp, curr_mz, mass_tol, iter, curr_id, pbar = None):
    startw, endw = window_precursor_mz(curr_mz, mass_tol)
    currw_sp = [sp for sp in currw_sp if sp is not None and sp[1].precursor_mz >= startw]

    while (len(currw_sp) == 0) or (currw_sp[-1][1].precursor_mz < endw):
        spec = next(iter, None)
        if spec is None:
            # All the spectra have been read.
            if currw_sp[-1][1].precursor_mz < endw:
                currw_sp.append(None)

            break
        else:
            if pbar is not None:
                pbar.update()

            # Process the spectrum
            spec = process_spectrum(spec)

            if spec is not None:
                queue_sp.append( (curr_id, spec) )
                currw_sp.append( (curr_id, spec) )
                curr_id = curr_id + 1

    return queue_sp, currw_sp, curr_id


def exact_sparse_matrix(sp_path, charge, precursor_tol_mass, dist_func):
    charge_count = joblib.load(os.path.join(sp_path, 'info.joblib'))
    pbar = tqdm(total=charge_count[charge])

    # Use the sliding window
    iter_sp = read_spectra(charge, sp_path)
    curr_id = 0
    queue_sp, currw_sp, curr_id = \
        update_window([], [], -1, precursor_tol_mass, iter_sp, curr_id, pbar)

    n_spectra, n_neighbors, n_sp_diff_bucket = 0, 0, 0
    data, indices, indptr = [], [], [0]

    while len(queue_sp) > 0:
        # Pull the next element from the queue of spectra
        i, curr_sp = queue_sp[0]
        if curr_sp is None:  # All the spectra have been processed
            break
        n_spectra = n_spectra + 1

        queue_sp = queue_sp[1:] # Remove the pulled spectra from the queue (decrease memory requirements)
        curr_mz = curr_sp.precursor_mz
        queue_sp, currw_sp, curr_id = \
            update_window(queue_sp, currw_sp, curr_mz, precursor_tol_mass, iter_sp, curr_id, pbar)

        indptr.append(indptr[-1])
        # Compare the spectrum with all the spectra in the same precursor mz window
        for j, pot_neighbor in currw_sp[:-1]:
            n_neighbors = n_neighbors + 1
            if math.floor(curr_sp.precursor_mz) != math.floor(pot_neighbor.precursor_mz):
                n_sp_diff_bucket = n_sp_diff_bucket + 1
            d = dist_func( (curr_sp, pot_neighbor) )
            data.append(d)
            indices.append(j)
            indptr[-1] = indptr[-1] + 1

    sparse_mat = ss.csr_matrix( (data, indices, indptr), shape=(n_spectra, n_spectra) )
    return sparse_mat, n_sp_diff_bucket

"""
    Helpers for sparse matrices
"""
def ind_in_sparse(mat, ind):
    i, j = ind
    ind_ptr = mat.indptr[i:i+2]
    indices = mat.indices[ind_ptr[0]:ind_ptr[1]]
    if j not in indices:
        return False
    return True

def ss_generator(mat):
    i = 0
    for i in range(0, mat.shape[0]):
        ind_ptr = mat.indptr[i:i + 2]
        indices = mat.indices[ind_ptr[0]:ind_ptr[1]]
        for j in indices:
            yield (i, j)

    return

# Check how many entries are in mat1 but not in mat2
def indices_lost(matrices):
    mat1, mat2 = matrices
    assert mat1.shape == mat2.shape
    n_spectra = mat1.shape[0]
    ind_lost = []

    for i in range(0, n_spectra):  # For each row
        indptr = [mat.indptr[i:i+2] for mat in matrices]
        indices = [mat.indices[ind_ptr[0]:ind_ptr[1]] for mat, ind_ptr in zip(matrices, indptr)]

        for j in indices[0]:
            if j not in indices[1]:
                ind_lost.append( (i, j) )

    return ind_lost

# Count the number of entries larger than thr in each row (= number of potential neighbors)
def extract_n_neighbors(mat, thresholds):
    n_spectra = mat.shape[0]
    n_neighbors = np.zeros( (n_spectra, len(thresholds)) )

    for i in tqdm(range(0, n_spectra)):
        indptr = mat.indptr[i:i+2]
        indices = mat.indices[indptr[0]:indptr[1]]
        indices = indices[np.where(indices != i)]

        for l in range(0, len(thresholds)):
            if len(indices) != 0:
                mask = np.where(mat.data[indices] < thresholds[l])
                n_neighbors[i,l] = len(mask[0])

    return n_neighbors

# Mat is a sparse matrix
def extract_nondiag_values(mat):
    n_spectra = mat.shape[0]
    data = []
    for i in tqdm(range(0, n_spectra)):
        indptr = mat.indptr[i:i+2]
        indices = mat.indices[indptr[0]:indptr[1]]
        indices = indices[np.where(indices != i)]
        data = data + mat.data[indices].tolist()
    return data


"""
    Distance functions
"""

def hdvectors_distance(sps):
    min_mz = hdvectors_min_mz
    max_mz = hdvectors_max_mz
    fragment_mz_tolerance = hdvectors_fragment_mz_tolerance
    vecs = []
    for sp in sps:
        l = int((max_mz - min_mz) / fragment_mz_tolerance)
        vec = np.zeros(l)
        for mz, intensity in zip(sp.mz, sp.intensity):
            ind = int(np.floor((mz - min_mz) / fragment_mz_tolerance))
            vec[ind] = vec[ind] + intensity
        vecs.append(vec)
    return spatial.distance.cosine( vecs[0], vecs[1] )

def msclustering_distance(sps):
    n_peaks = round((sps[0].precursor_charge * sps[0].precursor_mz)/1000 * 15)
    mz_int1_int2 = []
    for i, sp in zip( [0,1], sps):
        mz = sp.mz.tolist()
        intensity = sp.intensity.tolist()
        mz_intensity = [(i, m) for m, i in zip(mz, intensity)]
        mz_intensity.sort(reverse=True)
        mz_intensity = mz_intensity[:n_peaks]

        for int, mz in mz_intensity:
            if i == 0:
                mz_int1_int2.append( (mz, int, 0) )
            else:
                mz_int1_int2.append( (mz, 0, int) )

    mz_int1_int2.sort()
    mz_int1_int2_merged = []
    i = 0
    while i < ( len(mz_int1_int2) - 1):
        (mz1, int11, int12) = mz_int1_int2[i]
        (mz2, int21, int22) = mz_int1_int2[i+1]
        if(abs(mz1 - mz2) < msclustering_tolerance):
            mz_int1_int2_merged.append( (mz1+mz2/2, int11+int21, int12+int22) )
            i = i + 2
        else:
            mz_int1_int2_merged.append(mz_int1_int2[i])
            i = i+1

            if i == len(mz_int1_int2) - 2:
                mz_int1_int2_merged.append(mz_int1_int2[i+1])

    v1 = [int for _, int, _ in mz_int1_int2_merged]
    v2 = [int for _, _, int in mz_int1_int2_merged]

    return spatial.distance.cosine(v1, v2)