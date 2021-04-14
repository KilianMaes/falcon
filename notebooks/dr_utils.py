import os
import time
import random
import numpy as np
from sklearn import random_projection
from sklearn import manifold
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.utils import murmurhash3_32

from scipy.spatial.distance import cosine
import math

from tqdm.notebook import tqdm

# If pairs is not none, compute all the pairwise similarities between different spectra.
# Else, compute the similarities for the pairs of spectra in pairs
def compute_pairwise_cosine(vectors, pairs = None, title = None):
    if pairs is None:
        n, dim = np.shape(vectors)
        dist = np.zeros( (n*n - n,) )
        curr = 0

        pbar = tqdm(range(n))
        pbar.set_description(title)
        for i in pbar:
            for j in range(n):
                if j != i:
                    # cosine() refers to the cosine distance, not the cosine similarity
                    dist[curr] = cosine(vectors[i,:], vectors[j,:])
                    curr = curr + 1
    else:
        n = len(pairs)
        dist = np.zeros( (n,) )

        pbar = tqdm(range(n))
        pbar.set_description(title)
        for i_p in pbar:
            i, j = pairs[i_p]
            assert i != j
            dist[i_p] = cosine(vectors[i,:], vectors[j,:])

    return dist


def sp_to_vecHD(sps, min_mz, max_mz, fragment_mz_tolerance):
    nsp = len(sps)

    len_hd = math.ceil((max_mz - min_mz) / fragment_mz_tolerance)
    vec_hd = np.zeros( (len(sps), len_hd) )  # The size is (# of vec, vector size)
    for i in range(nsp):
        sp = sps[i]
        for mz, intensity in zip(sp.mz, sp.intensity):
            j = math.floor((mz - min_mz) / fragment_mz_tolerance)
            vec_hd[i, j] = vec_hd[i, j] + intensity
    return vec_hd


def reduction_falcon(vec_hd, n_components):
    nsp, vec_len = np.shape(vec_hd)
    hash_lookup = np.asarray([murmurhash3_32(i, 0, True) % n_components
                                  for i in range(vec_len)], np.uint32)

    vec_falcon = np.zeros( (nsp, n_components), np.float32)
    for i in tqdm(range(nsp)):
        for j in range(vec_len):
            # TODO norm ?
            hash_idx = hash_lookup[j]
            vec_falcon[i,hash_idx] += vec_hd[i,j]
    return vec_falcon


def reduction_gaussian(vec_hd, n_components):
    rng = np.random.RandomState(42)
    transformer = random_projection.GaussianRandomProjection(n_components=n_components, random_state=rng)
    vec_gauss = transformer.fit_transform(vec_hd)

    return vec_gauss


def reduction_sparse(vec_hd, n_components):
    rng = np.random.RandomState(42)
    transformer = random_projection.SparseRandomProjection(n_components=n_components, random_state=rng)
    vec_sparse = transformer.fit_transform(vec_hd)

    return vec_sparse


def reduction_tSNE(vec_hd, n_components):
    vec_tSNE = manifold.TSNE(n_components = n_components, method = 'exact').fit_transform(vec_hd)
    return vec_tSNE


def compare_reductions(vec_hd, n_components, funcs, methods, dirFig, pairs = None):
    nsp, vec_len = np.shape(vec_hd)

    fig, axs = plt.subplots(1, len(methods), figsize=(12, 4))
    title = 'Exhaustive' if pairs is None else 'Non exhaustive'
    title = title + ' comparison for %d spectra' % (nsp,)
    fig.suptitle(title)

    dist_hd = compute_pairwise_cosine(vec_hd, pairs, 'HD distance')
    dists = []
    for f, method in zip(funcs, methods):
        start_time = time.time()
        vec = f(vec_hd, n_components)
        print('Time needed for %s : %.2f s' % (method, time.time() - start_time))
        dists.append(compute_pairwise_cosine(vec, pairs=pairs, title=method))

    MSEs = []
    for i in range(len(methods)):
        mse = metrics.mean_squared_error(dist_hd, dists[i])
        MSEs.append(mse)
        axs[i].scatter(dist_hd, dists[i], s=0.1)
        axs[i].set(title="HD vectors vs %s\nMSE = %.4f" % (methods[i], mse),
                   xlabel="HD vectors cosine distance", ylabel="LD vectors cosine distance")

    plt.tight_layout()
    plt.savefig(dirFig, dpi=300)

    return MSEs


def generate_pairs(n_pairs, nsp, seed = 42):
    pairs = []
    while len(pairs) != n_pairs:
        i = random.randint(0, nsp-1)
        j = random.randint(0, nsp-1)
        if i != j:
            pairs.append( (i,j) )

    return pairs