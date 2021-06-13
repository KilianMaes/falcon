import os

import numpy as np


# Precursor charges and m/z's considered.
mz_interval = 1
charges, mzs = (2, 3, 4, 5), np.arange(50, 2501, mz_interval)
<<<<<<< HEAD
#charges, mzs = (2,), np.arange(50, 2501, mz_interval)
=======
>>>>>>> 220b666d2f358c95cb702600f44c8fcee106f4f0

# Spectrum preprocessing.
min_peaks = 5
min_mz_range = 250.
min_mz, max_mz = 101., 1500.
remove_precursor_tolerance = 0.5
min_intensity = 0.01
max_peaks_used = 50
scaling = None

# Spectrum to vector conversion.
fragment_mz_tolerance = 0.05
hash_len = 800

# Spectrum matching.
precursor_tol_mass, precursor_tol_mode = 20, 'ppm'

# NN index construction and querying.
n_neighbors, n_neighbors_ann = 64, 128
n_probe = 32
batch_size = 2**16

# DBSCAN clustering.
#eps = 0.1 # CHANGED
eps = 0.35
min_samples = 2

# Input/output.
overwrite = True # CHANGED
export_representatives = False
<<<<<<< HEAD
pxd = 'PXD000561'
io_buffer_read = 10000
io_limit = None
#peak_dir = os.path.abspath('/media/maesk/WD/MS/CCLE_Protein_01')
peak_dir = os.path.abspath('/media/maesk/WD/MS/PXD000561/mgf')
work_dir = os.path.abspath('/media/maesk/WD/falcon/PXD000561_computeTime')
nn_dir = os.path.join(work_dir,
                      'nn',
                      f'fragm_tol_{fragment_mz_tolerance}_hash_len_{hash_len}',
                      f'prec_tol_{precursor_tol_mass}')
=======
pxd = 'USI000000'
io_buffer_read = 10000
io_limit = 0.1*(10**6)
peak_dir = os.path.abspath('datasets')
work_dir = os.path.abspath('work_dir/test')
>>>>>>> 220b666d2f358c95cb702600f44c8fcee106f4f0
filenames = [os.path.join(peak_dir, filename)
             for filename in os.listdir(peak_dir)
             if filename.endswith('.db')]
