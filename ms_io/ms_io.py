import os
import pickle
from typing import Iterable, Iterator, List

from spectrum_utils.spectrum import MsmsSpectrum

from cluster import spectrum
from ms_io import mgf_io, mzml_io, mzxml_io, sql_io


def get_spectra(filename: str) -> Iterator[MsmsSpectrum]:
    """
    Get the MS/MS spectra from the given file.

    Supported file formats are MGF, MSP, mzML, mzXML.

    Parameters
    ----------
    filename : str
        The file name from which to read the spectra.

    Returns
    -------
    Iterator[MsmsSpectrum]
        An iterator over the spectra in the given file.
    """
    if not os.path.isfile(filename):
        raise ValueError(f'Non-existing peak file {filename}')

    _, ext = os.path.splitext(filename.lower())
    if ext == '.mgf':
        spectrum_io = mgf_io
    elif ext == '.mzml':
        spectrum_io = mzml_io
    elif ext == '.mzxml':
        spectrum_io = mzxml_io
    elif ext == '.db':
        spectrum_io = sql_io
    else:
        raise ValueError(f'Unknown spectrum file type with extension "{ext}"')

    for spec in spectrum_io.get_spectra(filename):
        spec.is_processed = False
        yield spec


def get_one_spectrum(filename: str, id: int) -> MsmsSpectrum:
    """
    Extract one spectrum from the indicated file.

    Parameters
    ----------
    filename : str
        The filename from which the spectrum has to be extracted.
    id : int
        The identifier of the spectrum.

    Returns
    -------
    MsmsSpectrum
        The requested spectrum.
    """
    _, ext = os.path.splitext(filename.lower())
    if ext == '.mgf':
        spectrum_io = mgf_io
    elif ext == '.db':
        spectrum_io = sql_io
    else:
        raise ValueError(f'Unknown spectrum file type with extension "{ext}"')

    spec = spectrum_io.get_one_spectrum(filename, id)
    spec.is_processed = False
    return spec

# TODO rename in "get_spectra_from_pkl" as it can get several spectra
def get_one_spectrum_from_pkl(dir, precursor_charge, precursor_mz, identifiers):
    """
    Works for all the input file formats because it reads the spectra directly in the .pkl files
    created during the clustering.

    Parameters
    ----------
    dir : str
        The directory containing the .pkl files.
    precursor_charge : int
        The charge of the spectra to retrieve.
    precursor_mz : float
        The precursor mz of the spectra to retrieve.
    identifier : str
        The string that identifies the spectrum.

    Returns
    -------
    MsmsSpectrum
        The requested spectrum.
    """
    res = []

    mz_split = int(precursor_mz) # Assume we use 1 mz wide buckets
    with open(os.path.join(dir, f'{precursor_charge}_{mz_split}.pkl'), 'rb') as f_in:
        for spec in pickle.load(f_in):
            if spec.identifier in identifiers:
                res.append(MsmsSpectrum(
                    spec.identifier, spec.precursor_mz, spec.precursor_charge,
                    spec.mz, spec.intensity))
                identifiers.remove(spec.identifier)
                if len(identifiers) == 0:
                    return res

    raise Exception('Not all spectra were found: ', str(identifiers))


def write_spectra(filename: str, spectra: Iterable[MsmsSpectrum]) -> None:
    """
    Write the given spectra to a peak file.

    Supported formats: MGF.

    Parameters
    ----------
    filename : str
        The file name where the spectra will be written.
    spectra : Iterable[MsmsSpectrum]
        The spectra to be written to the peak file.
    """
    ext = os.path.splitext(filename.lower())[1]
    if ext == '.mgf':
        spectrum_io = mgf_io
    else:
        raise ValueError('Unsupported peak file format (supported formats: '
                         'MGF)')

    spectrum_io.write_spectra(filename, spectra)
