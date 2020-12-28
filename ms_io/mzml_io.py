import logging
import os
from typing import Dict, IO, Iterable, Union

import pyteomics.mzml
import spectrum_utils.spectrum as sus
from lxml.etree import LxmlError


logger = logging.getLogger('spectrum_clustering')


def get_spectra(source: Union[IO, str]) -> Iterable[sus.MsmsSpectrum]:
    """
    Get the MS/MS spectra from the given mzML file.

    Parameters
    ----------
    source : Union[IO, str]
        The mzML source (file name or open file object) from which the spectra
        are read.

    Returns
    -------
    Iterable[MsmsSpectrum]
        An iterator over the spectra in the given file.
    """
    with pyteomics.mzml.MzML(source) as f_in:
        filename = os.path.basename(f_in.name)
        try:
            for spectrum_dict in f_in:
                if int(spectrum_dict.get('ms level', -1)) == 2:
                    # USI-inspired spectrum identifier.
                    scan_nr = spectrum_dict['id'][
                        spectrum_dict['id'].find('scan=') + 5:]
                    spectrum_dict['id'] = f'{filename}:scan:{scan_nr}'
                    try:
                        yield _parse_spectrum(spectrum_dict)
                    except ValueError:
                        pass
        except LxmlError as e:
            logger.warning('Failed to read file %s: %s', source, e)


def _parse_spectrum(spectrum_dict: Dict) -> sus.MsmsSpectrum:
    """
    Parse the Pyteomics spectrum dict.

    Parameters
    ----------
    spectrum_dict : Dict
        The Pyteomics spectrum dict to be parsed.

    Returns
    -------
    MsmsSpectrum
        The parsed spectrum.
    """
    spectrum_id = spectrum_dict['id']
    mz_array = spectrum_dict['m/z array']
    intensity_array = spectrum_dict['intensity array']
    retention_time = spectrum_dict['scanList']['scan'][0]['scan start time']

    precursor = spectrum_dict['precursorList']['precursor'][0]
    precursor_ion = precursor['selectedIonList']['selectedIon'][0]
    precursor_mz = precursor_ion['selected ion m/z']
    if 'charge state' in precursor_ion:
        precursor_charge = int(precursor_ion['charge state'])
    elif 'possible charge state' in precursor_ion:
        precursor_charge = int(precursor_ion['possible charge state'])
    else:
        raise ValueError('Unknown precursor charge')

    return sus.MsmsSpectrum(spectrum_id, precursor_mz, precursor_charge,
                            mz_array, intensity_array, None, retention_time)
