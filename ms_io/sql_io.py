from typing import Dict, Iterable

import spectrum_utils.spectrum as sus
import numpy as np
import sqlite3
import struct
import re

def get_spectra(source: str) -> Iterable[sus.MsmsSpectrum]:
    """
    Get the MS/MS spectra from the given SQLite file.

    Parameters
    ----------
    source : str
        The SQLite source (filename) from which the spectra are read.

    Returns
    -------
    Iterator[MsmsSpectrum]
        An iterator over the spectra in the given file.
    """
    conn = sqlite3.connect(source)
    c = conn.cursor()

    res = c.execute(
        f"SELECT _pkey, precursorMz, precursorCharge, mz, intensity, rtime FROM msdata ORDER BY _pkey"
    )

    # Use res as an iterator
    for row in res:
        try:
            spectrum = build_spectrum(row)
            yield spectrum
        except ValueError: # Ignore spectra badly formatted
            pass

    conn.close()


def get_one_spectrum(source: str, _pkey: int):
    """
    Get one spectrum from the given sqlite file

    Parameters
    ----------
    source : str
        The SQLite source (filename) from which the spectrum is read.
    _pkey : int
        The _pkey of the spectrum that must be retrieved.

    Returns
    -------
    MsmsSpectrum
        The requested spectrum.
    """
    conn = sqlite3.connect(source)
    c = conn.cursor()
    res = c.execute(f"SELECT _pkey, precursorMz, precursorCharge, mz, intensity, rtime FROM msdata WHERE _pkey = {_pkey}")
    return build_spectrum(res.fetchone())


def build_spectrum(row):
    """
    Convert the SQL row to the MsmsSpectrum format

    Parameters
    ----------
    row : Tuple
        The result of the SQL query.

    Returns
    -------
    MsmsSpectrum
        The converted spectrum.
    """
    identifier, precursor_mz, precursor_charge, serializedMz, serializedIntensity, retention_time = row
    mz = unserialize(serializedMz)
    intensity = unserialize(serializedIntensity)

    if len(mz) != len(intensity):
        #print("Skip spectra ", identifier, " because len(mz) != len(intensity)")
        raise ValueError("Problem: length(mz) != length(intensity)")

    # TODO : test
    precursor_mz = np.around(precursor_mz, 4)

    # print("precMz = {}, precCharge = {}".format(precursor_mz, precursor_charge))
    spectrum = sus.MsmsSpectrum(identifier, precursor_mz, precursor_charge,
                                mz, intensity, retention_time=retention_time)
    return spectrum


def unserialize(serialized):
    """
    Convert the serialized list of doubles as it's stored in the SQLite database
    to a Python list of doubles.

    Parameters
    ----------
    serialized : binary
        The binary data retrieved in the database.

    Returns
    -------
    List[double]
        The converted serialized data.
    """
    currInd = 31  # The first 30 bytes don't contain the doubles we need to extract
    inc = 8
    nBytes = len(serialized)
    doubles = []

    while currInd + inc <= nBytes:
        currBytes = serialized[currInd:currInd + inc]
        doubles.append(struct.unpack('>d', currBytes)[0])
        currInd = currInd + inc

    return doubles


def write_spectra(filename: str, spectra: Iterable[sus.MsmsSpectrum]) -> None:
    raise NotImplementedError("Not implemented for SQLite backend")


def _spectra_to_dicts(spectra: Iterable[sus.MsmsSpectrum]) -> Iterable[Dict]:
    raise NotImplementedError("Not implemented for SQLite backend")