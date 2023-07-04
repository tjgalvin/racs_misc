#!/usr/bin/env python

import logging 
from typing import NamedTuple, Optional, Tuple, List, Dict
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
from astropy.coordinates import Angle, SkyCoord
from astropy import units as u
from astropy.table import Table
from astropy.table.row import Row
from casacore.tables import table

logger = logging.getLogger()
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)

class Catalogue(NamedTuple):
    """A basic structure used to describe a known catalogue. 
    """
    file_name: str 
    freq: float # Hertz
    ra_col: str 
    dec_col: str 
    name_col: str 
    flux_col: str 
    maj_col: str 
    min_col: str 
    pa_col: str 


KNOWN_CATAS: Dict[str, Catalogue] = {
    'SUMSS': Catalogue(
        file_name="sumsscat.Mar-11-2008_CLH.fits", freq=843e6, ra_col='RA', dec_col='Dec', name_col='Mosaic', flux_col='Sp', maj_col='dMajAxis', min_col='dMinAxis', pa_col='dPA'
    ),
    'RACS': Catalogue(
        file_name="racs-low.fits", freq=887.56e6, ra_col="RA", dec_col="Dec", name_col="Gaussian_ID", flux_col="Total_flux_Gaussian", maj_col="DC_Maj", min_col="DC_Min", pa_col="DC_PA"
    ),
    'NVSS': Catalogue(
        file_name='NVSS_vizier.fits', freq=1400e6, ra_col="RAJ2000", dec_col="DEJ2000", name_col='NVSS', flux_col='S1_4', maj_col='MajAxis', min_col='MinAxis', pa_col='PA'
    )
}

class GaussianPB:
    """A simple PB model
    """
    
    def __init__(self, aperture: float = 12.0, frequency: float = 1.1e9) -> None:
        """Set up the PB Gaussian model

        Args:
            aperture (float, optional): The size of the dish, in meters. Defaults to 12.0.
            frequency (float, optional): The nominal observing frequency, in Herta. Defaults to 1.1e9.
        """
        
        # TODO: Correct the `evaluate` method to properly calculate the FWHM internally
        
        self.aperture = aperture
        self.expScaling = 4.*np.log(2.)
        self.frequency=frequency
        self.setXwidth(self.getFWHM())
        self.setYwidth(self.getFWHM())
        self.setAlpha(0.0)
        self.setXoff(0.0)
        self.setYoff(0.0)
        
    
    def getFWHM(self) -> float:
        """Calculate the size of the Gaussian at the nominal frequency using the
        provided aperture size.

        Returns:
            float: The FWHM in radians
        """
        sol = 299792458.0
        fwhm = sol / self.frequency / self.aperture
        return fwhm
    
    def evaluate(self,offset: float = 0.0, freq: float = 0.0) -> float:
        """Calculate the attenuation at a provuded offset.
        
        Args:
            offset (float, optional): Distance from the origin, in radians. Defaults to 0.0.
            freq (float, optional): Frequency to evaluate at, in Hertz. Defaults to 0.0.

        Returns:
            float: The estimated amount of attenuation
        """
        if (freq > 0):
            self.frequency=freq
            
        pb = np.exp(-offset * offset * self.expScaling / (self.getFWHM() * self.getFWHM()))
        return pb
    
    def setXwidth(self,xwidth):
        self.xwidth = xwidth
        
    def setYwidth(self,ywidth):
        self.ywidth = ywidth
        
    def setAlpha(self,Angle):
        self.Alpha = Angle
        
    def setXoff(self,xoff):
        self.xoff = xoff
        
    def setYoff(self,yoff):
        self.yoff = yoff
        
    def evaluateAtOffset(self,offsetPAngle=0, offsetDist=0, freq=0):
            
        # x-direction is assumed along the meridian in the direction of north celestial pole
        # the offsetPA angle is relative to the meridian
        # the Alpha angle is the rotation of the beam pattern relative to the meridian
        # Therefore the offset relative to the
                
        if (freq > 0):
            self.frequency=freq
            
        x_angle = offsetDist * np.cos(offsetPAngle - self.Alpha)
        y_angle = offsetDist * np.sin(offsetPAngle - self.Alpha)
                    
        x_pb = np.exp(-1.0 * self.expScaling * np.power((x_angle - self.xoff) / self.xwidth, 2.0))
        y_pb = np.exp(-1.0 * self.expScaling * np.power((y_angle - self.yoff) / self.ywidth, 2.0))

        return x_pb * y_pb

def dir_from_ms(ms_path: Path) -> SkyCoord:
    """Extract the pointing direction from a measurement set

    Args:
        ms_path (Path): Path to the measurement set to query

    Returns:
        SkyCoord: Pointing direction on the sky of the measurement set
    """
    tp = table(f"{str(ms_path)}/FIELD", readonly=True, ack=False)
    p_phase = tp.getcol('PHASE_DIR')
    tp.close()
    
    td = table(str(ms_path), readonly=True, ack=False)
    field = td.getcol("FIELD_ID", 0, 1)[0]
    
    return SkyCoord(
        Angle(p_phase[field][0][0], unit=u.rad), 
        Angle(p_phase[field][0][1], unit=u.rad)
    )

def freqs_from_ms(ms_path: Path) -> np.ndarray:
    """Extract the set of observing frequencies within the measurement set.

    Args:
        ms_path (Path): Path to the measurement set to query

    Returns:
        np.ndarray: Collection of channel frequencies. 
    """
    tf = table(f"{str(ms_path)}/SPECTRAL_WINDOW", ack=False)
    freqs = tf[0]["CHAN_FREQ"]
    tf.close()
    return freqs


def flux_nu(S1, alpha, nu1, nu2) -> float:
    """Scale the flux S1 measured at frequency nu1 to nu2 assuming
    a spectral index of alpha

    Args:
        S1 (float): The reference brightness
        alpha (float): Assumed spectral index
        nu1 (float): Reference frequency
        nu2 (float): Frequency to scale to

    Returns:
        float: Brights at nu2
    """
    return S1 * np.power(nu2 / nu1, alpha)


def get_known_catalogue(cata: str) -> Catalogue:
    """Get the parameters of a known catalogue

    Args:
        cata (str): The lookup name of the catalogue

    Returns:
        Catalogue: properties of known catalogue
    """
    assert cata.upper() in KNOWN_CATAS.keys(), f"'{cata}' not a known catalogue. Acceptable keys are: {KNOWN_CATAS.keys()}."

    cata_info = KNOWN_CATAS[cata.upper()]
    logger.info(f"Loading {cata}={cata_info.file_name}")

    return cata_info

def load_catalogue(catalogue_dir: Path, catalogue: Optional[str]=None, dec_point: Optional[float]=None) -> Tuple[Catalogue,Table]:
    """Load in a catalogue table given a name or measurement set declinattion. 

    Args:
        catalogue_dir (Path): Directory containing known catalogues
        catalogue (Optional[str], optional): Catalogue name to look up from known catalogues. Defaults to None.
        dec_point (Optional[float], optional): Pointing direction of the measurement set. Defaults to None.

    Raises:
        FileNotFoundError: Raised when a catalogue can not be resolved. 

    Returns:
        Tuple[Catalogue,Table]: The `Catalogue` information and `Table` of components loaded
    """
    assert catalogue is not None or dec_point is not None, "Either catalogue or dec_point have to be provided. "
    
    if catalogue:
        logger.info(f"Loading provided catalogue {catalogue=}")
        cata = get_known_catalogue(catalogue)
    
    else:
        logger.info(f"Automatically loading catalogue based on {dec_point=:.2f}")
        assert dec_point is not None, f"Invalid type of {dec_point=}"
        
        if dec_point < -75.:
            cata = get_known_catalogue('SUMSS')
        elif dec_point < 26.:
            cata = get_known_catalogue('RACS')
        else: 
            cata = get_known_catalogue('NVSS')
    
    cata_path = catalogue_dir / cata.file_name
    
    if not cata_path.exists():
        raise FileNotFoundError(f"Catalogue {cata_path} not found.")
    
    cata_tab = Table.read(cata_path)
    logger.info(f"Loaded table, found {len(cata_tab)} sources. ")
    
    return (cata, cata_tab)


def preprocess_catalogue(
    cata_info: Catalogue, cata_tab: Table, ms_pointing: SkyCoord, flux_cut: float=0.02, radial_cut: float=1.
) -> Table:
    """Apply the flux and separation cuts to a loaded table, and transform input column names to an 
    expected set of column names. 

    Args:
        cata_info (Catalogue): Description of the catalogue from known catalogues
        cata_tab (Table): The loaded catalogue table
        ms_pointing (SkyCoord): Pointing of the measurement set
        flux_cut (float, optional): Flux cut in Jy. Defaults to 0.02.
        radial_cut (float, optional): Radial separation cut in deg. Defaults to 1..

    Returns:
        Table: _description_
    """
    flux_mask = cata_tab[cata_info.flux_col] > flux_cut
    logger.info(f"{np.sum(flux_mask)} above {flux_cut} Jy.")
    
    sky_pos = SkyCoord(
        cata_tab[cata_info.ra_col], cata_tab[cata_info.dec_col]
    )
    sep_cut = radial_cut*u.deg
    sep_mask = ms_pointing.separation(sky_pos) < sep_cut
    logger.info(f"{np.sum(sep_mask)} sources within {sep_cut:.3f} DEG.")
    
    mask = flux_mask & sep_mask
    logger.info(f"{np.sum(sep_mask)} common sources selected. ")
    
    cata_tab = cata_tab[mask]
    cols = [
        cata_info.ra_col,
        cata_info.dec_col,
        cata_info.name_col,
        cata_info.flux_col,
        cata_info.maj_col,
        cata_info.min_col,
        cata_info.pa_col,
    ]
    out_cols = ['RA', 'DEC', 'Name', 'Flux', 'Maj', 'Min', 'PA']
    cata_tab = cata_tab[cols]

    for (orig, new) in zip(cols, out_cols):
        logger.debug(f"Updating Table column {orig} to {new}.")
        cata_tab[orig].name = new

    return cata_tab

def make_ds9_region(out_path: Path, sources: List[Row]) -> Path:
    """Create a DS9 region file of the sky-model derived

    Args:
        out_path (Path): Output path to of the region file to write
        sources (List[Row]): Collection of Row objects (with normalised column names)

    Returns:
        Path: Path to the region file created
    """
    logger.info(f"Writing DS9 region file, writing to {str(out_path)}.")
    with open(out_path, "wt") as out_file:
        
        out_file.write("# DS9 region file\n")
        out_file.write("fk5\n")
        
        for source in sources:
            if source["Maj"] < 1.0 and source["Min"] < 1.0:
                out_file.write(
                    "point(%f,%f) # point=circle color=red dash=1\n" %(source["RA"], source["DEC"])
                )
            else:
                out_file.write(
                    "ellipse(%f,%f,%f,%f,%f) # color=red dash=1\n" %(source["RA"], source["DEC"], source["Maj"], source["Min"], 90.0+source["PA"])
                )
        
    return out_path
        
def main(
    ms_path: Path, cata_dir: Path=Path("."), cata_name: Optional[str]=None, spectral_index: float=-0.83, flux_cutoff: float=0.02, fwhm_scale_cutoff: float=1
) -> Path:
    """Create a sky-model to calibrate RACS based measurement sets

    Args:
        ms_path (Path): Measurement set to create sky-model for
        cata_dir (Path, optional): Directory containing known catalogues. Defaults to Path(".").
        cata_name (Optional[str], optional): Name of the catalogue. If None, select based on MS properties. Defaults to None.
        spectral_index (float, optional): The assumed spectral index to use. Defaults to -0.83.
        flux_cutoff (float, optional): Sources whose *apparent* brightness (at the lowest channel of the MS) as excluded from sky-model. Defaults to 0.02.
        fwhm_scale_cutoff (float, optional): Scaling factor to stretch the analytical FWHM by when searching for sources. Defaults to 1.

    Returns:
        Path: Path to the model file created
    """
    
    assert ms_path.exists(), f"Measurement set {ms_path} does not exist. "
    
    direction = dir_from_ms(ms_path)
    logger.info(f"Extracting local sky catalogue centred on {direction.ra.deg} {direction.dec.deg}.")

    freqs = freqs_from_ms(ms_path)
    freqcent = np.mean(np.unique(freqs))
    f0 = freqs[0]
    fN = freqs[-1]
    logger.info("Frequency range: %.3f MHz - %.3f MHz (centre = %.3f MHz)" %(f0 / 1.0e6, fN / 1.0e6, freqcent / 1.0e6))
    
    pb = GaussianPB(frequency = freqcent) #, expscaling=1.09)

    radial_cutoff = fwhm_scale_cutoff * np.degrees(pb.getFWHM()) # Go out just over 2 times the half-power point.
    logger.info("Radial cutoff = %.3f degrees" %(radial_cutoff))

    cata_info, cata_tab = load_catalogue(
        catalogue_dir=cata_dir,
        catalogue=cata_name,
        dec_point=direction.dec.deg
    )
    cata_tab = preprocess_catalogue(
        cata_info, cata_tab, ms_pointing=direction, flux_cut=flux_cutoff, radial_cut=radial_cutoff
    )

    model_path = ms_path.with_suffix(".model")

    total_flux = 0.0
    # Will be used to generate ds9
    accepted_rows: List[Row] = []

    with open(model_path, 'wt') as fout:
        logger.info(f"Writing header to {model_path}.")
        fout.write("Format = Name, Type, Ra, Dec, I, SpectralIndex, LogarithmicSI, ReferenceFrequency='888500000.0', MajorAxis, MinorAxis, Orientation\n")
        for i, row in enumerate(cata_tab):
            src_pos = SkyCoord(row["RA"]*u.deg, row["DEC"]*u.deg)
            src_sep = src_pos.separation(direction).radian
            ra_str, dec_str = src_pos.to_string(style='hmsdms', sep=":").split()
            
            # This is the AO Calibrate format
            dec_str = dec_str.replace(":", ".")
            
            s_cat = row["Flux"] / 1000. 
            s_nu_low_int = s_cat * np.power(f0 / cata_info.freq, spectral_index)
            s_nu_high_int = s_cat * np.power(fN / cata_info.freq, spectral_index)
            
            s_nu_low_app = s_nu_low_int * pb.evaluate(src_sep, freq=f0)
            s_nu_high_app = s_nu_high_int * pb.evaluate(src_sep, freq=fN)
            
            if s_nu_low_app < flux_cutoff:
                continue
            
            accepted_rows.append(row)
            
            alpha_app = np.log(s_nu_low_app / s_nu_high_app) / np.log(f0 / fN)
            s_ref = flux_nu(s_nu_low_app, alpha_app, f0, freqcent)

            total_flux += (s_nu_low_app + s_nu_high_app) / 2.
            logger.info(
                f"{len(accepted_rows):05d} {row['Name']} {s_cat=:.4f} S0={s_nu_low_int:.4f} {s_nu_low_app:.4f} SN={s_nu_high_int:.4f} {s_nu_high_app:.4f} {src_sep:.2f} deg Sref={s_ref:0.4f} alpha={alpha_app:.3f}"
            )
            
            if row["Maj"] < 1.0 and row["Min"] < 1.0:
                fout.write(
                    f"s{i:05d},POINT,{ra_str},{dec_str},{s_ref},[{alpha_app},0.0],true,{freqcent},,,\n"
                )
            else:
                fout.write(
                    f"s{i:05d},GAUSS,{ra_str},{dec_str},{s_ref},[{alpha_app},0.0],true,{freqcent},{row['Maj']},{row['Min']},{row['PA']}\n"
                )

    logger.info(f"Written {model_path}, total flux = {total_flux:.4f}, no. sources {len(accepted_rows)}. ")

    region_path = ms_path.with_suffix(".model.reg")
    make_ds9_region(
        out_path=region_path, sources=accepted_rows
    )

    return model_path

def get_parser():
    parser = ArgumentParser(description="Create a calibrate compatible sky-model for a given measurement set. ")
    
    parser.add_argument('ms', type=Path, help="Path to the measurement set to create the sky-model for")
    parser.add_argument("--assumed-alpha", type=float, default=-0.83, help="Assumed spectral index. ")
    parser.add_argument("--fwhm-scale", type=float, default=2, help="Sources within this many FWHMs are selected. ")
    parser.add_argument('--flux-cutoff', type=float, default=0.02, help="Apparent flux density cutoff for sources to be above to be included in the model. ")
    parser.add_argument('--cata-dir', type=Path, default=Path('.'), help="Directory containing known catalogues. ")
    parser.add_argument('--cata-name', type=str, choices=KNOWN_CATAS.keys(), help=f"Name of catalogue to load. Options are: {KNOWN_CATAS.keys()}.")

    return parser

if __name__ == '__main__':
    parser = get_parser()
    
    args = parser.parse_args()
    
    logger.setLevel(logging.INFO)
    
    main(
        ms_path=args.ms,
        cata_dir=args.cata_dir,
        cata_name=args.cata_name,
        fwhm_scale_cutoff=args.fwhm_scale,
        spectral_index=args.assumed_alpha
    )