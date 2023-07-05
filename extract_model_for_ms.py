#!/usr/bin/env python

import logging
import yaml
from typing import NamedTuple, Optional, Tuple, List, Dict
from pathlib import Path
from argparse import ArgumentParser
from functools import partial

import numpy as np
from astropy.coordinates import Angle, SkyCoord
from astropy import units as u
from astropy.table import Table, QTable
from astropy.table.row import Row
from casacore.tables import table
from scipy.optimize import curve_fit

logger = logging.getLogger()
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)


class Catalogue(NamedTuple):
    """A basic structure used to describe a known catalogue."""

    file_name: str
    freq: float  # Hertz
    ra_col: str
    dec_col: str
    name_col: str
    flux_col: str
    maj_col: str
    min_col: str
    pa_col: str
    alpha_col: Optional[str] = None  # Used to scale the SED
    q_col: Optional[str] = None  # Used to scale the SED


class CurvedPL(NamedTuple):
    norm: float
    alpha: float
    q: float
    ref_nu: float


class GaussianTaper(NamedTuple):
    freqs: np.ndarray  # The frequencies the beam is evaluated at
    atten: np.ndarray  # The attenuation of the response
    fwhms: np.ndarray  # The full-width at half-maximum corresponding to freqs
    offset: float  # Angular offset of the source


# These columns are what we will normalise the all columns and units to
NORM_COLS = {"flux": "Jy", "maj": "arcsecond", "min": "arcsecond", "pa": "deg"}

KNOWN_CATAS: Dict[str, Catalogue] = {
    "SUMSS": Catalogue(
        file_name="sumsscat.Mar-11-2008_CLH.fits",
        freq=843e6,
        ra_col="RA",
        dec_col="Dec",
        name_col="Mosaic",
        flux_col="Sp",
        maj_col="dMajAxis",
        min_col="dMinAxis",
        pa_col="dPA",
    ),
    "RACS": Catalogue(
        file_name="racs-low.fits",
        freq=887.56e6,
        ra_col="RA",
        dec_col="Dec",
        name_col="Gaussian_ID",
        flux_col="Total_flux_Gaussian",
        maj_col="DC_Maj",
        min_col="DC_Min",
        pa_col="DC_PA",
    ),
    "NVSS": Catalogue(
        file_name="NVSS_vizier.fits",
        freq=1400e6,
        ra_col="RAJ2000",
        dec_col="DEJ2000",
        name_col="NVSS",
        flux_col="S1_4",
        maj_col="MajAxis",
        min_col="MinAxis",
        pa_col="PA",
    ),
}


def generate_gaussian_pb(
    freqs: u.Quantity, aperture: u.Quantity, offset: u.Quantity
) -> GaussianTaper:
    """Calculate the theoretical Gaussian taper for an aperture of
    known size

    Args:
        freqs (u.Quantity): Frequencies to evaluate the beam at
        aperture (u.Quantity): Size of the dish
        offset (u.Quantity): Offset from the centre of the beam

    Returns:
        GaussianTaper: Numerical results of the theoretical gaussian primary beam
    """
    c = 299792458.0 * u.meter / u.second
    solid_angle = 4.0 * np.log(2)

    offset = offset.to(u.rad)
    freqs_hz = freqs.to(u.hertz)
    aperture_m = aperture.to(u.meter)

    fwhms = (c / freqs_hz / aperture_m).decompose() * u.rad

    e = (-offset * offset * solid_angle / (fwhms**2)).decompose()

    taper = np.exp(e)

    return GaussianTaper(freqs=freqs, atten=taper, fwhms=fwhms, offset=offset)


def curved_power_law(
    nu: np.ndarray, norm: float, alpha: float, beta: float, ref_nu: float
) -> np.ndarray:
    """A curved power law model.

    Args:
        nu (np.ndarray): Frequency array.
        norm (float): Reference flux.
        alpha (float): Spectral index.
        beta (float): Spectral curvature.
        ref_nu (float): Reference frequency.

    Returns:
        np.ndarray: Model flux.
    """
    x = nu / ref_nu
    c = np.exp(beta * np.log(x) ** 2)

    return norm * x**alpha * c


def fit_curved_pl(freqs: u.Quantity, flux: u.Quantity, ref_nu: u.Quantity) -> CurvedPL:
    """Fit some specified set of datapoints with a generic
    curved powerlaw. This is _not_ meant for real data, ratther
    as a way of representing the functional form of a model
    after it has been perturbed by some assumed primary beam.

    Args:
        freqs (np.ndarray): Frequencies corresponding to each brightness
        flux (np.ndarray): Brightness corresponding to each frequency
        ref_nu (float): Reference frequency that the model is set to

    Returns:
        CurvedPL: The fitted parameter results
    """
    # Strip out the Quantity stuff
    freqs = freqs.to(u.Hz).value
    flux = flux.to(u.Jy).value
    ref_nu = ref_nu.to(u.Hz).value

    p0 = (
        np.median(flux),
        np.log(flux[0] / flux[-1]) / np.log(freqs[0] / freqs[-1]),
        0.0,
    )

    curve_pl = partial(curved_power_law, ref_nu=ref_nu)

    p, cov = curve_fit(curve_pl, freqs, flux, p0)

    params = CurvedPL(norm=p[0], alpha=p[1], q=p[2], ref_nu=ref_nu)

    return params


def evaluate_src_model(freqs: u.Quantity, src_row: Row, ref_nu: u.Quantity) -> u.Jy:
    """Evaluate a SED of an object using its recordded
    Normalisation, alpha and q components.

    Args:
        freqs (u.Quantity): Frequencies to evaluate
        src_row (Row): Source propertieis from which the parameters are extracted
        ref_nu (u.Quantity): Reference frequency of the model parameterization

    Returns:
        u.Jy: Brightness of model evaluated across frequency
    """

    fluxes = curved_power_law(
        nu=freqs.to(u.Hz).value,
        norm=src_row["flux"].to(u.Jy).value,
        alpha=src_row["alpha"],
        beta=src_row["q"],
        ref_nu=ref_nu.to(u.Hz).value,
    )

    return fluxes * u.Jy


def dir_from_ms(ms_path: Path) -> SkyCoord:
    """Extract the pointing direction from a measurement set

    Args:
        ms_path (Path): Path to the measurement set to query

    Returns:
        SkyCoord: Pointing direction on the sky of the measurement set
    """
    tp = table(f"{str(ms_path)}/FIELD", readonly=True, ack=False)
    p_phase = tp.getcol("PHASE_DIR")
    tp.close()

    td = table(str(ms_path), readonly=True, ack=False)
    field = td.getcol("FIELD_ID", 0, 1)[0]

    return SkyCoord(
        Angle(p_phase[field][0][0], unit=u.rad), Angle(p_phase[field][0][1], unit=u.rad)
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
    return np.sort(freqs)


def get_known_catalogue(cata: str) -> Catalogue:
    """Get the parameters of a known catalogue

    TODO: Replace with configuration based method to load known cata

    Args:
        cata (str): The lookup name of the catalogue

    Returns:
        Catalogue: properties of known catalogue
    """
    assert (
        cata.upper() in KNOWN_CATAS.keys()
    ), f"'{cata}' not a known catalogue. Acceptable keys are: {KNOWN_CATAS.keys()}."

    cata_info = KNOWN_CATAS[cata.upper()]
    logger.info(f"Loading {cata}={cata_info.file_name}")

    return cata_info


def load_catalogue(
    catalogue_dir: Path,
    catalogue: Optional[str] = None,
    ms_pointing: Optional[SkyCoord] = None,
    assumed_alpha: float=-0.83,
    assumed_q: float=0.0
) -> Tuple[Catalogue, Table]:
    """Load in a catalogue table given a name or measurement set declinattion.

    Args:
        catalogue_dir (Path): Directory containing known catalogues
        catalogue (Optional[str], optional): Catalogue name to look up from known catalogues. Defaults to None.
        ms_pointing (Optional[SkyCoord], optional): Pointing direction of the measurement set. Defaults to None.
        assumed_alpha (float, optional): The assumed spectral index to use if there is no spectral index column known in model catalogue. Defaults to -0.83.
        assumed_q (float, optional): The assumed curvature to use if there is no curvature column known in model catalogue. Defaults to 0.0.
        
    Raises:
        FileNotFoundError: Raised when a catalogue can not be resolved.

    Returns:
        Tuple[Catalogue,Table]: The `Catalogue` information and `Table` of components loaded
    """
    assert (
        catalogue is not None or ms_pointing is not None
    ), "Either catalogue or dec_point have to be provided. "

    if catalogue:
        logger.info(f"Loading provided catalogue {catalogue=}")
        cata = get_known_catalogue(catalogue)

    else:
        # Assertion is done to keep the linters happy
        assert ms_pointing is not None, f"Expected SkyCoord object, received None. "
        dec_point = float(ms_pointing.dec.deg)
        logger.info(f"Automatically loading catalogue based on {dec_point=:.2f}")

        if dec_point < -75.0:
            cata = get_known_catalogue("SUMSS")
        elif dec_point < 26.0:
            cata = get_known_catalogue("RACS")
        else:
            cata = get_known_catalogue("NVSS")

    cata_path = catalogue_dir / cata.file_name

    if not cata_path.exists():
        raise FileNotFoundError(f"Catalogue {cata_path} not found.")

    cata_tab = Table.read(cata_path)
    logger.info(f"Loaded table, found {len(cata_tab)} sources. ")

    _cols = cata._asdict()
    if cata.alpha_col is None:
        logger.info(f"No 'alpha' column, adding default spectral index of {assumed_alpha:.3f}. ")
        cata_tab["alpha"] = assumed_alpha
        _cols["alpha_col"] = "alpha"
    if cata.q_col is None:
        logger.info(f"No 'q' column, adding default {assumed_q:.3f}. ")
        cata_tab["q"] = assumed_q
        _cols["q_col"] = "q"

    cata = Catalogue(**_cols)

    return (cata, cata_tab)


def preprocess_catalogue(
    cata_info: Catalogue,
    cata_tab: Table,
    ms_pointing: SkyCoord,
    flux_cut: float = 0.02,
    radial_cut: u.deg = 1.0 * u.deg,
) -> QTable:
    """Apply the flux and separation cuts to a loaded table, and transform input column names to an
    expected set of column names.

    Args:
        cata_info (Catalogue): Description of the catalogue from known catalogues
        cata_tab (Table): The loaded catalogue table
        ms_pointing (SkyCoord): Pointing of the measurement set
        flux_cut (float, optional): Flux cut in Jy. Defaults to 0.02.
        radial_cut (u.deg, optional): Radial separation cut in deg. Defaults to 1..

    Returns:
        QTable: _description_
    """
    # First apply pre-processing options
    flux_mask = cata_tab[cata_info.flux_col] > flux_cut
    logger.info(f"{np.sum(flux_mask)} above {flux_cut} Jy.")

    sky_pos = SkyCoord(cata_tab[cata_info.ra_col], cata_tab[cata_info.dec_col])
    sep_mask = ms_pointing.separation(sky_pos) < radial_cut
    logger.info(f"{np.sum(sep_mask)} sources within {radial_cut.to(u.deg):.3f}.")

    mask = flux_mask & sep_mask
    logger.info(f"{np.sum(sep_mask)} common sources selected. ")

    cata_tab = cata_tab[mask]

    # Rename the columns to a expected form
    cols = [
        cata_info.ra_col,
        cata_info.dec_col,
        cata_info.name_col,
        cata_info.flux_col,
        cata_info.maj_col,
        cata_info.min_col,
        cata_info.pa_col,
        cata_info.alpha_col,
        cata_info.q_col,
    ]
    out_cols = ["RA", "DEC", "name", "flux", "maj", "min", "pa", "alpha", "q"]
    new_cata_tab = cata_tab[cols]

    for orig, new in zip(cols, out_cols):
        logger.debug(f"Updating Table column {orig} to {new}.")
        new_cata_tab[orig].name = new

    # Put the columns into expected units
    for key, unit_str in NORM_COLS.items():
        new_cata_tab[key] = new_cata_tab[key].to(u.Unit(unit_str))

    return QTable(new_cata_tab)


def make_ds9_region(out_path: Path, sources: List[Row]) -> Path:
    """Create a DS9 region file of the sky-model derived

    Args:
        out_path (Path): Output path to of the region file to write
        sources (List[Row]): Collection of Row objects (with normalised column names)

    Returns:
        Path: Path to the region file created
    """
    logger.info(
        f"Creating DS9 region file, writing {len(sources)} regions to {str(out_path)}."
    )
    with open(out_path, "wt") as out_file:
        out_file.write("# DS9 region file\n")
        out_file.write("fk5\n")

        for source in sources:
            if source["maj"] < 1.0 * u.arcsecond and source["min"] < 1.0 * u.arcsecond:
                out_file.write(
                    "point(%f,%f) # point=circle color=red dash=1\n"
                    % (source["RA"].value, source["DEC"].value)
                )
            else:
                out_file.write(
                    "ellipse(%f,%f,%f,%f,%f) # color=red dash=1\n"
                    % (
                        source["RA"].value,
                        source["DEC"].value,
                        source["maj"].value,
                        source["min"].value,
                        90.0 + source["pa"].value,
                    )
                )

    return out_path


def make_hyperdrive_model(out_path: Path, sources: List[Tuple[Row, CurvedPL]]) -> Path:
    """Writes a Hyperdrive sky-model to a yaml file.

    Args:
        out_path (Path): The output path that the sky-model would be written to
        sources (List[Tuple[Row,CurvedPL]]): Collection of sources to write, including the
        normalied row and the results of fitting to the estimated apparent SED

    Returns:
        Path: The path of the file created
    """
    logger.info(
        f"Creating hyperdrive sky-model, writing {len(sources)} components to {out_path}."
    )
    src_list = {}

    for row, cpl in sources:
        logger.debug(row)

        src_ra = float(row["RA"].to(u.deg).value)
        src_dec = float(row["DEC"].to(u.deg).value)
        comp_type = (
            "point"
            if (row["maj"] < 1.0 * u.arcsecond and row["min"] < 1.0 * u.arcsecond)
            else {
                "gaussian": {
                    "maj": float(row["maj"].to(u.arcsecond).value),
                    "min": float(row["min"].to(u.arcsecond).value),
                    "pa": float(row["pa"].to(u.arcsecond).value),
                }
            }
        )
        flux_type = {
            "curved_power_law": {
                "si": float(cpl.alpha),
                "q": float(cpl.q),
                "fd": {"freq": float(cpl.ref_nu), "i": float(cpl.norm)},
            }
        }

        src_list[row["name"]] = [
            {
                "ra": src_ra,
                "dec": src_dec,
                "comp_type": comp_type,
                "flux_type": flux_type,
            }
        ]

    with open(out_path, "w") as out_file:
        yaml.dump(src_list, stream=out_file)

    return out_path


def main(
    ms_path: Path,
    cata_dir: Path = Path("."),
    cata_name: Optional[str] = None,
    assumed_alpha: float = -0.83,
    assumed_q: float = 0.0,
    flux_cutoff: float = 0.02,
    fwhm_scale_cutoff: float = 1,
) -> None:
    """Create a sky-model to calibrate RACS based measurement sets

    Args:
        ms_path (Path): Measurement set to create sky-model for
        cata_dir (Path, optional): Directory containing known catalogues. Defaults to Path(".").
        cata_name (Optional[str], optional): Name of the catalogue. If None, select based on MS properties. Defaults to None.
        assumed_alpha (float, optional): The assumed spectral index to use if there is no spectral index column known in model catalogue. Defaults to -0.83.
        assumed_q (float, optional): The assumed curvature to use if there is no curvature column known in model catalogue. Defaults to 0.0.
        flux_cutoff (float, optional): Sources whose *apparent* brightness (at the lowest channel of the MS) as excluded from sky-model. Defaults to 0.02.
        fwhm_scale_cutoff (float, optional): Scaling factor to stretch the analytical FWHM by when searching for sources. Defaults to 1.

    """

    assert ms_path.exists(), f"Measurement set {ms_path} does not exist. "

    direction = dir_from_ms(ms_path)
    logger.info(
        f"Extracting local sky catalogue centred on {direction.ra.deg} {direction.dec.deg}."
    )

    freqs = freqs_from_ms(ms_path) * u.Hz
    logger.info(
        f"Frequency range: {freqs[0]/1000.:.3f} MHz - {freqs[-1]/1000.:.3f} MHz (centre = {np.mean(freqs/1000.):.3f} MHz)"
    )

    pb = generate_gaussian_pb(freqs=freqs, aperture=12.0 * u.m, offset=0 * u.rad)

    radial_cutoff = (
        fwhm_scale_cutoff * pb.fwhms[0]
    ).decompose()  # The lowest frequency FWHM is largest
    logger.info("Radial cutoff = %.3f degrees" % (radial_cutoff.to(u.deg).value))

    cata_info, cata_tab = load_catalogue(
        catalogue_dir=cata_dir, catalogue=cata_name, ms_pointing=direction, assumed_alpha=assumed_alpha, assumed_q=assumed_q
    )
    cata_tab = preprocess_catalogue(
        cata_info,
        cata_tab,
        ms_pointing=direction,
        flux_cut=flux_cutoff,
        radial_cut=radial_cutoff,
    )

    total_flux: u.Jy = 0.0 * u.Jy
    accepted_rows: List[Tuple[Row, CurvedPL]] = []

    for i, row in enumerate(cata_tab):
        src_pos = SkyCoord(row["RA"], row["DEC"])
        src_sep = src_pos.separation(direction)

        # Get the primary beam reasponse
        gauss_taper = generate_gaussian_pb(
            freqs=freqs, aperture=12.0 * u.m, offset=src_sep
        )
        
        # Calculate the expected model
        src_model = evaluate_src_model(
            freqs=freqs, src_row=row, ref_nu=cata_info.freq * u.Hz
        )
        
        # Estimate the apparent model (intrinsic*response), and
        # then numerically fit to it
        predict_model = fit_curved_pl(
            freqs=freqs, flux=src_model * gauss_taper.atten, ref_nu=freqs[0]
        )

        if predict_model.norm < flux_cutoff:
            continue

        accepted_rows.append((row, predict_model))
        total_flux += predict_model.norm * u.Jy

        logger.info(
            f"{len(accepted_rows):05d} Sep={src_sep.to(u.deg):.3f} S_ref={predict_model.norm:.3f} SI={predict_model.alpha:.3f} q={predict_model.q:.3f}"
        )

    logger.info(
        f"\nCreated model, total apparent flux = {total_flux:.4f}, no. sources {len(accepted_rows)}.\n"
    )

    hyperdrive_path = ms_path.with_suffix(".hyp.yaml")
    make_hyperdrive_model(out_path=hyperdrive_path, sources=accepted_rows)

    # TODO: Fix the sources kwarg
    region_path = ms_path.with_suffix(".model.reg")
    make_ds9_region(out_path=region_path, sources=[r[0] for r in accepted_rows])

    # TODO: What to return? Total flux/no sources? Path to models created?
    return 


def get_parser():
    parser = ArgumentParser(
        description="Create a calibrate compatible sky-model for a given measurement set. "
    )

    parser.add_argument(
        "ms", type=Path, help="Path to the measurement set to create the sky-model for"
    )
    parser.add_argument(
        "--assumed-alpha", type=float, default=-0.83, help="Assumed spectral index when no appropriate column in sky-catalogue. "
    )
    parser.add_argument(
        "--assumed-q", type=float, default=0.0, help="Assumed curvature when no apropriate column in sky-catalogue. "
    )
    parser.add_argument(
        "--fwhm-scale",
        type=float,
        default=2,
        help="Sources within this many FWHMs are selected. ",
    )
    parser.add_argument(
        "--flux-cutoff",
        type=float,
        default=0.02,
        help="Apparent flux density (in Jy) cutoff for sources to be above to be included in the model. ",
    )
    parser.add_argument(
        "--cata-dir",
        type=Path,
        default=Path("."),
        help="Directory containing known catalogues. ",
    )
    parser.add_argument(
        "--cata-name",
        type=str,
        choices=KNOWN_CATAS.keys(),
        help=f"Name of catalogue to load. Options are: {KNOWN_CATAS.keys()}.",
    )

    return parser


if __name__ == "__main__":
    parser = get_parser()

    args = parser.parse_args()

    logger.setLevel(logging.INFO)

    main(
        ms_path=args.ms,
        cata_dir=args.cata_dir,
        cata_name=args.cata_name,
        flux_cutoff=args.flux_cutoff,
        fwhm_scale_cutoff=args.fwhm_scale,
        assumed_alpha=args.assumed_alpha,
        assumed_q=args.assumed_q
    )
