"""Download and compress all FITS files for all stars in specified file."""

import csv
import itertools
from argparse import ArgumentParser
from concurrent.futures import as_completed, ThreadPoolExecutor
from pathlib import Path

import numpy as np
from astropy.table import Table
from astropy.io import fits
from rich.console import Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

overall_progress = Progress(
    MofNCompleteColumn(),
    BarColumn(bar_width=None),
    "",
    TextColumn("[bold white]Elapsed:"),
    TimeElapsedColumn(),
    "",
    TextColumn("[bold white]Remaining:"),
    TimeRemainingColumn(),
)

download_progress = Progress(
    "Plate",
    TextColumn("[bright_yellow]{task.fields[plate]:>5}", justify="right"),
    " MJD",
    TextColumn("[bright_yellow]{task.fields[mjd]:>5}", justify="right"),
    " Fiber",
    TextColumn("[bright_yellow]{task.fields[fiber]:>4}", justify="right"),
)


def cli() -> ArgumentParser:
    """Create CLI argument parser."""
    parser = ArgumentParser(
        prog="stargaze",
        description="look at the stars, but quickly!",
    )

    parser.add_argument(
        "csv_path",
        help="CSV file of `version,plate,mjd,fiber` records",
    )
    parser.add_argument(
        "out_dir",
        help="directory where FITS files should be downloaded",
    )

    return parser


def process_fits_file(overall_task: TaskID, version: str, plate: str, mjd: str, fiber: str) -> None:
    """Download, compress, and save FITS file to disk."""

    filename = f"{plate:>05}/{mjd:>05}/{fiber:>04}.fits"
    out_path = Path(args.out_dir) / version / filename

    if not out_path.exists():
        task_id = download_progress.add_task("download", plate=plate, mjd=mjd, fiber=fiber)

        out_path.parent.mkdir(parents=True, exist_ok=True)

        url = (
            "https://data.sdss.org/sas/dr18/spectro/sdss/redux"
            f"/{version}/spectra/lite/{plate:>04}"
            f"/spec-{plate:>04}-{mjd:>05}-{fiber:>04}.fits"
        )

        with fits.open(url, cache=False) as hdul:
            # Remove extra columns from table
            table = Table.read(hdul["COADD"])
            table.keep_columns(["flux", "model"])
            # Overwrite HDU and save to disk
            hdul["COADD"] = fits.table_to_hdu(table)
            hdul.writeto(out_path, overwrite=True)

        download_progress.remove_task(task_id)

    overall_progress.update(overall_task, advance=1)


if __name__ == '__main__':
    args = cli().parse_args()

    with open(args.csv_path, newline="") as f:
        stars = list(csv.DictReader(f))

    with Live(Group(overall_progress, download_progress)):
        task_id = overall_progress.add_task("overall", total=len(stars))

        with ThreadPoolExecutor() as executor:
            for star in stars:
                executor.submit(process_fits_file, task_id, **star)
