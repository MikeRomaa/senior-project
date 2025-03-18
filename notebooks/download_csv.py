from astropy.io import fits
from astropy.table import Table, vstack
from multiprocessing.pool import ThreadPool
import requests
import matplotlib.pyplot as plt
import numpy as np
import cudf
import csv
import io

# For this section, read the skysever-dump.csv file and condense it into
# a seperate csv file for later use

def process_transmit_files_table(url: str):
    try:
        file = io.BytesIO(requests.get(url).content)

        with fits.open(file) as hdul:
            table = Table.read(hdul["COADD"])
            table.keep_columns(["flux", "model"])

        return table
    
    except Exception as e:
        print(f"Failed to process {url}: {e}")
        return None
    

with open("./skyserver-dump.csv", newline="") as file:
    try:
        reader = csv.DictReader(file)
        urls = [row["url"] for row in reader]

    except Exception as e:
        print("Process failed")


fits_tables = ThreadPool(8).imap_unordered(process_transmit_files_table, urls)

fits_tables = [t for t in fits_tables if t is not None]

final_table = vstack(fits_tables)

with open("final_output.csv", mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(final_table)

final_table.write("final_output.fits", format = "fits", overwrite = True)

print(f"The fits file has been created")

hdul = fits.open("final_output.fits")
hdul.info()
print("-------------------------------------------------------")
print(hdul["COADD"].columns)
