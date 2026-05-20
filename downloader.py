import os
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timezone


def download_flac_files(
    house,
    week,
    year,
    hours=None,
    days=(2, 4, 6),          # Tuesday, Thursday, Saturday
    active_hrs=False,
    active_range=(7, 23),    # UTC hours
    download_dir="~/thesis"
):
    """
    Download UK-DALE FLAC files using UTC timestamp filtering.

    Parameters
    ----------
    house : int
        House number.

    week : int
        Week number.

    year : int
        Year folder.

    hours : int or None
        Maximum number of matching files to download.
        None = download all matching files.

    days : tuple
        ISO weekday numbers:
        Monday=1 ... Sunday=7

        Example:
        (2,4,6) = Tue, Thu, Sat

    active_hrs : bool
        If True, filter by UTC hour range.

    active_range : tuple
        UTC hour range.

        Example:
        (7,23) keeps:
        07:00 <= hour < 23:00 UTC

    download_dir : str
        Save directory root.
    """

    # Local save directory
    download_dir = os.path.expanduser(
        f"{download_dir}/house_{house}/flac_files/{year}/wk{week}"
    )
    os.makedirs(download_dir, exist_ok=True)

    # UK-DALE URL
    base_url = (
        f"https://dap.ceda.ac.uk/edc/efficiency/residential/"
        f"EnergyConsumption/Domestic/UK-DALE-2015/UK-DALE-16kHz/"
        f"house_{house}/{year}/wk{week}/"
    )

    # Fetch page
    try:
        response = requests.get(base_url)
        response.raise_for_status()

    except requests.HTTPError:
        print(f"ERROR: URL not found:\n{base_url}")
        return

    # Parse file links
    soup = BeautifulSoup(response.text, "html.parser")

    files = [
        link.get("href")
        for link in soup.find_all("a")
        if link.get("href") and link.get("href").endswith(".flac")
    ]

    if not files:
        print("No FLAC files found.")
        return

    files.sort()

    selected_files = []

    for file_name in files:

        # Example:
        # vi-2013-03-18T00:00:01+0000_0.flac

        try:
            timestamp_str = file_name.replace("vi-", "").split("_")[0]

            dt = datetime.fromtimestamp(
            int(timestamp_str),
            tz=timezone.utc
)

        except Exception:
            print(f"Could not parse timestamp: {file_name}")
            continue

        # UTC weekday
        weekday = dt.isoweekday()

        if weekday not in days:
            continue

        # UTC hour filtering
        if active_hrs:
            start_hr, end_hr = active_range

            if not (start_hr <= dt.hour < end_hr):
                continue

        selected_files.append(file_name)

    # Optional limit
    if hours is not None:
        selected_files = selected_files[:hours]

    print(f"Selected {len(selected_files)} files.")

    # Download files
    for file_name in selected_files:

        clean_name = file_name

        if clean_name.startswith("vi-"):
            clean_name = clean_name[3:]

        # Remove trailing _x
        base = clean_name.split("_")[0]
        clean_name = f"{base}.flac"

        output_path = os.path.join(download_dir, clean_name)

        if os.path.exists(output_path):
            print(f"Skipping existing: {clean_name}")
            continue

        url = base_url + file_name

        print(f"Downloading: {file_name}")

        try:
            r = requests.get(url, stream=True)
            r.raise_for_status()

        except requests.HTTPError:
            print(f"Failed: {file_name}")
            continue

        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"Saved: {output_path}")



def download_dat_files(house, channels, download_dir="~/thesis/data"):
    """
    Downloads mains.dat + appliance .dat files for a given house.
    Skips files that already exist in the download directory.

    Args:
        house (int): House number
        channels (list[int]): List of appliance channels
        download_dir (str): Directory to save the downloaded files
    """
    download_dir = os.path.expanduser(f"{download_dir}/house_{house}/dat_files/")
    os.makedirs(download_dir, exist_ok=True)

    base_url = f"https://dap.ceda.ac.uk/edc/efficiency/residential/EnergyConsumption/Domestic/UK-DALE-2015/UK-DALE-disaggregated/house_{house}/"


    # Download appliance channels
    for channel in channels:
        output_path = os.path.join(download_dir, f"house{house}_channel{channel}.dat")
        if os.path.exists(output_path):
            print(f"File already exists, skipping: {output_path}")
            continue

        url = f"{base_url}channel_{channel}.dat"
        print(f"Downloading channel {channel} → {output_path} ...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Download complete for house {house}, channel {channel}!")
        except requests.HTTPError:
            print(f"ERROR: File not found for house {house}, channel {channel}: {url}")