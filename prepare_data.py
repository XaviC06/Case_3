"""
prepare_data.py - eenmalig uitvoeren om de aangeleverde bronbestanden om te zetten
naar compacte bestanden in ./data (dit is NIET nodig om het dashboard te starten;
de resultaten staan in de repo).

Er wordt hier bewust NIET opgeschoond: alle opschoning gebeurt zichtbaar in
data_loader.py, zodat het dashboard (tab 'Data & opschoning') laat zien wat er is gedaan.

Gebruik:
    python prepare_data.py <map_met_bronbestanden>
Verwacht in die map: schedule_airport.csv, 06670_csv.gz, flightdata.zip
(en airports.dat van OpenFlights, zie BRONNEN in README.md).
"""
import sys, zipfile, io, gzip, shutil, re
from pathlib import Path
import pandas as pd
import pycountry

src = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
out = Path(__file__).parent / "data"
out.mkdir(exist_ok=True)

# 1. Schedule airport: ongewijzigd, alleen gecomprimeerd
pd.read_csv(src / "schedule_airport.csv", encoding="utf-8-sig", dtype=str, keep_default_na=False)\
  .to_csv(out / "schedule_airport.csv.gz", index=False)

# 2. Weer (Meteostat-formaat, station 06670 Zürich-Kloten): ongewijzigd gekopieerd
shutil.copy(src / "06670_csv.gz", out / "weather_06670.csv.gz")

# 3. Flight data: de 7 bestanden met 30-seconden-data samengevoegd (waarden als tekst, onbewerkt)
frames = []
with zipfile.ZipFile(src / "flightdata.zip") as z:
    for name in sorted(z.namelist()):
        m = re.match(r"30Flight (\d+)\.xlsx", name)
        if not m:
            continue
        df = pd.read_excel(io.BytesIO(z.read(name)), dtype=str)
        df.columns = ["t_sec", "lat", "lon", "alt_m", "alt_ft", "heading", "tas_kt"]
        df.insert(0, "flight", int(m.group(1)))
        frames.append(df)
pd.concat(frames).to_csv(out / "flights_30s_raw.csv.gz", index=False)

# 4. OpenFlights luchthavens (zelfde bron als de Kaggle-dataset uit de opdracht)
cols = ["id", "name", "city", "country", "iata", "icao", "lat", "lon", "alt_ft",
        "tz_offset", "dst", "tz", "type", "source"]
ap = pd.read_csv(src / "airports.dat", header=None, names=cols, na_values=["\\N"], keep_default_na=False)
ap = ap[(ap.type == "airport") & ap.icao.notna() & (ap.icao != "")]

MANUAL = {"Russia": "RUS", "South Korea": "KOR", "North Korea": "PRK", "Czech Republic": "CZE", "Burma": "MMR",
          "Congo (Brazzaville)": "COG", "Congo (Kinshasa)": "COD", "Cape Verde": "CPV", "Ivory Coast": "CIV",
          "Macedonia": "MKD", "Syria": "SYR", "Iran": "IRN", "Libya": "LBY", "Laos": "LAO", "Vietnam": "VNM",
          "Brunei": "BRN", "Moldova": "MDA", "Bolivia": "BOL", "Venezuela": "VEN", "Tanzania": "TZA",
          "Taiwan": "TWN", "Palestine": "PSE", "Kosovo": "XKX", "Falkland Islands": "FLK", "Macau": "MAC",
          "Hong Kong": "HKG", "Swaziland": "SWZ", "East Timor": "TLS", "Virgin Islands": "VIR",
          "Svalbard": "SJM", "Reunion": "REU", "Saint Helena": "SHN", "Micronesia": "FSM", "Turkey": "TUR"}

def iso3(name):
    if name in MANUAL:
        return MANUAL[name]
    try:
        return pycountry.countries.lookup(name).alpha_3
    except LookupError:
        return None

ap["iso3"] = ap.country.map(iso3)

# Codes die in schedule_airport voorkomen maar niet (of onder een andere code) in OpenFlights staan.
# Handmatig aangevuld (publiek bekende coördinaten, bron: Wikipedia). Alleen de vier grootste gevallen;
# de rest (< 10 vluchten per code) blijft bewust zonder locatie.
SUPPLEMENT = [
    ("FAJS", "JNB", "O.R. Tambo International Airport", "Johannesburg", "South Africa", "ZAF", -26.1392, 28.2460),
    ("BER",  "BER", "Berlin Brandenburg Airport", "Berlin", "Germany", "DEU", 52.3667, 13.5033),
    ("ISL",  "ISL", "Istanbul Atatürk Airport", "Istanbul", "Turkey", "TUR", 40.9769, 28.8146),
    ("ECN",  "ECN", "Ercan International Airport", "Nicosia", "Cyprus", "CYP", 35.1547, 33.4961),
]
sup = pd.DataFrame(SUPPLEMENT, columns=["icao", "iata", "name", "city", "country", "iso3", "lat", "lon"])
ap = pd.concat([ap[sup.columns], sup], ignore_index=True).drop_duplicates("icao", keep="last")
ap.to_csv(out / "airports.csv.gz", index=False)
print("klaar:", {p.name: round(p.stat().st_size / 1e6, 2) for p in out.iterdir()}, "MB")
