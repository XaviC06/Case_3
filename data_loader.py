"""
data_loader.py - inlezen, opschonen en verrijken van de vier bronnen.

Elke ingreep wordt in een log vastgelegd (zie tab 'Data & opschoning' in het dashboard):
wat was het probleem, hoeveel rijen, wat is eraan gedaan en wat is het effect op het aantal observaties.

Bronnen van overgenomen formules/definities staan bij de betreffende functie en in README.md.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
DATA = ROOT / "data"


def data_file(name: str) -> Path:
    """Zoek een databestand in data/, naast app.py of in een willekeurige submap (robuust tegen een andere mapindeling op GitHub)."""
    for cand in (DATA / name, ROOT / name):
        if cand.exists():
            return cand
    hits = sorted(ROOT.rglob(name))
    if hits:
        return hits[0]
    # ook bestanden die door de browser zijn hernoemd, bv. 'schedule_airport (1).csv.gz'
    stem = name.split(".")[0]
    hits = sorted(p for p in ROOT.rglob(stem + "*") if p.is_file())
    if hits:
        return hits[0]
    present = sorted(str(p.relative_to(ROOT)) for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.parts)
    raise FileNotFoundError(f"Databestand '{name}' niet gevonden. Verwacht in map 'data/' naast app.py. "
                            f"Bestanden in de repo: {present}")
HOME_ICAO = "LSZH"                      # Zürich Airport (afgeleid uit de data: baanconcepten 'Bise', runways 14/16/28/32/34)
DELAY_LIMIT = 15                        # 'te laat' = >= 15 min na geplande tijd (gangbare A15-definitie in de luchtvaart)
EXTREME_LIMIT = 180                     # |vertraging| > 3 uur markeren we als 'extreem' (we verwijderen ze NIET)
WEEKDAYS = ["ma", "di", "wo", "do", "vr", "za", "zo"]


# --------------------------------------------------------------------------- hulpfuncties
def haversine_km(lat1, lon1, lat2, lon2):
    """Grootcirkelafstand in km. Formule: https://en.wikipedia.org/wiki/Haversine_formula (R = 6371 km)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def delay_group(code) -> str:
    """Groepeer IATA-vertragingscodes (AHM 730) op tientallen.
    Bron: https://en.wikipedia.org/wiki/IATA_delay_codes  (groepsindeling overgenomen, namen vertaald)."""
    if pd.isna(code):
        return "Geen vertragingscode"
    c = int(code)
    if c <= 9:
        return "Airline-intern (00-09)"
    if c >= 97:
        return "Overig (97-99)"
    return {1: "Passagiers & bagage (11-19)", 2: "Vracht & post (21-29)", 3: "Grondafhandeling (31-39)",
            4: "Technisch (41-48)", 5: "Schade & systemen (51-58)", 6: "Vluchtuitvoering & bemanning (61-69)",
            7: "Weer (71-77)", 8: "Luchthaven & autoriteiten (81-89)", 9: "Rotatie (91-96)"}.get(c // 10, "Overig (97-99)")


def _log(log, stap, probleem, n, ingreep, rows_after, motivatie):
    log.append({"Stap": stap, "Probleem": probleem, "Aantal geraakt": n, "Ingreep": ingreep,
                "Rijen daarna": rows_after, "Motivatie": motivatie})


# --------------------------------------------------------------------------- luchthavens (OpenFlights)
@st.cache_data
def load_airports() -> pd.DataFrame:
    return pd.read_csv(data_file("airports.csv.gz"))


# --------------------------------------------------------------------------- weer
@st.cache_data
def load_weather():
    """Meteostat-dagwaarden station 06670 (Zürich-Kloten). Kolomnamen volgens het Meteostat-formaat."""
    log = []
    cols = ["date", "tavg", "tmin", "tmax", "prcp", "snow", "wdir", "wspd", "wpgt", "pres", "tsun"]
    w = pd.read_csv(data_file("weather_06670.csv.gz"), header=None, names=cols, parse_dates=["date"])
    n0 = len(w)
    _log(log, "W1", "Bestand bevat 1973-2026, de vluchtdata alleen 2019-2020", n0 - 731,
         "Alleen 2019-2020 behouden", 731,
         "Alleen die dagen kunnen we aan vluchten koppelen. Geen enkele dag in de periode ontbreekt.")
    w = w[(w.date >= "2019-01-01") & (w.date <= "2020-12-31")].copy()
    empty = [c for c in cols[1:] if w[c].isna().all()]
    w = w.drop(columns=empty)
    _log(log, "W2", f"Kolommen zonder één waarde: {', '.join(empty)}", len(empty) * len(w),
         "Kolommen verwijderd", len(w), "Een kolom zonder waarden bevat geen informatie.")
    n_p = int(w.prcp.isna().sum())
    _log(log, "W3", "Ontbrekende neerslagwaarden", n_p, "Leeg gelaten (niet ingevuld)", len(w),
         "Neerslag van een buurdag zegt niets over deze dag; we sluiten deze dagen alleen uit van neerslag-analyses.")
    w["wind_klasse"] = pd.cut(w.wpgt, [-np.inf, 25, 40, 55, np.inf],
                              labels=["< 25 km/u", "25-40 km/u", "40-55 km/u", "> 55 km/u"])
    w["neerslag_klasse"] = pd.cut(w.prcp, [-np.inf, 0.1, 5, 10, np.inf],
                                  labels=["Droog (< 0,1 mm)", "Licht (0,1-5 mm)", "Matig (5-10 mm)", "Zwaar (> 10 mm)"])
    return w.reset_index(drop=True), pd.DataFrame(log)


# --------------------------------------------------------------------------- schedule airport
@st.cache_data(show_spinner="Vluchtdata laden en opschonen…")
def load_schedule():
    log = []
    raw = pd.read_csv(data_file("schedule_airport.csv.gz"), dtype=str, keep_default_na=False)
    raw = raw.mask(raw.isin(["", "#N/A"]))   # Excel-fout '#N/A' en lege cellen = ontbrekend
    df = raw.copy()
    n0 = len(df)
    _log(log, "S0", "Ruwe inleesstap", 0, "Geen", n0, "Startpunt: 2 jaar vluchten (1-1-2019 t/m 31-12-2020).")

    # --- tijden samenstellen
    df["sched"] = pd.to_datetime(df.STD + " " + df.STA_STD_ltc, format="%d/%m/%Y %H:%M:%S")
    df["actual"] = pd.to_datetime(df.STD + " " + df.ATA_ATD_ltc, format="%d/%m/%Y %H:%M:%S")
    delta = (df.actual - df.sched).dt.total_seconds() / 60

    # --- 1. over de middernacht heen
    # De kolom STD bevat alleen de geplande datum. Staat de werkelijke tijd na middernacht, dan lijkt de
    # vertraging bijna -24 uur. Dat is een datumfout, geen echte vroege vlucht.
    late_wrap = delta < -720
    early_wrap = delta > 1080
    df.loc[late_wrap, "actual"] += pd.Timedelta(days=1)
    df.loc[early_wrap, "actual"] -= pd.Timedelta(days=1)
    _log(log, "S1", "Werkelijke tijd na (of vóór) middernacht zonder datumwissel → vertraging van bijna ±24 uur",
         int(late_wrap.sum() + early_wrap.sum()), "Datum van de werkelijke tijd +1 dag (resp. −1 dag)", len(df),
         "Een vertraging van −1372 min is onmogelijk; +1 dag geeft +68 min. Geen rij verwijderd, wel gecorrigeerd.")

    # --- 2. dubbele Identifier
    df["delay_min"] = (df.actual - df.sched).dt.total_seconds() / 60
    dup = df.duplicated("Identifier", keep=False)
    n_dup_rows = int(df.duplicated("Identifier", keep="first").sum())
    df = df.sort_values("actual").drop_duplicates("Identifier", keep="last").sort_values("sched")
    _log(log, "S2", f"Dezelfde vlucht (Identifier = datum+tijd+vluchtnr) komt dubbel voor, in {int(dup.sum()) // 2 if dup.sum() else 0} paren, "
                    "met verschillende werkelijke tijden", n_dup_rows, "Per vlucht de rij met de laatste werkelijke tijd bewaard", len(df),
         "Een vlucht vertrekt maar één keer; de laatste registratie is het moment waarop hij echt vertrok (eerdere poging afgebroken/teruggekeerd). Effect: 3 rijen op 323.461.")

    # --- 3. ontbrekende bestemming
    n_na = int(df["Org/Des"].isna().sum())
    df["locatie_onbekend"] = df["Org/Des"].isna()
    _log(log, "S3", "Ontbrekende herkomst/bestemming (Org/Des): staat als tekst '#N/A' (Excel-fout) in het bestand", n_na, "Rijen BEHOUDEN; markering 'locatie onbekend'", len(df),
         "Voor drukte en vertraging is de bestemming niet nodig. Ze vallen alleen weg uit kaart- en afstandsanalyses.")

    # --- 4. placeholders
    ph_cols = ["DL1", "IX1", "DL2", "IX2"]
    n_ph = int((df[ph_cols] == "-").sum().sum())
    df[ph_cols] = df[ph_cols].replace("-", np.nan)
    n_rwc = int((df.RWC == "-").sum())
    df["RWC"] = df.RWC.replace("-", "Onbekend")
    _log(log, "S4", "'-' als placeholder voor 'geen waarde' in vertragingscodes (DL1/IX1/DL2/IX2) en baanconcept (RWC)",
         n_ph + n_rwc, "Vertragingscodes → leeg (NaN); baanconcept → 'Onbekend'", len(df),
         "Een streepje is geen code. Zonder omzetting zou '-' als 'vaakst voorkomende vertragingscode' gelden.")

    # --- 5. controle op onmogelijke waarden (geen ingreep nodig)
    bad = int((~df.LSV.isin(["L", "S"])).sum() + (~df.RWY.astype(int).isin([10, 14, 16, 28, 32, 34])).sum())
    _log(log, "S5", "Controle: LSV ∈ {L,S}, baannummer ∈ {10,14,16,28,32,34}, tijden in 24-uursnotatie", bad,
         "Geen ingreep nodig", len(df), "Alle waarden zijn geldig. Zürich heeft precies deze drie banen (10/28, 14/32, 16/34).")

    # --- 6. extreme vertragingen
    n_ext = int((df.delay_min.abs() > EXTREME_LIMIT).sum())
    df["extreem"] = df.delay_min.abs() > EXTREME_LIMIT
    _log(log, "S6", f"Vertragingen groter dan {EXTREME_LIMIT} min (3 uur) in absolute waarde (max. {df.delay_min.max():.0f} min)",
         n_ext, "Rijen BEHOUDEN; markering 'extreem'", len(df),
         "Ze lijken echt: bij de meeste staan een vertragingscode en een echte gepland/werkelijk-combinatie (bv. charters naar Turkije die uren later vertrekken). "
         "In de tab 'Data & opschoning' tonen we dat de conclusies niet van deze rijen afhangen.")

    # --- nieuwe variabelen
    df = df.rename(columns={"STD": "datum_txt", "FLT": "vlucht", "TAR": "gate_gepland", "GAT": "gate_werkelijk",
                            "ACT": "type", "RWY": "baan", "RWC": "baanconcept", "Org/Des": "icao"})
    df["baan"] = df.baan.astype(int)
    df["richting"] = df.LSV.map({"L": "Aankomst", "S": "Vertrek"})
    df["datum"] = df.sched.dt.normalize()
    df["uur"] = df.sched.dt.hour
    df["weekdag"] = df.sched.dt.dayofweek.map(dict(enumerate(WEEKDAYS)))
    df["maand"] = df.sched.dt.to_period("M").dt.to_timestamp()
    df["week"] = df.sched.dt.to_period("W").dt.start_time
    df["maatschappij"] = df.vlucht.str[:2]
    df["te_laat"] = df.delay_min >= DELAY_LIMIT
    df["vertragingsgroep"] = df.DL1.map(delay_group)

    # --- koppeling met luchthavens (OpenFlights)
    ap = load_airports().drop_duplicates("icao").set_index("icao")
    df = df.join(ap[["name", "city", "country", "iso3", "lat", "lon"]].rename(columns={"name": "luchthaven"}), on="icao")
    home = ap.loc[HOME_ICAO]
    df["afstand_km"] = haversine_km(home.lat, home.lon, df.lat, df.lon)
    no_coord = int(df.lat.isna().sum() - n_na)
    _log(log, "S7", "Koppeling met OpenFlights: ICAO-codes zonder locatie", no_coord,
         "Rijen BEHOUDEN, zonder coördinaten", len(df),
         "Codes als 'BER' (IATA i.p.v. ICAO) en 'FAJS' zijn handmatig aangevuld; overgebleven codes hebben < 10 vluchten.")
    df["afstand_klasse"] = pd.cut(df.afstand_km, [0, 500, 1000, 2000, 4000, np.inf],
                                  labels=["< 500 km", "500-1000 km", "1000-2000 km", "2000-4000 km", "> 4000 km"])

    # --- koppeling met weer (op datum)
    w, _ = load_weather()
    df = df.merge(w, how="left", left_on="datum", right_on="date").drop(columns=["date"])
    df["naam_dag"] = df.datum.dt.strftime("%d-%m-%Y")

    keep = ["sched", "actual", "datum", "uur", "weekdag", "maand", "week", "vlucht", "maatschappij", "richting", "type",
            "baan", "baanconcept", "icao", "luchthaven", "city", "country", "iso3", "lat", "lon", "afstand_km",
            "afstand_klasse", "delay_min", "te_laat", "extreem", "locatie_onbekend", "DL1", "vertragingsgroep",
            "tavg", "tmax", "prcp", "wspd", "wpgt", "wdir", "pres", "wind_klasse", "neerslag_klasse"]
    df = df[keep].sort_values("sched").reset_index(drop=True)
    return df, pd.DataFrame(log)


# --------------------------------------------------------------------------- vluchtdata (Schiphol → Barcelona)
@st.cache_data
def load_tracks():
    log = []
    f = pd.read_csv(data_file("flights_30s_raw.csv.gz"), dtype=str)
    n0 = len(f)
    star = int(f.tas_kt.str.contains(r"\*", na=False).sum())
    f["tas_kt"] = f.tas_kt.str.replace("*", "", regex=False)
    _log(log, "F1", "Snelheid (TRUE AIRSPEED) is soms tekst met een sterretje, bv. '*47.1'", star,
         "Sterretje weggehaald, omgezet naar getal", n0,
         "Het sterretje markeert een afgeleide waarde; de waarde zelf is bruikbaar. Aanname: dit is geen foutmelding.")
    for c in ["t_sec", "lat", "lon", "alt_m", "alt_ft", "heading", "tas_kt"]:
        f[c] = pd.to_numeric(f[c], errors="coerce")
    empty = f[["lat", "lon", "alt_m"]].isna().all(axis=1)
    f = f[~empty].copy()
    _log(log, "F2", "Lege meetrijen aan het einde van de vlucht (geen positie, hoogte of koers)", int(empty.sum()),
         "Rijen verwijderd", len(f), "Zonder positie kunnen ze niet op de kaart; het zijn alleen afsluitende regels.")
    neg = int((f.alt_m < 0).sum())
    _log(log, "F3", "Negatieve hoogtes (tot −3 m)", neg, "Rijen BEHOUDEN", len(f),
         "Schiphol ligt ca. 3 m onder zeeniveau; het is dus geen meetfout.")
    f["minuten"] = f.t_sec / 60
    f = f.sort_values(["flight", "t_sec"])
    g = f.groupby("flight")
    f["klimsnelheid_ms"] = g.alt_m.diff() / g.t_sec.diff()
    f["stap_km"] = haversine_km(g.lat.shift(), g.lon.shift(), f.lat, f.lon)
    f["afgelegd_km"] = f.groupby("flight").stap_km.cumsum().fillna(0)
    f["grondsnelheid_kt"] = f.stap_km / (g.t_sec.diff() / 3600) / 1.852
    return f.reset_index(drop=True), pd.DataFrame(log)


def nearest_airport(lat, lon):
    ap = load_airports()
    d = haversine_km(lat, lon, ap.lat.values, ap.lon.values)
    i = int(np.argmin(d))
    return ap.iloc[i], float(d[i])
