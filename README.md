# ✈️ Zürich Airport 2019-2020: wanneer en waarom lopen vluchten uit de pas?

Streamlit-dashboard voor Case 3 (Vluchten). Draait zonder handmatige stappen: alle data staat in `data/`.

## Starten

```bash
pip install -r requirements.txt
streamlit run app.py
```

**Publiceren (voorwaarde van de case):** push deze map naar GitHub → [share.streamlit.io](https://share.streamlit.io) → *New app* → repo kiezen, *Main file path* = `app.py`. Zet de link in je inlevering.

## Mappen

| Bestand | Doel |
|---|---|
| `app.py` | Dashboard (7 tabs) |
| `data_loader.py` | Inlezen, opschonen (met log), nieuwe variabelen, koppelingen |
| `prepare_data.py` | Eenmalig: bronbestanden → `data/` (niet nodig om de app te draaien) |
| `data/` | Gecomprimeerde bronnen + `airports.csv.gz` (OpenFlights) |

## Waar staat welk rubriekcriterium?

| Criterium | Waar in het dashboard |
|---|---|
| Opschonen | Tab *Data & opschoning*: ruwe inspectie, logtabel per stap met effect op aantal rijen, wat is **niet** verwijderd + gevoeligheidsanalyse (slider) |
| Data naar informatie | Nieuwe variabelen: vertraging in minuten (met middernacht-correctie), `te_laat`, afstand (haversine), vertragingsgroep (IATA), weerklassen. Koppeling schedule × luchthavens × weer: tab *Vertraging & weer* (blok 3) en *Kaart* |
| Voorspellen | Tab *Voorspelling*: lineaire regressie, 95%-bandbreedte, toets op achtergehouden weken, naïeve vergelijking, vijf scenario's inclusief 'voorspelling breekt' |
| Informatiearchitectuur | Eerste laag = tab *Overzicht* (geen filters nodig); detail pas in tabs; bewust weggelaten staat in een uitklap |
| Lijngrafiek | Tab *Drukte*: tijdseenheid kiezen, aankomst/vertrek of per bestemming, rangeslider, gaten zichtbaar, log-schaal, drukke vs rustige periode naast elkaar |
| Kaart | Tab *Kaart*: landen ingekleurd of luchthavens als punten, legenda, landen selecteerbaar, kwantielen/log/lineair, opklimmende kleurschaal |

## Bronnen en overgenomen code

Alle code is zelf geschreven; formules/definities en patronen zijn overgenomen uit:

- Haversine-formule: <https://en.wikipedia.org/wiki/Haversine_formula> (`data_loader.haversine_km`; de aangeleverde `vinc.py` is niet gebruikt)
- IATA-vertragingscodes en groepsindeling: <https://en.wikipedia.org/wiki/IATA_delay_codes> (`data_loader.delay_group`)
- Voorspelinterval OLS: <https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLSResults.get_prediction.html>
- Gebinde choropleth/kleurschalen: <https://plotly.com/python/colorscales/>
- Luchthavens: OpenFlights `airports.dat` (<https://github.com/jpatokal/openflights>), de bron van de Kaggle-dataset uit de opdracht. Vier ICAO-codes die erin ontbreken (FAJS, BER, ISL, ECN) zijn handmatig aangevuld (zie `prepare_data.py`).

> Als jullie code uit andere bronnen (StackOverflow, Streamlit-voorbeelden) toevoegen: zet de link in een comment bij die code, anders is het criterium hoogstens *Onvoldoende*. Zorg ook dat jullie kunnen uitleggen wat elke functie doet.

## Aannames en beperkingen (goed om te noemen in de presentatie)

- Het schedule betreft **Zürich (LSZH)**; dat is afgeleid uit de data (Zwitserse baanconcepten als 'Bise', banen 14/16/28/32/34, weerstation 06670 = Zürich-Kloten). De 7 vluchten in `flightdata.zip` vertrekken vanaf **Schiphol** (naar Barcelona) en staan daarom apart.
- 'Te laat' = ≥ 15 minuten na de geplande tijd.
- Weer is op dagniveau gekoppeld (geen uurweer), dus de weerkoppeling is grof.
- Er zijn maar twee jaar data, waarvan 2020 sterk door corona bepaald is. Een seizoensmodel is daarom niet betrouwbaar te schatten; de voorspelling is bewust een kortetermijn-trend.
- Landnamen op de kaart komen uit OpenFlights en zijn Engelstalig.
