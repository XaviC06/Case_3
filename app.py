"""
Dashboard: Zürich Airport 2019-2020 - wanneer en waarom lopen vluchten uit de pas?

Datasets (alle drie uit de opdracht + weer):
  1. schedule_airport.csv   - 323.461 geplande/werkelijke vluchten
  2. flightdata.zip         - 7 vluchtprofielen (per 30 s)
  3. OpenFlights airports   - locaties van luchthavens (zelfde bron als de Kaggle-dataset)
  4. 06670_csv.gz           - dagweer Zürich-Kloten (Meteostat)

Opbouw: eerste laag = 'Overzicht' (rustig beeld, geen filters nodig);
tweede laag = tabs met detail, pas zichtbaar als iemand erom vraagt.

Bronnen van overgenomen code/ideeën staan bij de functies (kopje 'Bron:') en in README.md.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st
from plotly.subplots import make_subplots

from data_loader import (data_file, DELAY_LIMIT, EXTREME_LIMIT, HOME_ICAO, WEEKDAYS, haversine_km, load_airports,
                         load_schedule, load_tracks, load_weather, nearest_airport)

st.set_page_config(page_title="Zürich Airport 2019-2020", page_icon="✈️", layout="wide")

# --------------------------------------------------------------------------- vaste instellingen
C_ARR, C_DEP, C_GREY, C_ACC = "#2b6cb0", "#dd6b20", "#8a8f98", "#c53030"   # blauw/oranje: kleurenblind-vriendelijk
TEMPLATE = "plotly_white"
CORONA_DATE = "2020-03-16"      # start Zwitserse lockdown (publiek bekend); alleen als oriëntatiepunt in grafieken
METRICS = ["Gem. vertraging (min)", "Mediane vertraging (min)", "% te laat (≥ 15 min)", "Aantal vluchten"]

df, LOG_S = load_schedule()
WEATHER, LOG_W = load_weather()
TRACKS, LOG_F = load_tracks()
AIRPORTS = load_airports()
df["bestemming"] = np.where(df.icao.isna(), "Onbekend", df.city.fillna(df.icao) + " (" + df.icao.fillna("") + ")")
YEARS = df.datum.dt.year


# --------------------------------------------------------------------------- hulpfuncties
def nl_int(n):
    return f"{n:,.0f}".replace(",", ".")


def nl_dec(x, d=1):
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pct(x, d=0):
    return nl_dec(100 * x, d) + "%"


def subset(period="2019 + 2020", richting="Alle"):
    m = pd.Series(True, index=df.index)
    if period != "2019 + 2020":
        m &= YEARS == int(period)
    if richting != "Alle":
        m &= df.richting == richting
    return df[m]


def agg_metric(d, by, metric):
    """Aggregeer een grootheid per groep. Geeft kolommen: by, waarde, n."""
    g = d.groupby(by, observed=True)
    out = g.size().rename("n").to_frame()
    if metric == "Gem. vertraging (min)":
        out["waarde"] = g.delay_min.mean()
    elif metric == "Mediane vertraging (min)":
        out["waarde"] = g.delay_min.median()
    elif metric == "% te laat (≥ 15 min)":
        out["waarde"] = g.te_laat.mean() * 100
    else:
        out["waarde"] = out.n
    return out.reset_index()


def line_style(fig, title, xtitle, ytitle, height=430):
    fig.update_layout(title=title, xaxis_title=xtitle, yaxis_title=ytitle, template=TEMPLATE, height=height,
                      margin=dict(l=10, r=10, t=60, b=10), legend=dict(orientation="h", y=1.1, x=0))
    return fig


def add_corona_marker(fig, row=None, col=None):
    fig.add_shape(type="line", x0=CORONA_DATE, x1=CORONA_DATE, y0=0, y1=1, yref="paper",
                  line=dict(color=C_GREY, dash="dot", width=1.5))
    fig.add_annotation(x=CORONA_DATE, y=1.0, yref="paper", text="Corona-maatregelen (mrt 2020)", showarrow=False,
                       xanchor="left", yanchor="bottom", font=dict(size=11, color=C_GREY))


def color_mapping(vals, mode, palette, title, fmt=lambda v: nl_dec(v, 0)):
    """Kleurschaal voor kaarten. Opklimmende (sequentiële) palette, géén regenboog.
    mode: 'Kwantielen' | 'Logaritmisch' | 'Lineair'.
    Aanpak 'kwantielen': klassen met evenveel gebieden per klasse, zodat één grote waarde niet alle kleur opeist
    (idee: gebinde choropleth, zie https://plotly.com/python/colorscales/ en https://en.wikipedia.org/wiki/Choropleth_map)."""
    vals = np.asarray(vals, dtype=float)
    if mode == "Kwantielen":
        edges = np.unique(np.quantile(vals, np.linspace(0, 1, 7)))
        if len(edges) < 2:
            edges = np.array([vals.min(), vals.min() + 1])
        k = len(edges) - 1
        idx = np.clip(np.searchsorted(edges[1:-1], vals, side="left"), 0, k - 1)
        cols = px.colors.sample_colorscale(palette, [0.2 + 0.8 * i / max(k - 1, 1) for i in range(k)])
        scale = []
        for i, c in enumerate(cols):
            scale += [[i / k, c], [(i + 1) / k, c]]
        labels = [f"{fmt(edges[i])} - {fmt(edges[i + 1])}" for i in range(k)]
        return idx + 0.5, dict(colorscale=scale, zmin=0, zmax=k,
                               colorbar=dict(title=title + "<br>(kwantielklassen)", tickmode="array",
                                             tickvals=[i + 0.5 for i in range(k)], ticktext=labels, len=0.75))
    if mode == "Logaritmisch":
        z = np.log10(np.clip(vals, 1, None))
        lo, hi = int(np.floor(z.min())), int(np.ceil(z.max()))
        ticks = list(range(lo, hi + 1))
        return z, dict(colorscale=palette, zmin=lo, zmax=hi,
                       colorbar=dict(title=title + "<br>(log-schaal)", tickmode="array", tickvals=ticks,
                                     ticktext=[nl_int(10 ** t) for t in ticks], len=0.75))
    return vals, dict(colorscale=palette, zmin=float(vals.min()), zmax=float(vals.max()),
                      colorbar=dict(title=title + "<br>(lineair)", len=0.75))


# =========================================================================== TAB 1 - OVERZICHT
@st.cache_data
def weekly_overview():
    wk = df.groupby(["week", "richting"]).size().unstack(fill_value=0)
    late = df.groupby("week").te_laat.mean() * 100
    n = df.groupby("week").size()
    full = n[(n.index >= "2019-01-07") & (n.index <= "2020-12-21")].index      # alleen volledige weken
    return wk.loc[full], late.loc[full]


@st.cache_data
def daily_table():
    d = df.groupby("datum").agg(n=("delay_min", "size"), vertraging=("delay_min", "mean"), te_laat=("te_laat", "mean"),
                                wpgt=("wpgt", "first"), wspd=("wspd", "first"), prcp=("prcp", "first"),
                                tavg=("tavg", "first"), pres=("pres", "first"))
    d["jaar"] = d.index.year.astype(str)
    return d


def r2(X, y):
    X = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    return 1 - res.var() / y.var()


@st.fragment
def tab_overzicht():
    n_tot = len(df)
    y19, y20 = df[YEARS == 2019], df[YEARS == 2020]
    daily = daily_table()
    r_drukte = daily.n.corr(daily.vertraging)
    r_drukte19 = daily[daily.jaar == "2019"].n.corr(daily[daily.jaar == "2019"].vertraging)
    d19 = df[YEARS == 2019]
    wind_hi = d19[d19.wind_klasse == "> 55 km/u"].delay_min.mean()
    wind_lo = d19[d19.wind_klasse == "< 25 km/u"].delay_min.mean()
    coded = df[df.DL1.notna()]
    rot_share = (coded.vertragingsgroep == "Rotatie (91-96)").mean()

    st.markdown(f"## Hoe drukker het is op Zürich Airport, hoe later de vluchten")
    st.caption(f"Twee jaar vluchten ({nl_int(n_tot)} bewegingen, 2019-2020), gekoppeld aan weer en luchthavendata. "
               "Vraag: *wanneer en waarom lopen vluchten uit de pas?*")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vluchten in 2 jaar", nl_int(n_tot))
    c2.metric("Te laat in 2019 (≥ 15 min)", pct(y19.te_laat.mean()))
    c3.metric("Te laat in 2020 (≥ 15 min)", pct(y20.te_laat.mean()),
              delta=f"{(y20.te_laat.mean() - y19.te_laat.mean()) * 100:+.0f} procentpunt".replace(".", ","), delta_color="off")
    apr19 = (df.maand == "2019-04-01").sum()
    apr20 = (df.maand == "2020-04-01").sum()
    c4.metric("Verkeer april 2020 t.o.v. april 2019", f"{(apr20 / apr19 - 1) * 100:.0f}%".replace(".", ","))

    wk, late = weekly_overview()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.6, 0.4],
                        subplot_titles=("Vluchten per week", "Aandeel vluchten ≥ 15 min te laat"))
    fig.add_trace(go.Scatter(x=wk.index, y=wk.Aankomst, name="Aankomst", line=dict(color=C_ARR, width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=wk.index, y=wk.Vertrek, name="Vertrek", line=dict(color=C_DEP, width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=late.index, y=late, name="% te laat", line=dict(color=C_ACC, width=2.5), showlegend=False,
                             hovertemplate="%{x|%d-%m-%Y}: %{y:.0f}%<extra></extra>"), row=2, col=1)
    fig.update_yaxes(title_text="Vluchten per week", row=1, col=1)
    fig.update_yaxes(title_text="% te laat", row=2, col=1, rangemode="tozero")
    fig.update_xaxes(title_text="Week (maandag)", row=2, col=1)
    fig.update_layout(template=TEMPLATE, height=500, margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", y=1.12, x=0))
    add_corona_marker(fig)
    st.plotly_chart(fig, width="stretch")

    a, b, c = st.columns(3)
    a.info(f"**Drukte → vertraging.** Dagen met meer vluchten hebben ook een hogere gemiddelde vertraging "
           f"(r = {nl_dec(r_drukte, 2)} over twee jaar; r = {nl_dec(r_drukte19, 2)} alleen in 2019, zonder corona-effect). Zie tab *Vertraging & weer*.")
    b.info(f"**Weer speelt mee.** In 2019 was de vertraging bij windstoten > 55 km/u gemiddeld "
           f"{nl_dec(wind_hi - wind_lo, 1)} min hoger dan bij < 25 km/u.")
    c.info(f"**Domino-effect.** {pct(rot_share)} van de vluchten mét vertragingscode is vertraagd door de "
           f"*rotatie* van het toestel (eerdere late vlucht).")

    with st.expander("Wat hebben we bewust weggelaten, en waarom?"):
        st.markdown(
            "- **Geen filters op de beginpagina.** Wie opent ziet direct het kantelpunt (corona) en dat vertraging de drukte volgt.\n"
            "- **Geen gate-, baan- en vliegtuigdetails op het eerste scherm.** Die verklaren minder dan drukte en weer; "
            "ze staan een klik verder (tab *Vertraging & weer*).\n"
            "- **Geen tweede kaart of tabel hier.** De kaart beantwoordt een ándere vraag (*waarheen vliegen we en met welke vertraging?*) "
            "en krijgt een eigen tab.\n"
            "- **De 7 Schiphol-vluchten staan apart.** Ze horen bij een andere luchthaven en zouden de boodschap vertroebelen.")


# =========================================================================== TAB 2 - DRUKTE (LIJNGRAFIEK)
@st.cache_data
def traffic_series(unit, split, dests: tuple):
    """Aantal vluchten per tijdseenheid, optioneel uitgesplitst. Vult ontbrekende tijdstippen aan met 0 (reindex)."""
    if unit == "Per uur":
        key, full = df.sched.dt.floor("h"), pd.date_range(df.datum.min(), df.datum.max() + pd.Timedelta(hours=23), freq="h")
    elif unit == "Per dag":
        key, full = df.datum, pd.date_range(df.datum.min(), df.datum.max(), freq="D")
    elif unit == "Per week":
        key, full = df.week, pd.date_range(df.week.min(), df.week.max(), freq="7D")
    else:
        key, full = df.maand, pd.date_range(df.maand.min(), df.maand.max(), freq="MS")
    if split == "Aankomst vs vertrek":
        t = df.groupby([key, df.richting]).size().unstack(fill_value=0)
    elif split == "Per bestemming/herkomst":
        sel = df[df.bestemming.isin(dests)]
        t = sel.groupby([key[sel.index], sel.bestemming]).size().unstack(fill_value=0)
    else:
        t = df.groupby(key).size().to_frame("Totaal")
    return t.reindex(full, fill_value=0)


@st.cache_data
def hour_profile(start, end):
    s = df[(df.datum >= pd.Timestamp(start)) & (df.datum <= pd.Timestamp(end))]
    ndays = max((pd.Timestamp(end) - pd.Timestamp(start)).days + 1, 1)
    p = s.groupby(["uur", "richting"]).size().unstack(fill_value=0).reindex(range(24), fill_value=0)
    for col in ["Aankomst", "Vertrek"]:
        if col not in p:
            p[col] = 0
    return p / ndays, len(s), ndays


@st.fragment
def tab_drukte():
    st.markdown("### Hoeveel vliegtuigen komen en gaan er, en wanneer?")
    wk, _ = weekly_overview()
    busiest = wk.sum(axis=1).idxmax()
    quietest = wk[wk.index >= "2020-03-30"].sum(axis=1).idxmin()

    c1, c2, c3 = st.columns([1, 1.3, 1.2])
    unit = c1.selectbox("Tijdseenheid", ["Per uur", "Per dag", "Per week", "Per maand"], index=1,
                        help="Per uur toont het dagritme, per dag/week de trend, per maand het seizoen.")
    split = c2.radio("Uitsplitsing", ["Aankomst vs vertrek", "Per bestemming/herkomst", "Totaal"], horizontal=True)
    scale = c3.radio("Y-as", ["Lineair", "Logaritmisch"], horizontal=True,
                     help="Logaritmisch laat de rustige corona-weken naast de drukke zomer van 2019 zien.")
    dests = ()
    if split == "Per bestemming/herkomst":
        top = df[df.bestemming != "Onbekend"].bestemming.value_counts()
        dests = tuple(st.multiselect("Kies bestemmingen/herkomsten (max. 6 voor leesbaarheid)", list(top.index[:60]),
                                     default=list(top.index[:4]), max_selections=6))
        if not dests:
            st.warning("Kies minstens één bestemming.")
            return
    gaps = st.checkbox("Periodes zonder vluchten tonen als **gat** (niet als lijn naar 0)", value=True,
                       help="0 vluchten komt voor bij de nachtstop (ca. 23:30-06:00) en tijdens de lockdown. "
                            "Een gat laat zien dát er niets was, in plaats van een lijn die aan de data lijkt te hangen.")
    if scale == "Logaritmisch":
        gaps = True

    t = traffic_series(unit, split, dests)
    if gaps:
        t = t.where(t > 0)
    colors = {"Aankomst": C_ARR, "Vertrek": C_DEP}
    palette = px.colors.qualitative.Safe
    fig = go.Figure()
    for i, col in enumerate(t.columns):
        fig.add_trace(go.Scattergl(x=t.index, y=t[col], name=str(col), mode="lines", connectgaps=False,
                                   line=dict(color=colors.get(col, palette[i % len(palette)]), width=1.6),
                                   hovertemplate="%{x|%d-%m-%Y %H:%M}: %{y:.0f}<extra>" + str(col) + "</extra>"))
    unit_txt = unit.lower().replace("per ", "")
    line_style(fig, f"Aantal vluchten per {unit_txt}", "Datum" if unit != "Per uur" else "Datum en uur",
               f"Aantal vluchten per {unit_txt}")
    fig.update_yaxes(type="log" if scale == "Logaritmisch" else "linear", rangemode="tozero" if scale == "Lineair" else "normal")
    fig.update_xaxes(rangeslider_visible=True, rangeselector=dict(buttons=[
        dict(count=7, label="1 wk", step="day", stepmode="backward"), dict(count=1, label="1 mnd", step="month", stepmode="backward"),
        dict(count=6, label="6 mnd", step="month", stepmode="backward"), dict(step="all", label="Alles")]))
    if unit == "Per uur":      # per uur over 2 jaar is onleesbaar: start in de drukste week; slider/knoppen om te verschuiven
        fig.update_xaxes(range=[busiest, busiest + pd.Timedelta(days=7)])
    add_corona_marker(fig)
    fig.update_layout(height=520)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "**Aggregatie en schaal.** 'Vlucht' = één beweging (landing of start). Per uur zie je het dagritme (golven rond 07, 12 en 17 uur, "
        "nachtstop zichtbaar als gat), per dag het weekpatroon, per week de trend. De lineaire schaal toont het absolute verschil; "
        "de log-schaal maakt de lockdown-weken (~100/week) leesbaar naast de zomer van 2019 (~5.000/week). "
        "Eerste en laatste week zijn gedeeltelijk gevuld (1 jan 2019 is een dinsdag).")

    st.markdown("#### Drukke en rustige periode naast elkaar")
    st.caption("Gemiddeld aantal vluchten per uur van de dag, over de gekozen periode. Beide grafieken delen dezelfde y-as, zodat de verschillen eerlijk zichtbaar zijn.")
    dmin, dmax = df.datum.min().date(), df.datum.max().date()
    cols = st.columns(2)
    defaults = {"A": (busiest.date(), (busiest + pd.Timedelta(days=6)).date()),
                "B": (quietest.date(), (quietest + pd.Timedelta(days=6)).date())}
    profiles = {}
    for col, key, name in zip(cols, ["A", "B"], ["Drukke periode", "Rustige periode"]):
        rng = col.date_input(name, value=defaults[key], min_value=dmin, max_value=dmax, key=f"cmp_{key}", format="DD-MM-YYYY")
        if isinstance(rng, (list, tuple)) and len(rng) == 2:
            profiles[key] = (rng, *hour_profile(rng[0], rng[1]))
    if len(profiles) == 2:
        ymax = max(p[1].to_numpy().max() for p in profiles.values()) * 1.1
        for col, key, name in zip(cols, ["A", "B"], ["Drukke periode", "Rustige periode"]):
            rng, prof, n, nd = profiles[key]
            f = go.Figure()
            for side, color in [("Aankomst", C_ARR), ("Vertrek", C_DEP)]:
                f.add_trace(go.Scatter(x=prof.index, y=prof[side], name=side, mode="lines+markers", line=dict(color=color, width=2.5)))
            line_style(f, f"{name}: {rng[0]:%d-%m-%Y} t/m {rng[1]:%d-%m-%Y}", "Uur van de dag (gepland)",
                       "Gem. vluchten per uur", height=340)
            f.update_yaxes(range=[0, ymax])
            f.update_xaxes(dtick=2)
            col.plotly_chart(f, width="stretch")
            col.caption(f"{nl_int(n)} vluchten in {nd} dagen = {nl_dec(n / nd, 0)} per dag.")


# =========================================================================== TAB 3 - KAART
@st.cache_data
def airport_table():
    d = df[df.icao.notna() & df.lat.notna()]
    return d


@st.fragment
def tab_kaart():
    st.markdown("### Waarheen vliegt Zürich, en met hoeveel vertraging?")
    c1, c2, c3, c4 = st.columns(4)
    view = c1.radio("Weergave", ["Landen (ingekleurd)", "Luchthavens (punten)"])
    metric = c2.selectbox("Grootheid", ["Aantal vluchten", "% te laat (≥ 15 min)", "Gem. vertraging (min)"])
    is_count = metric == "Aantal vluchten"
    modes = ["Kwantielen", "Logaritmisch", "Lineair"] if is_count else ["Kwantielen", "Lineair"]
    mode = c3.radio("Kleurschaal", modes, help="Het aantal vluchten per land is zeer scheef verdeeld (Duitsland ≫ rest). "
                                               "Kwantielen of log voorkomen dat één land alle kleur opeist.")
    region = c4.radio("Regio", ["Europa", "Wereld"], horizontal=True)

    d1, d2, d3 = st.columns(3)
    period = d1.selectbox("Periode", ["2019 + 2020", "2019", "2020"])
    richting = d2.selectbox("Richting", ["Alle", "Aankomst", "Vertrek"])
    min_n = d3.slider("Minimum aantal vluchten (voor vertragings­metingen)", 1, 500, 100,
                      help="Gemiddelden over weinig vluchten zijn onbetrouwbaar; alleen van toepassing op % te laat / gem. vertraging.")

    d = subset(period, richting)
    d = d[d.icao.notna() & d.lat.notna()]
    palette = "Blues" if is_count else "Oranges"

    if view.startswith("Landen"):
        g = d.groupby("iso3").agg(land=("country", "first"), n=("delay_min", "size"), gem=("delay_min", "mean"),
                                  late=("te_laat", "mean")).reset_index()
    else:
        g = d.groupby("icao").agg(land=("country", "first"), naam=("luchthaven", "first"), stad=("city", "first"),
                                  lat=("lat", "first"), lon=("lon", "first"), n=("delay_min", "size"),
                                  gem=("delay_min", "mean"), late=("te_laat", "mean")).reset_index()
        g["iso3"] = g.icao
    g["waarde"] = {"Aantal vluchten": g.n, "% te laat (≥ 15 min)": g.late * 100, "Gem. vertraging (min)": g.gem}[metric]
    if not is_count:
        g = g[g.n >= min_n]
    if g.empty:
        st.warning("Geen gebieden met genoeg vluchten; verlaag het minimum.")
        return

    countries = d.groupby("country").size().sort_values(ascending=False).index.tolist()
    chosen = st.multiselect("Selecteer landen (omlijnd op de kaart; details eronder)", countries)

    z, cm = color_mapping(g.waarde, mode, palette, metric)
    fmt = (lambda v: nl_int(v)) if is_count else (lambda v: nl_dec(v, 1))
    fig = go.Figure()
    if view.startswith("Landen"):
        fig.add_trace(go.Choropleth(
            locations=g.iso3, z=z, text=g.land, customdata=np.column_stack([g.n, g.gem.round(1), (g.late * 100).round(1)]),
            hovertemplate="<b>%{text}</b><br>Vluchten: %{customdata[0]:,.0f}<br>Gem. vertraging: %{customdata[1]} min"
                          "<br>% te laat: %{customdata[2]}%<extra></extra>",
            marker_line_color="white", marker_line_width=0.6, **cm))
        if chosen:
            iso_sel = d[d.country.isin(chosen)].iso3.unique()
            fig.add_trace(go.Choropleth(locations=iso_sel, z=[1] * len(iso_sel), showscale=False, hoverinfo="skip",
                                        colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
                                        marker_line_color="#111", marker_line_width=2.6))
    else:
        nmax = g.n.max()
        size = np.maximum(4, 34 * np.sqrt(g.n / nmax))           # oppervlakte ∝ aantal vluchten
        sel = g.land.isin(chosen) if chosen else pd.Series(False, index=g.index)
        fig.add_trace(go.Scattergeo(
            lon=g.lon, lat=g.lat, text=g.naam + " (" + g.icao + ")", mode="markers",
            customdata=np.column_stack([g.n, g.gem.round(1), (g.late * 100).round(1)]),
            hovertemplate="<b>%{text}</b><br>Vluchten: %{customdata[0]:,.0f}<br>Gem. vertraging: %{customdata[1]} min"
                          "<br>% te laat: %{customdata[2]}%<extra></extra>",
            marker=dict(size=size, color=z, colorscale=cm["colorscale"], cmin=cm["zmin"], cmax=cm["zmax"],
                        colorbar=cm["colorbar"], line=dict(width=np.where(sel, 2.5, 0.5), color=np.where(sel, "#111", "white")),
                        opacity=0.9), showlegend=False))
        top_lines = st.checkbox("Routes vanaf Zürich tonen (top 25)", value=False)
        home = AIRPORTS[AIRPORTS.icao == HOME_ICAO].iloc[0]
        if top_lines:
            for _, r in g.nlargest(25, "n").iterrows():
                fig.add_trace(go.Scattergeo(lon=[home.lon, r.lon], lat=[home.lat, r.lat], mode="lines", hoverinfo="skip",
                                            line=dict(width=0.8, color="rgba(60,60,60,0.35)"), showlegend=False))
    home = AIRPORTS[AIRPORTS.icao == HOME_ICAO].iloc[0]
    fig.add_trace(go.Scattergeo(lon=[home.lon], lat=[home.lat], mode="markers+text", text=["Zürich (LSZH)"], textposition="top center",
                                marker=dict(symbol="star", size=14, color="#111"), showlegend=False, hoverinfo="skip"))
    fig.update_geos(scope="europe" if region == "Europa" else "world", projection_type="natural earth" if region == "Wereld" else "equirectangular",
                    showcountries=True, countrycolor="#c9c9c9", landcolor="#f4f4f4", showocean=True, oceancolor="#e8f0f8",
                    showland=True, framecolor="#ddd")
    fig.update_layout(title=f"{metric} per {'land' if view.startswith('Landen') else 'luchthaven'} "
                            f"({period}, {richting.lower()})", template=TEMPLATE, height=560,
                      margin=dict(l=0, r=0, t=50, b=0))
    st.plotly_chart(fig, width="stretch")
    if view.startswith("Luchthavens"):
        st.caption("Grootte van de cirkel = aantal vluchten (oppervlakte evenredig); kleur = gekozen grootheid. ★ = Zürich.")

    # --- patroon in woorden (dynamisch, op basis van de gekozen selectie)
    all_d = subset(period, richting)
    all_d = all_d[all_d.country.notna()]
    share = all_d.groupby("country").size().sort_values(ascending=False) / len(all_d)
    near = (all_d.afstand_km < 1500).mean()
    by_dist = all_d.groupby("afstand_klasse", observed=True).delay_min.mean()
    st.markdown(
        f"**Patroon:** {pct(near)} van de vluchten gaat naar een bestemming binnen 1.500 km; "
        f"{share.index[0]} ({pct(share.iloc[0])}), {share.index[1]} ({pct(share.iloc[1])}) en {share.index[2]} ({pct(share.iloc[2])}) "
        f"zijn samen goed voor {pct(share.iloc[:3].sum())}. De gemiddelde vertraging verschilt weinig met de afstand "
        f"({nl_dec(by_dist.min(), 1)} tot {nl_dec(by_dist.max(), 1)} min), dus *waar je heen vliegt* verklaart weinig; "
        f"*wanneer* wel (zie tab Vertraging & weer).")

    if chosen:
        st.markdown("**Geselecteerde landen: top 10 luchthavens**")
        t = (all_d[all_d.country.isin(chosen)].groupby(["country", "luchthaven"])
             .agg(Vluchten=("delay_min", "size"), **{"Gem. vertraging (min)": ("delay_min", "mean"), "% te laat": ("te_laat", "mean")})
             .reset_index().sort_values("Vluchten", ascending=False).head(10))
        t["Gem. vertraging (min)"] = t["Gem. vertraging (min)"].round(1)
        t["% te laat"] = (t["% te laat"] * 100).round(1)
        st.dataframe(t.rename(columns={"country": "Land", "luchthaven": "Luchthaven"}), hide_index=True, width="stretch")


# =========================================================================== TAB 4 - VERTRAGING & WEER
FACTORS = {
    "Uur van de dag (gepland)": "uur", "Weekdag": "weekdag", "Maand": "maand", "Baanconcept (windrichting)": "baanconcept",
    "Windstoten": "wind_klasse", "Neerslag": "neerslag_klasse", "Afstand tot Zürich": "afstand_klasse",
    "Vliegtuigtype (top 12)": "type", "Maatschappij (top 12)": "maatschappij", "Vertragingsoorzaak (IATA-groep)": "vertragingsgroep"}
ORDERED = {"weekdag": WEEKDAYS}


@st.fragment
def tab_vertraging():
    st.markdown("### Wanneer en waarom lopen vluchten uit de pas?")
    st.markdown("#### 1. Hoe ziet vertraging eruit? (verdeling)")
    cA, cB = st.columns([3, 2])
    clip = cA.slider("Toon vertragingen tussen (min)", -60, 300, (-30, 120))
    hist_d = df[(df.delay_min >= clip[0]) & (df.delay_min <= clip[1])]
    fig = px.histogram(hist_d, x="delay_min", color="richting", nbins=int(clip[1] - clip[0]), histnorm="percent",
                       barmode="overlay", opacity=0.65, color_discrete_map={"Aankomst": C_ARR, "Vertrek": C_DEP})
    fig.add_vline(x=DELAY_LIMIT, line_dash="dot", line_color=C_GREY, annotation_text="15 min = 'te laat'")
    line_style(fig, "Verdeling van vertraging (werkelijk − gepland)", "Vertraging (min; negatief = te vroeg)", "% van de vluchten", 360)
    cA.plotly_chart(fig, width="stretch")
    s = df.groupby("richting").delay_min.agg(Vluchten="size", Gemiddelde="mean", Mediaan="median",
                                              P90=lambda x: x.quantile(0.9), Maximum="max")
    s["% te laat"] = df.groupby("richting").te_laat.mean() * 100
    cB.markdown("**Samenvatting (min)**")
    cB.dataframe(s.round(1).reset_index().rename(columns={"richting": "Richting"}), hide_index=True, width="stretch")
    mean_dep, med_dep = s.loc["Vertrek", "Gemiddelde"], s.loc["Vertrek", "Mediaan"]
    cB.markdown(f"**Wat valt op?** De verdeling is *rechtsscheef*: mediaan ({nl_dec(med_dep, 1)} min) < gemiddelde "
                f"({nl_dec(mean_dep, 1)} min) bij vertrek, door een staart van zeer late vluchten. "
                f"Vertrek loopt gemiddeld {nl_dec(s.loc['Vertrek', 'Gemiddelde'] - s.loc['Aankomst', 'Gemiddelde'], 1)} min later "
                f"dan aankomst.")

    st.markdown("#### 2. Waar zit het verschil? (kies een factor)")
    c1, c2, c3, c4 = st.columns(4)
    factor = c1.selectbox("Factor", list(FACTORS))
    metric = c2.selectbox("Grootheid", METRICS)
    period = c3.selectbox("Periode", ["2019 + 2020", "2019", "2020"], key="v_period",
                          help="2020 was veel rustiger; kijk je alleen naar 2019, dan vergelijk je dagen met vergelijkbare drukte.")
    split = c4.checkbox("Splits naar aankomst/vertrek", value=False)
    col = FACTORS[factor]
    d = subset(period).copy()
    if col == "maand":
        d["maand"] = d.maand.dt.strftime("%Y-%m")
    if col in ("type", "maatschappij"):
        d = d[d[col].isin(d[col].value_counts().index[:12])]
    by = [col, "richting"] if split else [col]
    a = agg_metric(d, by, metric)
    a = a[a.n >= 100]
    if col in ORDERED:
        a[col] = pd.Categorical(a[col], ORDERED[col], ordered=True)
    if col in ("baanconcept", "type", "maatschappij", "vertragingsgroep"):
        order = a.groupby(col, observed=True).waarde.mean().sort_values(ascending=False).index.tolist()
        a[col] = pd.Categorical(a[col], order, ordered=True)
    a = a.sort_values(by)
    fig = px.bar(a, x=col, y="waarde", color="richting" if split else None, barmode="group",
                 color_discrete_map={"Aankomst": C_ARR, "Vertrek": C_DEP}, hover_data={"n": ":,"},
                 color_discrete_sequence=[C_ACC])
    line_style(fig, f"{metric} per {factor.lower()}", factor, metric, 400)
    fig.update_layout(xaxis_type="category")
    st.plotly_chart(fig, width="stretch")
    st.caption("Groepen met minder dan 100 vluchten zijn weggelaten (te weinig om iets over te zeggen). Hover toont het aantal vluchten.")

    st.markdown("#### 3. Weer en drukte gekoppeld (per dag)")
    daily = daily_table()
    c1, c2 = st.columns([1, 2])
    wvars = {"Windstoten (km/u)": "wpgt", "Windsnelheid (km/u)": "wspd", "Neerslag (mm)": "prcp", "Temperatuur (°C)": "tavg"}
    wname = c1.selectbox("Weervariabele", list(wvars))
    wp = c1.radio("Periode", ["2019 + 2020", "2019", "2020"], key="w_period", horizontal=True)
    dd = daily if wp == "2019 + 2020" else daily[daily.jaar == wp]
    dd = dd.dropna(subset=[wvars[wname]])
    fig = px.scatter(dd, x=wvars[wname], y="vertraging", size="n", color="jaar", size_max=14, opacity=0.65,
                     color_discrete_map={"2019": C_ARR, "2020": C_DEP},
                     labels={"vertraging": "Gem. vertraging per dag (min)", wvars[wname]: wname, "n": "Vluchten", "jaar": "Jaar"})
    if len(dd) > 2:
        k, b0 = np.polyfit(dd[wvars[wname]], dd.vertraging, 1)
        xs = np.array([dd[wvars[wname]].min(), dd[wvars[wname]].max()])
        fig.add_trace(go.Scatter(x=xs, y=k * xs + b0, mode="lines", name="Lineaire trend", line=dict(color="#111", dash="dash")))
    line_style(fig, f"Dagelijkse vertraging vs. {wname.lower()} (bolgrootte = aantal vluchten)", wname, "Gem. vertraging per dag (min)", 400)
    c2.plotly_chart(fig, width="stretch")
    r = dd[wvars[wname]].corr(dd.vertraging)
    c1.metric(f"Correlatie r ({wp})", nl_dec(r, 2))

    d19 = daily[daily.jaar == "2019"].dropna()
    y = d19.vertraging.to_numpy()
    r2_n = r2(d19[["n"]].to_numpy(), y)
    r2_nw = r2(d19[["n", "wpgt", "prcp"]].to_numpy(), y)
    r2_w = r2(d19[["wpgt", "prcp"]].to_numpy(), y)
    st.markdown(
        f"**Koppeling van de datasets.** Alleen in 2019 (normaal jaar) verklaart de *drukte* {pct(r2_n)} van de variatie in dagelijkse "
        f"vertraging, het *weer* (wind + neerslag) {pct(r2_w)}, en beide samen {pct(r2_nw)}. "
        "Beide doen ertoe en versterken elkaar; drukte is iets sterker. Dit antwoord is met één dataset niet te geven (schedule + weer gekoppeld op datum).")

    corr = pd.DataFrame({
        "Drukte (vluchten/dag)": [daily[daily.jaar == j].n.corr(daily[daily.jaar == j].vertraging) for j in ["2019", "2020"]],
        "Windstoten": [daily[daily.jaar == j].wpgt.corr(daily[daily.jaar == j].vertraging) for j in ["2019", "2020"]],
        "Neerslag": [daily[daily.jaar == j].prcp.corr(daily[daily.jaar == j].vertraging) for j in ["2019", "2020"]],
        "Temperatuur": [daily[daily.jaar == j].tavg.corr(daily[daily.jaar == j].vertraging) for j in ["2019", "2020"]]},
        index=["2019", "2020"]).T.reset_index().melt(id_vars="index", var_name="Jaar", value_name="r")
    fig = px.bar(corr, x="index", y="r", color="Jaar", barmode="group", color_discrete_map={"2019": C_ARR, "2020": C_DEP})
    line_style(fig, "Correlatie met dagelijkse gem. vertraging", "Variabele", "Correlatiecoëfficiënt r", 320)
    st.plotly_chart(fig, width="stretch")


# =========================================================================== TAB 5 - VOORSPELLING
@st.cache_data
def weekly_counts():
    n = df.groupby("week").size()
    return n[(n.index >= "2019-01-07") & (n.index <= "2020-12-21")]          # alleen volledige weken


PRESETS = {
    "Stabiele periode (zomer 2019)": ("2019-05-27", 16, 8),
    "Corona-schok (feb 2020)": ("2020-02-24", 12, 6),
    "Herstel (zomer 2020)": ("2020-06-29", 8, 8),
    "Tweede golf (herfst 2020)": ("2020-09-28", 8, 8),
    "Vooruitkijken (na dec 2020)": ("2020-12-21", 8, 8),
    "Zelf kiezen": None}


def linear_forecast(y, h, alpha=0.05):
    """Lineaire regressie (OLS) door de laatste weken, met voorspelinterval.
    Bron voor get_prediction/summary_frame: https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLSResults.get_prediction.html"""
    X = sm.add_constant(np.arange(len(y)))
    m = sm.OLS(y, X).fit()
    pf = m.get_prediction(sm.add_constant(np.arange(len(y), len(y) + h), has_constant="add")).summary_frame(alpha=alpha)
    return m, pf


@st.fragment
def tab_voorspelling():
    st.markdown("### Hoe lang is een trend door te trekken?")
    wk = weekly_counts()
    c1, c2 = st.columns([1.2, 2])
    name = c1.selectbox("Scenario", list(PRESETS))
    if PRESETS[name] is None:
        train_end = pd.Timestamp(c1.select_slider("Trainen tot (maandag van de laatste trainingsweek)", wk.index.strftime("%Y-%m-%d").tolist(),
                                                  value="2020-06-29"))
        win = c1.slider("Trainingsvenster (weken)", 4, 40, 8)
        h = c1.slider("Voorspelhorizon (weken)", 1, 12, 8)
    else:
        te_s, win, h = PRESETS[name]
        train_end = pd.Timestamp(te_s)
        c1.caption(f"Training t/m week van {train_end:%d-%m-%Y}, venster {win} weken, horizon {h} weken.")
    tr = wk[wk.index <= train_end].iloc[-win:]
    future_idx = pd.date_range(train_end + pd.Timedelta(days=7), periods=h, freq="7D")
    actual = wk.reindex(future_idx).dropna()
    m, pf = linear_forecast(tr.values.astype(float), h)
    pf.index = future_idx
    for c in ["mean", "obs_ci_lower", "obs_ci_upper"]:
        pf[c] = pf[c].clip(lower=0)
    slope = m.params[1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wk.index, y=wk.values, name="Werkelijk", mode="lines", line=dict(color=C_GREY, width=2)))
    fig.add_trace(go.Scatter(x=tr.index, y=tr.values, name="Trainingsdata", mode="lines+markers", line=dict(color=C_ARR, width=3.5)))
    fig.add_trace(go.Scatter(x=pf.index, y=pf.obs_ci_upper, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=pf.index, y=pf.obs_ci_lower, mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(221,107,32,0.2)", name="95%-bandbreedte", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=pf.index, y=pf["mean"], name="Voorspelling (lineair)", mode="lines+markers",
                             line=dict(color=C_DEP, width=3, dash="dash")))
    fig.add_trace(go.Scatter(x=future_idx, y=[tr.values[-1]] * h, name="Naïef (laatste week doortrekken)", mode="lines",
                             line=dict(color="#555", width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=actual.index, y=actual.values, name="Werkelijk (achtergehouden)", mode="markers",
                             marker=dict(color=C_ACC, size=9, symbol="diamond")))
    line_style(fig, "Vluchten per week: trainen, voorspellen en toetsen", "Week (maandag)", "Vluchten per week", 470)
    zoom_lo = max(wk.index[0], train_end - pd.Timedelta(weeks=max(win, 12) + 4))
    zoom_hi = min(wk.index[-1] + pd.Timedelta(days=14), future_idx[-1] + pd.Timedelta(days=14))
    fig.update_xaxes(range=[zoom_lo, zoom_hi])
    c2.plotly_chart(fig, width="stretch")

    if len(actual):
        p = pf.loc[actual.index]
        mae = (p["mean"] - actual).abs().mean()
        naive = (tr.values[-1] - actual).abs().mean()
        mape = ((p["mean"] - actual).abs() / actual).mean()
        inside = ((actual >= p.obs_ci_lower) & (actual <= p.obs_ci_upper)).mean()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Gem. afwijking model (MAE)", f"{nl_int(mae)} vluchten/wk")
        m2.metric("Gem. afwijking naïef", f"{nl_int(naive)} vluchten/wk",
                  delta="model beter" if mae < naive else "naïef beter", delta_color="normal" if mae < naive else "inverse")
        m3.metric("Procentuele fout (MAPE)", pct(mape))
        m4.metric("Weken binnen bandbreedte", f"{inside * 100:.0f}%".replace(".", ","))
        if inside >= 0.75:
            st.success("**De aanname hield stand** in deze periode: de achtergehouden weken liggen grotendeels binnen de bandbreedte.")
        else:
            st.error("**De voorspelling werd gebroken:** de werkelijke weken vallen buiten de bandbreedte. "
                     "Een lineaire trend kent geen schokken.")
    else:
        st.info("Dit is een echte vooruitblik: er is geen werkelijke data om mee te vergelijken. "
                f"Het model verwacht {nl_int(pf['mean'].iloc[-1])} vluchten per week over {h} weken "
                f"(bandbreedte {nl_int(pf.obs_ci_lower.iloc[-1])} - {nl_int(pf.obs_ci_upper.iloc[-1])}).")

    a, b, c = st.columns(3)
    a.markdown(f"**Wat is de trend?**\n\nIn de trainingsweken verandert het verkeer met gemiddeld **{nl_int(slope)} vluchten per week** "
               f"({'groei' if slope > 0 else 'krimp'}).")
    b.markdown("**Aanname**\n\nDe ontwikkeling van de afgelopen weken zet zich rechtlijnig voort: geen seizoenseffect, geen nieuwe maatregelen. "
               "Redelijk voor **ongeveer 4-8 weken** vooruit, zolang het verkeer niet door een externe gebeurtenis wordt geraakt.")
    c.markdown("**Wat breekt de voorspelling?**\n\nEen schok van buitenaf: lockdown en reisbeperkingen (maart 2020), een tweede golf (najaar 2020), "
               "of het begin/einde van het vakantieseizoen. Probeer de scenario's hierboven.")
    st.caption("Methode: lineaire regressie (kleinste kwadraten) op wekelijkse aantallen, 95%-voorspelinterval. "
               "De 'toets' gebeurt op weken die niet in de training zaten. Eerste en laatste (onvolledige) week zijn uitgesloten. "
               "Er is slechts één normaal jaar data; daarom is een seizoensmodel niet betrouwbaar te schatten.")


# =========================================================================== TAB 6 - VLUCHTPROFIELEN (Flight data)
@st.fragment
def tab_vluchten():
    st.markdown("### Zeven vluchten van Schiphol naar Barcelona")
    st.caption("Aparte dataset (30-secondenmetingen). Andere luchthaven dan het schedule: de vluchten laten zien hoe één vlucht eruitziet.")
    ids = sorted(TRACKS.flight.unique())
    choice = st.radio("Vlucht", ["Alle 7 vergelijken"] + [f"Vlucht {i}" for i in ids], horizontal=True)
    ap_s, ds = nearest_airport(*TRACKS.sort_values("t_sec").groupby("flight")[["lat", "lon"]].first().iloc[0])
    ap_e, de = nearest_airport(*TRACKS.sort_values("t_sec").groupby("flight")[["lat", "lon"]].last().iloc[0])
    gc = haversine_km(ap_s.lat, ap_s.lon, ap_e.lat, ap_e.lon)

    if choice.startswith("Alle"):
        t = TRACKS
        summ = t.groupby("flight").agg(**{"Duur (min)": ("minuten", "max"), "Afgelegd (km)": ("afgelegd_km", "max"),
                                          "Max. hoogte (m)": ("alt_m", "max"), "Max. snelheid (kt)": ("tas_kt", "max")})
        summ["Omweg t.o.v. rechte lijn"] = (summ["Afgelegd (km)"] / gc - 1) * 100
        st.dataframe(summ.round(0).astype(int).rename_axis("Vlucht").reset_index(), hide_index=True, width="stretch")
    else:
        t = TRACKS[TRACKS.flight == int(choice.split()[1])]
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Duur", f"{t.minuten.max():.0f} min")
        k2.metric("Afgelegd", f"{nl_int(t.afgelegd_km.max())} km")
        k3.metric("Omweg t.o.v. rechte lijn", f"{(t.afgelegd_km.max() / gc - 1) * 100:.0f}%")
        k4.metric("Max. hoogte", f"{nl_int(t.alt_m.max())} m")
        k5.metric("Max. snelheid (TAS)", f"{t.tas_kt.max():.0f} kt")

    c1, c2 = st.columns([1.2, 1])
    fig = go.Figure()
    if choice.startswith("Alle"):
        for i, (fid, g) in enumerate(t.groupby("flight")):
            fig.add_trace(go.Scattergeo(lon=g.lon, lat=g.lat, mode="lines", name=f"Vlucht {fid}", line=dict(width=2)))
    else:
        fig.add_trace(go.Scattergeo(lon=t.lon, lat=t.lat, mode="lines", line=dict(width=1.5, color=C_GREY), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scattergeo(lon=t.lon, lat=t.lat, mode="markers", showlegend=False,
                                    marker=dict(size=7, color=t.alt_m, colorscale="Viridis", colorbar=dict(title="Hoogte (m)", len=0.7)),
                                    text=t.minuten.round(1).astype(str) + " min", customdata=t.alt_m.round(0),
                                    hovertemplate="%{text}<br>Hoogte: %{customdata} m<extra></extra>"))
    fig.add_trace(go.Scattergeo(lon=[ap_s.lon, ap_e.lon], lat=[ap_s.lat, ap_e.lat], mode="markers+text", showlegend=False,
                                text=[f"{ap_s['name']} ({ap_s.icao})", f"{ap_e['name']} ({ap_e.icao})"], textposition="top center",
                                marker=dict(symbol="star", size=13, color="#111")))
    fig.update_geos(fitbounds="locations", showcountries=True, countrycolor="#c9c9c9", landcolor="#f4f4f4", showocean=True,
                    oceancolor="#e8f0f8")
    fig.update_layout(title="Route", template=TEMPLATE, height=460, margin=dict(l=0, r=0, t=50, b=0),
                      legend=dict(orientation="h", y=-0.02))
    c1.plotly_chart(fig, width="stretch")

    sub = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("Hoogte", "Snelheid door de lucht (TAS)"))
    for fid, g in t.groupby("flight"):
        kw = dict(name=f"Vlucht {fid}", legendgroup=str(fid), line=dict(width=2))
        sub.add_trace(go.Scatter(x=g.minuten, y=g.alt_m, **kw), row=1, col=1)
        sub.add_trace(go.Scatter(x=g.minuten, y=g.tas_kt, showlegend=False, **{**kw, "name": None}), row=2, col=1)
    sub.update_yaxes(title_text="Hoogte (m)", row=1, col=1)
    sub.update_yaxes(title_text="Snelheid (kt)", row=2, col=1)
    sub.update_xaxes(title_text="Minuten sinds start van de meting", row=2, col=1)
    sub.update_layout(template=TEMPLATE, height=460, margin=dict(l=10, r=10, t=50, b=10), showlegend=choice.startswith("Alle"))
    c2.plotly_chart(sub, width="stretch")

    zrh_bcn = df[df.icao == ap_e.icao]
    st.caption(f"Begin- en eindpunt zijn automatisch gekoppeld aan de dichtstbijzijnde OpenFlights-luchthaven ({ap_s.icao}: {nl_dec(ds, 1)} km, "
               f"{ap_e.icao}: {nl_dec(de, 1)} km). Rechte lijn {ap_s.icao}-{ap_e.icao}: {nl_int(gc)} km. "
               f"Ter vergelijking: in het Zürich-schedule staan {nl_int(len(zrh_bcn))} vluchten van/naar {ap_e.icao} "
               f"met gemiddeld {nl_dec(zrh_bcn.delay_min.mean(), 1)} min vertraging.")


# =========================================================================== TAB 7 - DATA & OPSCHONING
@st.cache_data
def raw_missing():
    raw = pd.read_csv(data_file("schedule_airport.csv.gz"), dtype=str, keep_default_na=False)
    rows = []
    for c in raw.columns:
        rows.append({"Kolom": c, "Leeg": int((raw[c] == "").sum()), "'#N/A'": int((raw[c] == "#N/A").sum()),
                     "'-' (placeholder)": int((raw[c] == "-").sum()), "Unieke waarden": raw[c].nunique()})
    return pd.DataFrame(rows)


def key_conclusions(d):
    hours = d.groupby("uur").delay_min.agg(["mean", "size"])
    hours = hours[hours["size"] >= 1000]
    daily_ = d.groupby("datum").agg(n=("delay_min", "size"), v=("delay_min", "mean"))
    d19 = d[d.datum.dt.year == 2019]
    wind = (d19[d19.wind_klasse == "> 55 km/u"].delay_min.mean() - d19[d19.wind_klasse == "< 25 km/u"].delay_min.mean())
    groups = d.groupby("vertragingsgroep").delay_min.agg(["mean", "size"])
    groups = groups[(groups["size"] >= 1000) & (groups.index != "Geen vertragingscode")]
    return {"Aantal vluchten": nl_int(len(d)), "Gemiddelde vertraging (min)": nl_dec(d.delay_min.mean(), 2),
            "Mediane vertraging (min)": nl_dec(d.delay_min.median(), 2), "% te laat (≥ 15 min)": nl_dec(d.te_laat.mean() * 100, 1),
            "Uur met hoogste gem. vertraging (≥ 1.000 vluchten)": f"{hours['mean'].idxmax()}:00",
            "Windeffect: > 55 vs < 25 km/u in 2019 (min)": nl_dec(wind, 1),
            "Correlatie drukte ↔ vertraging (per dag)": nl_dec(daily_.n.corr(daily_.v), 2),
            "Oorzaakgroep met hoogste gem. vertraging": groups["mean"].idxmax()}


@st.fragment
def tab_data():
    st.markdown("### Data inspectie en opschoning")
    n0, n1 = int(LOG_S.iloc[0]["Rijen daarna"]), int(LOG_S.iloc[-1]["Rijen daarna"])
    k1, k2, k3 = st.columns(3)
    k1.metric("Rijen ingelezen", nl_int(n0))
    k2.metric("Rijen na opschonen", nl_int(n1), delta=f"{n1 - n0} rijen", delta_color="off")
    k3.metric("Rijen gemarkeerd i.p.v. verwijderd", nl_int(int(df.extreem.sum() + df.locatie_onbekend.sum())))

    st.markdown("#### 1. Wat zat er in de ruwe data?")
    miss = raw_missing()
    st.dataframe(miss[(miss[["Leeg", "'#N/A'", "'-' (placeholder)"]].sum(axis=1) > 0)], hide_index=True, width="stretch")
    st.caption("Alleen kolommen met leeg/#N/A/'-' getoond. Verder: 3 vluchten (Identifier) komen dubbel voor, en 6 vluchten hebben een werkelijke tijd "
               "na middernacht zonder datumwissel (vertraging van bijna −24 uur).")

    st.markdown("#### 2. Wat is ermee gedaan? (schedule airport)")
    st.dataframe(LOG_S, hide_index=True, width="stretch",
                 column_config={"Aantal geraakt": st.column_config.NumberColumn(format="%d"),
                                "Rijen daarna": st.column_config.NumberColumn(format="%d")})
    st.caption("S4 telt cellen (4 vertragingskolommen + baanconcept), de andere stappen tellen rijen. Alleen S2 verwijdert rijen (3 van 323.461 = 0,001%).")
    with st.expander("Weerdata en vluchtdata (Schiphol): opschoning"):
        st.markdown("**Weer (Meteostat, station 06670)**")
        st.dataframe(LOG_W, hide_index=True, width="stretch")
        st.markdown("**Vluchtdata (flightdata.zip, 30-secondenbestanden)**")
        st.dataframe(LOG_F, hide_index=True, width="stretch")

    st.markdown("#### 3. Waar hebben we níets verwijderd, en maakt dat uit?")
    st.markdown(f"**{nl_int(int(df.extreem.sum()))} vluchten** ({pct(df.extreem.mean(), 2)}) hebben een vertraging van meer dan {EXTREME_LIMIT} minuten. "
                "Ze zijn **niet verwijderd**: bij het merendeel staat een vertragingscode en het patroon (bv. vluchten die ruim een halve dag later vertrekken) is "
                "plausibel voor echte annuleringen/herplanningen. Hieronder zie je dat de conclusies niet van deze keuze afhangen.")
    cut = st.slider("Sluit vluchten uit met |vertraging| groter dan (min)", 60, 600, EXTREME_LIMIT, 30)
    a = key_conclusions(df)
    b = key_conclusions(df[df.delay_min.abs() <= cut])
    comp = pd.DataFrame({"Conclusie": list(a), "Alle vluchten (wat wij gebruiken)": list(a.values()),
                         f"Zonder |vertraging| > {cut} min": list(b.values())})
    comp["Zelfde conclusie?"] = np.where(comp.iloc[:, 1] == comp.iloc[:, 2], "✓ gelijk", "≈ licht verschoven")
    st.dataframe(comp, hide_index=True, width="stretch")
    st.caption("Waarden en rangorde (piekuur, grootste oorzaakgroep, richting van het windeffect) blijven in essentie gelijk; alleen gemiddelden verschuiven iets, "
               "de mediaan nauwelijks. Daarom blijven de uitschieters staan.")

    h_all = df.groupby("uur").delay_min.mean()
    h_cut = df[df.delay_min.abs() <= cut].groupby("uur").delay_min.mean()
    hh = pd.DataFrame({"Alle vluchten": h_all, f"Zonder uitschieters (> {cut} min)": h_cut})
    hh = hh[(hh.index >= 6) & (hh.index <= 22)]
    fig = px.line(hh.reset_index().melt(id_vars="uur", var_name="Dataset", value_name="Gem. vertraging"), x="uur", y="Gem. vertraging",
                  color="Dataset", markers=True, color_discrete_sequence=[C_ACC, C_GREY])
    line_style(fig, "Gemiddelde vertraging per uur: mét en zonder uitschieters", "Uur van de dag (gepland)", "Gem. vertraging (min)", 340)
    st.plotly_chart(fig, width="stretch")

    st.markdown("**De 10 meest extreme vluchten (allemaal behouden)**")
    ext = df.reindex(df.delay_min.abs().sort_values(ascending=False).index[:10])[
        ["vlucht", "richting", "bestemming", "sched", "actual", "delay_min", "vertragingsgroep"]]
    ext["delay_min"] = ext.delay_min.round(0).astype(int)
    st.dataframe(ext.rename(columns={"vlucht": "Vlucht", "richting": "Richting", "bestemming": "Route", "sched": "Gepland", "actual": "Werkelijk",
                                     "delay_min": "Vertraging (min)", "vertragingsgroep": "Oorzaak"}), hide_index=True, width="stretch")


# =========================================================================== OPBOUW VAN DE PAGINA
with st.sidebar:
    st.markdown("## ✈️ Zürich Airport")
    st.caption("Case 3 · Vluchten · Van data naar informatie")
    st.markdown("**Begin bij tab *Overzicht*.** De andere tabs beantwoorden de vervolgvragen:")
    st.markdown("- *Drukte*: wanneer is het druk?\n- *Kaart*: waarheen, en hoe laat?\n- *Vertraging & weer*: waarom?\n"
                "- *Voorspelling*: hoe lang houdt de trend stand?\n- *Vluchtprofielen*: één vlucht in detail\n- *Data & opschoning*: wat is er gecorrigeerd?")
    with st.expander("Bronnen"):
        st.markdown(
            "- `schedule_airport.csv`, `flightdata.zip`, `06670_csv.gz`: aangeleverd (Brightspace)\n"
            "- Luchthavens: [OpenFlights airports.dat](https://github.com/jpatokal/openflights) (bron van de Kaggle-dataset)\n"
            "- Haversine-formule: [Wikipedia](https://en.wikipedia.org/wiki/Haversine_formula)\n"
            "- IATA-vertragingscodes: [Wikipedia](https://en.wikipedia.org/wiki/IATA_delay_codes)\n"
            "- OLS-voorspelinterval: [statsmodels docs](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLSResults.get_prediction.html)\n"
            "- Kleurschalen/choropleth: [Plotly docs](https://plotly.com/python/colorscales/)\n"
            "- Streamlit-fragmenten en tabs: [Streamlit docs](https://docs.streamlit.io)")

tabs = st.tabs(["Overzicht", "Drukte", "Kaart", "Vertraging & weer", "Voorspelling", "Vluchtprofielen", "Data & opschoning"])
with tabs[0]:
    tab_overzicht()
with tabs[1]:
    tab_drukte()
with tabs[2]:
    tab_kaart()
with tabs[3]:
    tab_vertraging()
with tabs[4]:
    tab_voorspelling()
with tabs[5]:
    tab_vluchten()
with tabs[6]:
    tab_data()
