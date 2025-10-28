# app.py
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from statsmodels.tsa.filters.hp_filter import hpfilter as sm_hpfilter
from pathlib import Path

# -------------------- CONFIG: file paths --------------------
ROOT = Path(__file__).parent
LIFE_XLSX = ROOT / "nltuk1517reg.xlsx"
NTA_XLSX  = ROOT / "nta_UK.xlsx"
WAGE_XLS  = ROOT / "Age Group Table 6.5a Hourly pay - Gross 2012.xls"  # set to your exact filename
POP_CSV   = ROOT / "population.csv"  # add this file to the repo

# -------------------- Helpers --------------------
def hp_trend(x, lamb=100.0):
    s = pd.Series(np.asarray(x, dtype=float))
    cycle, trend = sm_hpfilter(s, lamb=lamb)
    return trend.to_numpy()

def compute_value_functions(ages, y, c, w, S_male, S_female,
                            hours_endowment=4000, sigma=0.8, eta=2.0,
                            z0_z=0.1, beta=0.98, r_rho=0.02, hp_lambda=100.0):
    # Smooth wage and compute hours/leisure
    w_smooth = hp_trend(w, lamb=hp_lambda)
    y = np.asarray(y, float); c = np.asarray(c, float)
    w = np.asarray(w, float); w_smooth = np.asarray(w_smooth, float)
    T = min(len(ages), len(y), len(c), len(w_smooth))
    ages = np.asarray(ages[:T], int)
    y, c, w, w_smooth = y[:T], c[:T], w[:T], w_smooth[:T]
    with np.errstate(divide='ignore', invalid='ignore'):
        h = np.divide(y, w_smooth, out=np.zeros_like(y), where=~np.isclose(w_smooth, 0))
    l = hours_endowment - h

    # Full income/consumption
    yL = l * w_smooth
    yF = y + yL
    cF = c + yL

    # phi (Murphy–Topel)
    phi = (1.0/(sigma-1.0)) * (1.0 - sigma * (z0_z)**(1.0 - 1.0/sigma))
    v   = yF + phi * cF  # value of a life-year

    # Survival (align to T)
    S_m = np.asarray(S_male[:T], float)
    S_f = np.asarray(S_female[:T], float)
    S_dis_m = S_m * (beta ** ages)
    S_dis_f = S_f * (beta ** ages)

    # Expected value of remaining life, conditional on being alive at age a
    V_m = np.full(T, np.nan)
    V_f = np.full(T, np.nan)
    for i in range(T):
        if S_dis_m[i] > 0:
            V_m[i] = np.nansum(v[i:] * S_dis_m[i:]) / S_dis_m[i]
        if S_dis_f[i] > 0:
            V_f[i] = np.nansum(v[i:] * S_dis_f[i:]) / S_dis_f[i]
    return dict(ages=ages, v=v, V_m=V_m, V_f=V_f,
                y=y, c=c, w=w, w_smooth=w_smooth, h=h, l=l,
                yL=yL, yF=yF, cF=cF, S_m=S_m, S_f=S_f)

def aggregate_values(N_age, v, V, which='flow'):
    N = np.asarray(N_age, float)
    T = min(len(N), len(v), len(V))
    return float(np.nansum(N[:T] * (v[:T] if which=='flow' else V[:T])))

# -------------------- App --------------------
st.set_page_config(page_title="Full-Income Counterfactuals (Local Files)", layout="wide")
st.title("Full-Income Counterfactuals (Local Files)")

# --- Parameters (left sidebar)
with st.sidebar:
    st.header("Parameters")
    sigma = st.number_input("CRRA (σ)", value=0.8, step=0.05, min_value=0.2, max_value=3.0)
    eta   = st.number_input("Elasticity (η)", value=2.0, step=0.1, min_value=0.5, max_value=5.0)
    z0z   = st.number_input("z0/z", value=0.10, step=0.01, min_value=0.0, max_value=1.0)
    beta  = st.number_input("Discount factor (β)", value=0.98, step=0.005, min_value=0.90, max_value=0.999)
    rrho  = st.number_input("r − ρ", value=0.02, step=0.005, min_value=0.0, max_value=0.10)
    hours_endowment = st.number_input("Annual time endowment (hours)", value=4000, min_value=1000, max_value=10000, step=100)
    hp_lambda = st.number_input("HP filter λ", value=100.0, step=50.0, min_value=10.0, max_value=2000.0)

    st.header("Scenario")
    scen = st.selectbox("Choose scenario", ["None", "Fertility (TFR)", "Productivity (ages 20–64)"])

# --- Guardrails
missing = [p.name for p in [LIFE_XLSX, NTA_XLSX, WAGE_XLS, POP_CSV] if not p.exists()]
if missing:
    st.error(f"Missing file(s): {', '.join(missing)}. Place them in the repo folder and reload.")
    st.stop()

# --- Load local data ---
# Life table: use the same slice as your MATLAB (A8:L108)
life_raw = pd.read_excel(LIFE_XLSX, sheet_name=0, header=None)
block = life_raw.iloc[7:108, :12].to_numpy()
nlt_age = block[:, 0].astype(int)
l_m = block[:, 3]   # male 'l_x' (survivors out of 100k)
l_f = block[:, 9]   # female 'l_x'
S_male   = np.concatenate([l_m/100000.0, np.zeros(10)])
S_female = np.concatenate([l_f/100000.0, np.zeros(10)])

# NTA (use your real column names)
Q = pd.read_excel(NTA_XLSX)
ages_nta = Q.loc[Q['Type'].eq('NTA'), 'Age'].to_numpy()
y  = Q.loc[Q['Type'].eq('Smooth Mean'), 'Labor Income'].to_numpy()
c  = Q.loc[Q['Type'].eq('Smooth Mean'), 'Consumption'].to_numpy()

# Wages (ASHE)
wage_tbl = pd.read_excel(WAGE_XLS, sheet_name=0, header=None)
w = np.zeros(111)
w[:17]   = wage_tbl.iloc[1, 3]
w[17:21] = wage_tbl.iloc[2, 3]
w[21:29] = wage_tbl.iloc[3, 3]
w[29:39] = wage_tbl.iloc[4, 3]
w[39:49] = wage_tbl.iloc[5, 3]
w[49:59] = wage_tbl.iloc[6, 3]
w[59:   ]= wage_tbl.iloc[7, 3]

# Population
popdf = pd.read_csv(POP_CSV).sort_values('Age')
N_age = popdf['Population'].to_numpy()
ages_pop = popdf['Age'].to_numpy()

# Align everything to NTA age grid
A = ages_nta.astype(int)
S_m_interp = np.interp(A, np.arange(len(S_male)),   S_male,   left=S_male[0], right=0.0)
S_f_interp = np.interp(A, np.arange(len(S_female)), S_female, left=S_female[0], right=0.0)
w_interp   = np.interp(A, np.arange(len(w)),        w,        left=w[0], right=w[-1])
N_interp   = np.interp(A, ages_pop, N_age,          left=N_age[0], right=0.0)

# Baseline computation
base = compute_value_functions(
    ages=A, y=y, c=c, w=w_interp,
    S_male=S_m_interp, S_female=S_f_interp,
    hours_endowment=hours_endowment, sigma=sigma, eta=eta,
    z0_z=z0z, beta=beta, r_rho=rrho, hp_lambda=hp_lambda
)

TEV_flow_base  = aggregate_values(N_interp, base['v'], base['V_m'], which='flow')
TEV_stock_base = aggregate_values(N_interp, base['v'], base['V_m'], which='stock')

# -------------------- Scenario logic --------------------
TEV_flow_scn, TEV_stock_scn = TEV_flow_base, TEV_stock_base
v_new, V_new = base['v'], base['V_m']
desc = "No scenario"

if scen == "Fertility (TFR)":
    st.sidebar.markdown("---")
    st.sidebar.subheader("Fertility settings")
    # TFR slider: baseline 1.5, allow increments of 0.1
    TFR_baseline = 1.5
    TFR = st.sidebar.slider("Total Fertility Rate (TFR)", min_value=1.0, max_value=2.5, value=1.5, step=0.1)
    dTFR = TFR - TFR_baseline

    # Convert ΔTFR to Δ annual births using a simple approximation:
    # births ≈ (TFR / span_years) * (# women aged 15–49)
    span_years = 35
    # We don't have sex-by-age; approximate 50% women
    mask_15_49 = (A >= 15) & (A <= 49)
    women_15_49 = 0.5 * np.sum(N_interp[mask_15_49])
    births_base = (TFR_baseline / span_years) * women_15_49
    delta_births = (dTFR / span_years) * women_15_49

    # Monetary effect this year on the stock of the living:
    # ΔTEV_stock ≈ ΔBirths × V(0)
    V0 = float(V_new[0])
    TEV_stock_scn = TEV_stock_base + delta_births * V0
    TEV_flow_scn  = TEV_flow_base  # immediate flow unchanged without cohort propagation
    desc = f"TFR from {TFR_baseline:.1f} → {TFR:.1f} (ΔBirths ≈ {delta_births:,.0f}/yr; baseline births ≈ {births_base:,.0f}/yr)"

elif scen == "Productivity (ages 20–64)":
    st.sidebar.markdown("---")
    st.sidebar.subheader("Productivity settings")
    pct = st.sidebar.slider("Productivity boost (%)", -50, 200, 10) / 100.0
    amin, amax = 20, 64
    alpha = st.sidebar.slider("Consumption pass-through (α)", 0.0, 1.0, 0.5, 0.05)

    mask = (A >= amin) & (A <= amax)
    y_scn = y.copy(); c_scn = c.copy()
    y_scn[mask] = y_scn[mask] * (1 + pct)
    c_scn[mask] = c_scn[mask] * (1 + alpha * pct)

    scn = compute_value_functions(
        ages=A, y=y_scn, c=c_scn, w=w_interp,
        S_male=S_m_interp, S_female=S_f_interp,
        hours_endowment=hours_endowment, sigma=sigma, eta=eta,
        z0_z=z0z, beta=beta, r_rho=rrho, hp_lambda=hp_lambda
    )
    v_new, V_new = scn['v'], scn['V_m']
    TEV_flow_scn  = aggregate_values(N_interp, v_new, V_new, 'flow')
    TEV_stock_scn = aggregate_values(N_interp, v_new, V_new, 'stock')
    desc = f"Productivity +{pct*100:.0f}% on ages {amin}-{amax}, α={alpha:.2f}"

# -------------------- Display --------------------
col1, col2 = st.columns(2)
with col1:
    st.subheader("Baseline")
    st.metric("TEV_flow (this year)", f"£{TEV_flow_base:,.0f}")
    st.metric("TEV_stock (living population)", f"£{TEV_stock_base:,.0f}")
with col2:
    st.subheader("Scenario")
    st.metric("TEV_flow (this year)", f"£{TEV_flow_scn:,.0f}",
              f"{(TEV_flow_scn-TEV_flow_base)/max(1,TEV_flow_base)*100:.2f}%")
    st.metric("TEV_stock (living population)", f"£{TEV_stock_scn:,.0f}",
              f"{(TEV_stock_scn-TEV_stock_base)/max(1,TEV_stock_base)*100:.2f}%")
st.caption(desc)

# Plots
fig, ax = plt.subplots(1, 2, figsize=(12,4))
ax[0].plot(A, base['v'], label='Baseline'); ax[0].plot(A, v_new, label='Scenario')
ax[0].set_title("Value of a life-year v(a)"); ax[0].set_xlabel("Age"); ax[0].set_ylabel("£"); ax[0].legend()
ax[1].plot(A, base['V_m'], label='Baseline'); ax[1].plot(A, V_new, label='Scenario')
ax[1].set_title("Value of remaining life V(a)"); ax[1].set_xlabel("Age"); ax[1].set_ylabel("£"); ax[1].legend()
st.pyplot(fig)

# For fertility, show ΔBirths summary
if scen == "Fertility (TFR)":
    st.info(f"Approximate **annual** ΔBirths = {delta_births:,.0f}; Baseline annual births ≈ {births_base:,.0f}. "
            f"Immediate stock effect: ΔBirths × V(0) = £{(delta_births*V0):,.0f}. "
            "Add a cohort propagation later to see flows over time.")
