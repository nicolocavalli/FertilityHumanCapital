# app.py
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from statsmodels.tsa.filters.hp_filter import hpfilter as sm_hpfilter

# ---------- Helpers ----------
def hp_trend(x, lamb=100.0):
    s = pd.Series(np.asarray(x, dtype=float))
    cycle, trend = sm_hpfilter(s, lamb=lamb)
    return trend.to_numpy()

def compute_value_functions(ages, y, c, w, S_male, S_female,
                            hours_endowment=4000, sigma=0.8, eta=2.0,
                            z0_z=0.1, beta=0.98, r_rho=0.02, hp_lambda=100.0):
    """Replicates your pipeline, returns dict with v, V, and pieces."""
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
        num_m = np.nansum(v[i:] * S_dis_m[i:])
        den_m = S_dis_m[i]
        V_m[i] = num_m/den_m if den_m > 0 else np.nan
        num_f = np.nansum(v[i:] * S_dis_f[i:])
        den_f = S_dis_f[i]
        V_f[i] = num_f/den_f if den_f > 0 else np.nan

    return dict(ages=ages, v=v, V_m=V_m, V_f=V_f,
                y=y, c=c, w=w, w_smooth=w_smooth, h=h, l=l,
                yL=yL, yF=yF, cF=cF, S_m=S_m, S_f=S_f, phi=phi)

def aggregate_values(N_age, v, V, which='flow'):
    """Sum over ages using population counts."""
    N = np.asarray(N_age, float)
    T = min(len(N), len(v), len(V))
    if which == 'flow':
        return float(np.nansum(N[:T] * v[:T]))
    else:
        return float(np.nansum(N[:T] * V[:T]))

def survival_to_one_year_probs(S):
    """Given S(a)=P(survive to age a), return p(a)=P(survive from a to a+1)."""
    S = np.asarray(S, float)
    p = np.ones_like(S)
    p[:-1] = np.divide(S[1:], S[:-1], out=np.ones_like(S[:-1]), where=~np.isclose(S[:-1], 0))
    p[-1] = 0.0
    return p

def probs_to_survival(p):
    """Reconstruct S from one-year survival p(a)."""
    S = np.empty_like(p, float)
    S[0] = 1.0
    for a in range(len(p)-1):
        S[a+1] = S[a] * p[a]
    return S

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Full-Income Counterfactuals", layout="wide")
st.title("Full‑Income Counterfactuals: Fertility vs Productivity vs Older‑Age Quality")

with st.sidebar:
    st.header("1) Upload data")
    life_file = st.file_uploader("ONS Life Table (.xlsx)", type=["xlsx"])
    nta_file  = st.file_uploader("NTA profiles (.xlsx)", type=["xlsx"])
    wage_file = st.file_uploader("ASHE wages (.xls/.xlsx)", type=["xls","xlsx"])
    pop_file  = st.file_uploader("Population by age (.csv with Age,Population)", type=["csv"])

    st.header("2) Core parameters")
    sigma = st.number_input("CRRA (σ)", value=0.8, step=0.05, min_value=0.2, max_value=3.0)
    eta   = st.number_input("Elasticity (η)", value=2.0, step=0.1, min_value=0.5, max_value=5.0)
    z0z   = st.number_input("z0/z", value=0.10, step=0.01, min_value=0.0, max_value=1.0)
    beta  = st.number_input("Discount factor (β)", value=0.98, step=0.005, min_value=0.90, max_value=0.999)
    rrho  = st.number_input("r − ρ", value=0.02, step=0.005, min_value=0.0, max_value=0.10)
    hours_endowment = st.number_input("Annual time endowment (hours)", value=4000, min_value=1000, max_value=10000, step=100)
    hp_lambda = st.number_input("HP filter λ", value=100.0, step=50.0, min_value=10.0, max_value=2000.0)

    st.header("3) Scenario")
    scenario = st.selectbox("Choose scenario", ["Fertility", "Productivity (all or ages)", "Human capital (young ages)", "Older‑age quality (survival or QALY)", "Combined"])

    if scenario == "Fertility":
        births_delta = st.number_input("Δ births (absolute)", value=10000, step=1000, min_value=0)
        horizon_years = st.slider("PV horizon (years)", 1, 100, 90)
    if scenario.startswith("Productivity"):
        pct = st.slider("Productivity boost (%)", -50, 200, 10) / 100.0
        amin, amax = st.slider("Affected ages", 0, 100, (20, 64))
        alpha = st.slider("Consumption pass-through (α)", 0.0, 1.0, 0.5, 0.05)
    if scenario.startswith("Human capital"):
        pct = st.slider("Peak productivity gain at age 30 (%)", -50, 200, 15) / 100.0
        width = st.slider("Age window half-width", 2, 20, 8)  # shape of the bump
        alpha = st.slider("Consumption pass-through (α)", 0.0, 1.0, 0.5, 0.05)
    if scenario.startswith("Older‑age quality"):
        mode = st.selectbox("Improve", ["Survival (65+)", "QALY weight (65+)"])
        if mode == "Survival (65+)":
            surv_pct = st.slider("Increase 1‑year survival (relative, %)", 0, 50, 10) / 100.0
        else:
            qaly_pct = st.slider("QALY weight uplift for 65+ (%)", 0, 50, 10) / 100.0

# ---------- Load data ----------
if not (life_file and nta_file and wage_file and pop_file):
    st.info("Upload the four files to proceed (life table, NTA, wages, population).")
    st.stop()

# Life table (expects Age in first col, l_m at col 4, l_f at col 10 in the typical ONS layout).
life_raw = pd.read_excel(life_file, sheet_name=0, header=None)
# Try to auto-detect a 101-row age block; fallback to first 101 rows
guess = life_raw.iloc[7:108, :12].to_numpy()
nlt_age = guess[:, 0].astype(int)
l_m = guess[:, 3]
l_f = guess[:, 9]
S_male   = np.concatenate([l_m/100000.0, np.zeros(10)])  # pad like your script
S_female = np.concatenate([l_f/100000.0, np.zeros(10)])

# NTA
Q = pd.read_excel(nta_file)
# Use the column names you showed earlier
y       = Q.loc[Q['Type'].eq('Smooth Mean'), 'Labor Income'].to_numpy()
c       = Q.loc[Q['Type'].eq('Smooth Mean'), 'Consumption'].to_numpy()

ages_nta = Q.loc[Q['Type'].eq('NTA'), 'Age'].to_numpy()

# Wages (ASHE). Try xls first; if not, read as xlsx
wage_tbl = pd.read_excel(wage_file, sheet_name=0, header=None)
# Pull two columns by position (median in col 2, mean in col 4 in original script)
w = np.zeros(111)
# Fill by bins like your MATLAB
w[:17]   = wage_tbl.iloc[1, 3]
w[17:21] = wage_tbl.iloc[2, 3]
w[21:29] = wage_tbl.iloc[3, 3]
w[29:39] = wage_tbl.iloc[4, 3]
w[39:49] = wage_tbl.iloc[5, 3]
w[49:59] = wage_tbl.iloc[6, 3]
w[59:   ]= wage_tbl.iloc[7, 3]

# Population by age
popdf = pd.read_csv(pop_file)
popdf = popdf.sort_values('Age')
N_age = popdf['Population'].to_numpy()
ages_pop = popdf['Age'].to_numpy()

# Align everything to a common age grid (use NTA ages)
A = ages_nta.astype(int)
# Interpolate S and w to the NTA age grid if needed
S_m_interp = np.interp(A, np.arange(len(S_male)), S_male, left=S_male[0], right=0.0)
S_f_interp = np.interp(A, np.arange(len(S_female)), S_female, left=S_female[0], right=0.0)
w_interp   = np.interp(A, np.arange(len(w)), w, left=w[0], right=w[-1])
N_interp   = np.interp(A, ages_pop, N_age, left=N_age[0], right=0.0)

# ---------- Baseline ----------
base = compute_value_functions(
    ages=A, y=y, c=c, w=w_interp,
    S_male=S_m_interp, S_female=S_f_interp,
    hours_endowment=hours_endowment, sigma=sigma, eta=eta,
    z0_z=z0z, beta=beta, r_rho=rrho, hp_lambda=hp_lambda
)

TEV_flow_base = aggregate_values(N_interp, base['v'], base['V_m'], which='flow')
TEV_stock_base = aggregate_values(N_interp, base['v'], base['V_m'], which='stock')

# ---------- Scenario ----------
Ages = base['ages']
v_new = base['v'].copy()
V_new = base['V_m'].copy()
S_new = S_m_interp.copy()

if scenario == "Fertility":
    # Stock effect now: ΔTEV_stock = ΔBirths × V(0)
    delta_stock = births_delta * (V_new[0] if len(V_new) > 0 else 0.0)
    TEV_flow_scn = TEV_flow_base  # no immediate flow change absent dynamics
    TEV_stock_scn = TEV_stock_base + delta_stock
    expl = f"ΔTEV_stock = ΔBirths × V(0) = {births_delta:,} × £{V_new[0]:,.0f}"

elif scenario.startswith("Productivity"):
    # Boost y and (optionally) c for selected ages
    mask = (Ages >= amin) & (Ages <= amax)
    y_scn = y.copy(); c_scn = c.copy()
    y_scn[mask] = y_scn[mask] * (1 + pct)
    c_scn[mask] = c_scn[mask] * (1 + alpha * pct)

    scn = compute_value_functions(
        ages=Ages, y=y_scn, c=c_scn, w=w_interp,
        S_male=S_m_interp, S_female=S_f_interp,
        hours_endowment=hours_endowment, sigma=sigma, eta=eta,
        z0_z=z0z, beta=beta, r_rho=rrho, hp_lambda=hp_lambda
    )
    v_new, V_new = scn['v'], scn['V_m']
    TEV_flow_scn  = aggregate_values(N_interp, v_new, V_new, 'flow')
    TEV_stock_scn = aggregate_values(N_interp, v_new, V_new, 'stock')
    expl = f"Productivity +{pct*100:.0f}% on ages {amin}-{amax}, consumption pass-through α={alpha:.2f}"

elif scenario.startswith("Human capital"):
    # Triangular productivity bump centered at 30
    peak_age = 30
    bump = np.maximum(0, 1 - np.abs(Ages - peak_age)/width) * pct
    y_scn = y * (1 + bump)
    c_scn = c * (1 + alpha * bump)
    scn = compute_value_functions(
        ages=Ages, y=y_scn, c=c_scn, w=w_interp,
        S_male=S_m_interp, S_female=S_f_interp,
        hours_endowment=hours_endowment, sigma=sigma, eta=eta,
        z0_z=z0z, beta=beta, r_rho=rrho, hp_lambda=hp_lambda
    )
    v_new, V_new = scn['v'], scn['V_m']
    TEV_flow_scn  = aggregate_values(N_interp, v_new, V_new, 'flow')
    TEV_stock_scn = aggregate_values(N_interp, v_new, V_new, 'stock')
    expl = f"Human capital peak +{pct*100:.0f}% at 30 (width {width}), α={alpha:.2f}"

elif scenario.startswith("Older‑age quality"):
    if mode == "Survival (65+)":
        p = survival_to_one_year_probs(S_m_interp)
        mask = Ages >= 65
        p2 = p.copy()
        p2[mask] = np.minimum(1.0, p2[mask]*(1 + surv_pct))
        S_new = probs_to_survival(p2)
        scn = compute_value_functions(
            ages=Ages, y=y, c=c, w=w_interp,
            S_male=S_new, S_female=S_f_interp,
            hours_endowment=hours_endowment, sigma=sigma, eta=eta,
            z0_z=z0z, beta=beta, r_rho=rrho, hp_lambda=hp_lambda
        )
        v_new, V_new = scn['v'], scn['V_m']
        TEV_flow_scn  = aggregate_values(N_interp, v_new, V_new, 'flow')
        TEV_stock_scn = aggregate_values(N_interp, v_new, V_new, 'stock')
        expl = f"Increase 1‑year survival by {surv_pct*100:.0f}% for ages 65+"
    else:
        q = np.ones_like(Ages, float)
        q[Ages >= 65] *= (1 + qaly_pct)
        v_new = base['v'] * q
        # V needs to reflect q weights over future ages; approximate by scaling current V by same factor at age
        # (quick, conservative). For precision, re-sum with q weights across future ages.
        V_new = base['V_m'].copy()
        TEV_flow_scn  = aggregate_values(N_interp, v_new, V_new, 'flow')
        TEV_stock_scn = aggregate_values(N_interp, v_new, V_new, 'stock')
        expl = f"QALY weight +{qaly_pct*100:.0f}% for ages 65+ (flow scaled)"

else:  # Combined (example: user could combine above—left as an extension)
    v_new, V_new = base['v'], base['V_m']
    TEV_flow_scn, TEV_stock_scn = TEV_flow_base, TEV_stock_base
    expl = "No combined policy implemented in this minimal demo."

# ---------- Results ----------
col1, col2 = st.columns(2)
with col1:
    st.subheader("Baseline")
    st.metric("TEV_flow (this year)", f"£{TEV_flow_base:,.0f}")
    st.metric("TEV_stock (living population)", f"£{TEV_stock_base:,.0f}")
with col2:
    st.subheader("Scenario")
    st.metric("TEV_flow (this year)", f"£{TEV_flow_scn:,.0f}", f"{(TEV_flow_scn-TEV_flow_base)/max(1,TEV_flow_base)*100:.2f}%")
    st.metric("TEV_stock (living population)", f"£{TEV_stock_scn:,.0f}", f"{(TEV_stock_scn-TEV_stock_base)/max(1,TEV_stock_base)*100:.2f}%")
st.caption(f"Scenario: {expl}")

# Plots: v(a) and V(a)
fig, ax = plt.subplots(1, 2, figsize=(12,4))
ax[0].plot(Ages, base['v'], label='Baseline'); ax[0].plot(Ages, v_new, label='Scenario')
ax[0].set_title("Value of a life-year v(a)"); ax[0].set_xlabel("Age"); ax[0].set_ylabel("£")
ax[0].legend()
ax[1].plot(Ages, base['V_m'], label='Baseline'); ax[1].plot(Ages, V_new, label='Scenario')
ax[1].set_title("Value of remaining life V(a)"); ax[1].set_xlabel("Age"); ax[1].set_ylabel("£")
ax[1].legend()
st.pyplot(fig)

# Age contributions to ΔTEV_stock
delta_per_age = (V_new - base['V_m']) * np.asarray(N_interp)
fig2, ax2 = plt.subplots(figsize=(10,4))
ax2.bar(Ages, delta_per_age, width=0.9)
ax2.set_title("Contribution by age to ΔTEV_stock")
ax2.set_xlabel("Age"); ax2.set_ylabel("£")
st.pyplot(fig2)

# Fertility note
if scenario == "Fertility":
    st.info(f"For fertility, the immediate stock gain is **ΔBirths × V(0)**. "
            f"To see annual flows over time, add a simple cohort simulation: Nₜ₊₁(a+1)=Nₜ(a)·p(a).")
