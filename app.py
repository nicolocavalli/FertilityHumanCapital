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
WAGE_XLS  = ROOT / "Age Group Table 6.5a  Hourly pay - Gross 2012.xls"  # set to your exact filename
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
    return float(np.nansum(N[:T]()
