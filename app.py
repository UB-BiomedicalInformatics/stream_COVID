# Modified version for Erie County, New York
# Contact: ganaya@buffalo.edu

from functools import reduce
from typing import Generator, Tuple, Dict, Any, Optional
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib
from bs4 import BeautifulSoup
import requests
import ipyvuetify as v
from traitlets import Unicode, List
from datetime import date, datetime, timedelta
import time
import altair as alt
from collections import namedtuple

matplotlib.use("Agg")
import matplotlib.pyplot as plt

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)


# Models and base functions
########
def sir(
    s: float, i: float, r: float, beta: float, gamma: float, n: float
    ) -> Tuple[float, float, float]:
    """The SIR model, one time step."""
    s_n = (-beta * s * i) + s
    i_n = (beta * s * i - gamma * i) + i
    r_n = gamma * i + r
    if s_n < 0.0:
        s_n = 0.0
    if i_n < 0.0:
        i_n = 0.0
    if r_n < 0.0:
        r_n = 0.0

    scale = n / (s_n + i_n + r_n)
    return s_n * scale, i_n * scale, r_n * scale
    
def gen_sir(
    s: float, i: float, r: float, beta: float, gamma: float, n_days: int
    ) -> Generator[Tuple[float, float, float], None, None]:
    """Simulate SIR model forward in time yielding tuples."""
    s, i, r = (float(v) for v in (s, i, r))
    n = s + i + r
    for _ in range(n_days + 1):
        yield s, i, r
        s, i, r = sir(s, i, r, beta, gamma, n)

def sim_sir(
    s: float, i: float, r: float, beta: float, gamma: float, n_days: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate the SIR model forward in time."""
    s, i, r = (float(v) for v in (s, i, r))
    n = s + i + r
    s_v, i_v, r_v = [s], [i], [r]
    for day in range(n_days):
        s, i, r = sir(s, i, r, beta, gamma, n)
        s_v.append(s)
        i_v.append(i)
        r_v.append(r)

    return (
        np.array(s_v),
        np.array(i_v),
        np.array(r_v),
    )
    
def sim_sir_df(
    p) -> pd.DataFrame:
    """Simulate the SIR model forward in time.

    p is a Parameters instance. for circuluar dependency reasons i can't annotate it.
    """
    return pd.DataFrame(
        data=gen_sir(S, total_infections, recovered, beta, gamma, n_days),
        columns=("Susceptible", "Infected", "Recovered"),
    )

def get_dispositions(
    patient_state: np.ndarray, rates: Tuple[float, ...], regional_hosp_share: float = 1.0
    ) -> Tuple[np.ndarray, ...]:
    """Get dispositions of infected adjusted by rate and market_share."""
    return (*(patient_state * rate * regional_hosp_share for rate in rates),)

def build_admissions_df(
    dispositions) -> pd.DataFrame:
    """Build admissions dataframe from Parameters."""
    days = np.array(range(0, n_days + 1))
    data_dict = dict(
        zip(
            ["day", "hosp", "icu", "vent"], 
            [days] + [disposition for disposition in dispositions],
        )
    )
    projection = pd.DataFrame.from_dict(data_dict)
    
    # New cases
    projection_admits = projection.iloc[:-1, :] - projection.shift(1)
    projection_admits["day"] = range(projection_admits.shape[0])
    return projection_admits

def build_census_df(
    projection_admits: pd.DataFrame) -> pd.DataFrame:
    """ALOS for each category of COVID-19 case (total guesses)"""
    #n_days = np.shape(projection_admits)[0]
    los_dict = {
    "hosp": hosp_los, "icu": icu_los, "vent": vent_los,
    }

    census_dict = dict()
    for k, los in los_dict.items():
        census = (
            projection_admits.cumsum().iloc[:-los, :]
            - projection_admits.cumsum().shift(los).fillna(0)
        ).apply(np.ceil)
        census_dict[k] = census[k]

    census_df = pd.DataFrame(census_dict)
    census_df["day"] = census_df.index
    census_df = census_df[["day", "hosp", "icu", "vent"
    ]]
    
    census_df['total_county_icu'] = icu_county
    census_df['total_county_beds'] = beds_county
    census_df['expanded_icu_county'] = expanded_icu_county_05
    census_df['expanded_beds_county'] = expanded_beds_county_05
    census_df['expanded_icu_county2'] = expanded_icu_county_1
    census_df['expanded_beds_county2'] = expanded_beds_county_1
    
    # PPE for hosp/icu
    census_df['ppe_mild_d'] = census_df['hosp'] * ppe_mild_val_lower
    census_df['ppe_mild_u'] = census_df['hosp'] * ppe_mild_val_upper
    census_df['ppe_severe_d'] = census_df['icu'] * ppe_severe_val_lower
    census_df['ppe_severe_u'] = census_df['icu'] * ppe_severe_val_upper
    census_df['ppe_mean_mild'] = census_df[["ppe_mild_d","ppe_mild_u"]].mean(axis=1)
    census_df['ppe_mean_severe'] = census_df[["ppe_severe_d","ppe_severe_u"]].mean(axis=1)
    
    census_df = census_df.head(n_days-10)
    
    return census_df

def seir(
    s: float, e: float, i: float, r: float, beta: float, gamma: float, alpha: float, n: float
    ) -> Tuple[float, float, float, float]:
    """The SIR model, one time step."""
    s_n = (-beta * s * i) + s
    e_n = (beta * s * i) - alpha * e + e
    i_n = (alpha * e - gamma * i) + i
    r_n = gamma * i + r
    if s_n < 0.0:
        s_n = 0.0
    if e_n < 0.0:
        e_n = 0.0
    if i_n < 0.0:
        i_n = 0.0
    if r_n < 0.0:
        r_n = 0.0

    scale = n / (s_n + e_n+ i_n + r_n)
    return s_n * scale, e_n * scale, i_n * scale, r_n * scale

def sim_seir(
    s: float, e:float, i: float, r: float, beta: float, gamma: float, alpha: float, n_days: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate the SIR model forward in time."""
    s, e, i, r = (float(v) for v in (s, e, i, r))
    n = s + e + i + r
    s_v, e_v, i_v, r_v = [s], [e], [i], [r]
    for day in range(n_days):
        s, e, i, r = seir(s, e, i, r, beta, gamma, alpha, n)
        s_v.append(s)
        e_v.append(e)
        i_v.append(i)
        r_v.append(r)

    return (
        np.array(s_v),
        np.array(e_v),
        np.array(i_v),
        np.array(r_v),
    )

def gen_seir(
    s: float, e: float, i: float, r: float, beta: float, gamma: float, alpha: float, n_days: int
    ) -> Generator[Tuple[float, float, float, float], None, None]:
    """Simulate SIR model forward in time yielding tuples."""
    s, e, i, r = (float(v) for v in (s, e, i, r))
    n = s + e + i + r
    for _ in range(n_days + 1):
        yield s, e, i, r
        s, e, i, r = seir(s, e, i, r, beta, gamma, alpha, n)
# phase-adjusted https://www.nature.com/articles/s41421-020-0148-0     
   
def sim_seir_decay(
    s: float, e:float, i: float, r: float, beta: float, gamma: float, alpha: float, n_days: int,
    decay1:float, decay2:float, decay3: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate the SIR model forward in time."""
    s, e, i, r = (float(v) for v in (s, e, i, r))
    n = s + e + i + r
    s_v, e_v, i_v, r_v = [s], [e], [i], [r]
    for day in range(n_days):
        if 0<=day<=21:
            beta_decay=beta*(1-decay1)
        elif 22<=day<=28:
            beta_decay=beta*(1-decay2)
        else: 
            beta_decay=beta*(1-decay3)
        s, e, i, r = seir(s, e, i, r, beta_decay, gamma, alpha, n)
        s_v.append(s)
        e_v.append(e)
        i_v.append(i)
        r_v.append(r)

    return (
        np.array(s_v),
        np.array(e_v),
        np.array(i_v),
        np.array(r_v),
    )
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4552173/

def seird(
    s: float, e: float, i: float, r: float, d: float, beta: float, gamma: float, alpha: float, n: float, fatal: float
    ) -> Tuple[float, float, float, float]:
    """The SIR model, one time step."""
    s_n = (-beta * s * i) + s
    e_n = (beta * s * i) - alpha * e + e
    i_n = (alpha * e - gamma * i) + i
    r_n = (1-fatal)*gamma * i + r
    d_n = (fatal)*gamma * i +d
    if s_n < 0.0:
        s_n = 0.0
    if e_n < 0.0:
        e_n = 0.0
    if i_n < 0.0:
        i_n = 0.0
    if r_n < 0.0:
        r_n = 0.0
    if d_n < 0.0:
        d_n = 0.0

    scale = n / (s_n + e_n+ i_n + r_n + d_n)
    return s_n * scale, e_n * scale, i_n * scale, r_n * scale, d_n * scale

def sim_seird_decay(
    s: float, e:float, i: float, r: float, d: float, beta: float, gamma: float, alpha: float, n_days: int,
    decay1:float, decay2:float, decay3: float, fatal: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate the SIR model forward in time."""
    s, e, i, r, d= (float(v) for v in (s, e, i, r, d))
    n = s + e + i + r + d
    s_v, e_v, i_v, r_v, d_v = [s], [e], [i], [r], [d]
    for day in range(n_days):
        if 0<=day<=21:
            beta_decay=beta*(1-decay1)
        elif 22<=day<=28:
            beta_decay=beta*(1-decay2)
        else: 
            beta_decay=beta*(1-decay3)
        s, e, i, r,d = seird(s, e, i, r, d, beta_decay, gamma, alpha, n, fatal)
        s_v.append(s)
        e_v.append(e)
        i_v.append(i)
        r_v.append(r)
        d_v.append(d)

    return (
        np.array(s_v),
        np.array(e_v),
        np.array(i_v),
        np.array(r_v),
        np.array(d_v)
    )

def seijcrd(
    s: float, e: float, i: float, j:float, c:float, r: float, d: float, beta: float, gamma: float, alpha: float, n: float, fatal_hosp: float, hosp_rate:float, icu_rate:float, icu_days:float,crit_lag:float, death_days:float
    ) -> Tuple[float, float, float, float]:
    """The SIR model, one time step."""
    s_n = (-beta * s * (i+j+c)) + s
    e_n = (beta * s * (i+j+c)) - alpha * e + e
    i_n = (alpha * e - gamma * i) + i
    j_n = hosp_rate * i * gamma + (1-icu_rate)* c *icu_days + j
    c_n = icu_rate * j * (1/crit_lag) - c *  (1/death_days)
    r_n = (1-hosp_rate)*gamma * i + (1-icu_rate) * (1/crit_lag)* j + r
    d_n = (fatal_hosp)* c * (1/crit_lag)+d
    if s_n < 0.0:
        s_n = 0.0
    if e_n < 0.0:
        e_n = 0.0
    if i_n < 0.0:
        i_n = 0.0
    if j_n < 0.0:
        j_n = 0.0
    if c_n < 0.0:
        c_n = 0.0
    if r_n < 0.0:
        r_n = 0.0
    if d_n < 0.0:
        d_n = 0.0

    scale = n / (s_n + e_n+ i_n + j_n+ c_n+ r_n + d_n)
    return s_n * scale, e_n * scale, i_n * scale, j_n* scale, c_n*scale, r_n * scale, d_n * scale

def sim_seijcrd_decay(
    s: float, e:float, i: float, j:float, c: float, r: float, d: float, beta: float, gamma: float, alpha: float, n_days: int,
    decay1:float, decay2:float, decay3: float, fatal_hosp: float, hosp_rate: float, icu_rate: float, icu_days:float, crit_lag: float, death_days:float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate the SIR model forward in time."""
    s, e, i, j, c, r, d= (float(v) for v in (s, e, i, c, j, r, d))
    n = s + e + i + j+r + d
    s_v, e_v, i_v, j_v, c_v, r_v, d_v = [s], [e], [i], [j], [c], [r], [d]
    for day in range(n_days):
        if 0<=day<=21:
            beta_decay=beta*(1-decay1)
        elif 22<=day<=28:
            beta_decay=beta*(1-decay2)
        else: 
            beta_decay=beta*(1-decay3)
        s, e, i,j, c, r,d = seijcrd(s, e, i,j, c, r, d, beta_decay, gamma, alpha, n, fatal_hosp, hosp_rate, icu_rate, icu_days, crit_lag, death_days)
        s_v.append(s)
        e_v.append(e)
        i_v.append(i)
        j_v.append(j)
        c_v.append(c)
        r_v.append(r)
        d_v.append(d)

    return (
        np.array(s_v),
        np.array(e_v),
        np.array(i_v),
        np.array(j_v),
        np.array(c_v),
        np.array(r_v),
        np.array(d_v)
    )

### Less complicated
##
##def seijcrd2(
##    s: float, e: float, i: float, j:float, r: float, d: float, beta: float, gamma: float, alpha: float, n: float, fatal: float, fatal_hosp: float, hosp_rate:float,
##    hosp_day_rate:float, l:float
##    ) -> Tuple[float, float, float, float, float,float]:
##    """The SIR model, one time step."""
##    s_n = -beta*s*(i + (l*j)) +s
##    e_n = beta*s*(i + (l*j)) - alpha * e + e
##    i_n = alpha * e - (hosp_rate + gamma) * i + i 
##    j_n = hosp_rate * i - hosp_day_rate*j +j
##    r_n = gamma * (1-fatal)*i + ((1-fatal_hosp) * hosp_day_rate * j) +r
##    d_n = gamma * (fatal)*i + ((fatal_hosp)*hosp_day_rate*j) +d
##    if s_n < 0.0:
##        s_n = 0.0
##    if e_n < 0.0:
##        e_n = 0.0
##    if i_n < 0.0:
##        i_n = 0.0
##    if j_n < 0.0:
##        j_n = 0.0
##    if r_n < 0.0:
##        r_n = 0.0
##    if d_n < 0.0:
##        d_n = 0.0
##
##    scale = n / (s_n + e_n+ i_n + j_n+ r_n + d_n)
##    return s_n * scale, e_n * scale, i_n * scale, j_n* scale, r_n * scale, d_n * scale
### 
##
##def sim_seijcrd_decay2(
##    s: float, e:float, i: float, j:float, r: float, d: float, beta: float, gamma: float, alpha: float, n_days: int,
##    decay2:float, decay3: float, fatal: float, fatal_hosp: float, hosp_rate: float, hosp_day_rate:float, l:float
##    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,np.ndarray, np.ndarray]:
##    """Simulate the SIR model forward in time."""
##    s, e, i, j, r, d= (float(v) for v in (s, e, i, j, r, d))
##    n = s + e + i + j+r + d
##    s_v, e_v, i_v, j_v, r_v, d_v = [s], [e], [i], [j], [r], [d]
##    for day in range(n_days):
##        if 0<=day<=21:
##            beta_decay=beta
##        elif 22<=day<=28:
##            beta_decay=beta*(1-decay2)
##        else: 
##            beta_decay=beta*(1-decay3)
##        s, e, i,j, r,d = seijcrd2(s, e, i,j, r, d, beta_decay, gamma, alpha, n, fatal, fatal_hosp, hosp_rate,hosp_day_rate, l)
##        s_v.append(s)
##        e_v.append(e)
##        i_v.append(i)
##        j_v.append(j)
##        r_v.append(r)
##        d_v.append(d)
##
##    return (
##        np.array(s_v),
##        np.array(e_v),
##        np.array(i_v),
##        np.array(j_v),
##        np.array(r_v),
##        np.array(d_v)
##    )
## 

# End Models # 

# Add dates #
def add_date_column(
    df: pd.DataFrame, drop_day_column: bool = False, date_format: Optional[str] = None,
    ) -> pd.DataFrame:
    """Copies input data frame and converts "day" column to "date" column

    Assumes that day=0 is today and allocates dates for each integer day.
    Day range can must not be continous.
    Columns will be organized as original frame with difference that date
    columns come first.

    Arguments:
        df: The data frame to convert.
        drop_day_column: If true, the returned data frame will not have a day column.
        date_format: If given, converts date_time objetcts to string format specified.

    Raises:
        KeyError: if "day" column not in df
        ValueError: if "day" column is not of type int
    """
    if not "day" in df:
        raise KeyError("Input data frame for converting dates has no 'day column'.")
    if not pd.api.types.is_integer_dtype(df.day):
        raise KeyError("Column 'day' for dates converting data frame is not integer.")

    df = df.copy()
    # Prepare columns for sorting
    non_date_columns = [col for col in df.columns if not col == "day"]

    # Allocate (day) continous range for dates
    n_days = int(df.day.max())
    start = datetime(2020, 3, 1)
    end = start + timedelta(days=n_days + 1)
    # And pick dates present in frame
    dates = pd.date_range(start=start, end=end, freq="D")[df.day.tolist()]

    if date_format is not None:
        dates = dates.strftime(date_format)

    df["date"] = dates

    if drop_day_column:
        df.pop("day")
        date_columns = ["date"]
    else:
        date_columns = ["day", "date"]

    # sort columns
    df = df[date_columns + non_date_columns]

    return df

    

# Adding ICU bed for Erie county
# 3/25/20
icu_county = 468
beds_county = 2762
# Bed expansion at 50%
expanded_icu_county_05 = 369
expanded_beds_county_05 = 3570
# Bed expansion at 100%
expanded_icu_county_1 = 492
expanded_beds_county_1 = 4760

# PPE Values
ppe_mild_val_lower = 14
ppe_mild_val_upper = 15
ppe_severe_val_lower = 15
ppe_severe_val_upper = 24

#
groups = ['hosp', 'icu', 'vent']

# Load erie county data
url = "https://raw.githubusercontent.com/UB-BiomedicalInformatics/stream_COVID/master/Cases_Erie.csv"
erie_df = pd.read_csv(url)
erie_df['Date'] = pd.to_datetime(erie_df['Date'])

# Populations and Infections
erie = 1500000
cases_erie = 1235
S_default = erie
known_infections = 1235
known_cases = 221
regional_hosp_share = 1.0
S = erie


# Widgets
model_options = st.sidebar.radio(
    "Service", ('Inpatient', 'ICU', 'Ventilated'))

current_hosp = st.sidebar.number_input(
    "Total Hospitalized Cases", value=known_cases, step=1, format="%i")

doubling_time = st.sidebar.number_input(
    "Doubling Time (days)", value=3.0, step=1.0, format="%f")

relative_contact_rate = st.sidebar.number_input(
    "Social distancing (% reduction in social contact) Unadjusted Model", 0, 100, value=0, step=5, format="%i")/100.0

decay1 = st.sidebar.number_input(
    "Social distancing (% reduction in social contact) in Week 0-2", 0, 100, value=0, step=5, format="%i")/100.0

decay2 = st.sidebar.number_input(
    "Social distancing (% reduction in social contact) in Week 3", 0, 100, value=15, step=5, format="%i")/100.0

decay3 = st.sidebar.number_input(
    "Social distancing (% reduction in social contact) after Week 3", 0, 100, value=30 ,step=5, format="%i")/100.0

hosp_rate = (
    st.sidebar.number_input("Hospitalization %", 0.0, 100.0, value=2.5, step=0.50, format="%f")/ 100.0)

icu_rate = (
    st.sidebar.number_input("ICU %", 0.0, 100.0, value=1.25, step=0.25, format="%f") / 100.0)

vent_rate = (
    st.sidebar.number_input("Ventilated %", 0.0, 100.0, value=1.0, step=0.25, format="%f")/ 100.0)

incubation_period =(
    st.sidebar.number_input("Incubation Period", 0.0, 12.0, value=5.2, step=0.1, format="%f"))

recovery_days =(
    st.sidebar.number_input("Recovery Period", 0.0, 21.0, value=11.0 ,step=0.1, format="%f"))

infectious_period =(
    st.sidebar.number_input("Infectious Period", 0.0, 18.0, value=3.0,step=0.1, format="%f"))

fatal = st.sidebar.number_input(
    "Overall Fatality (%)", 0.0, 100.0, value=0.5 ,step=0.1, format="%f")/100.0

fatal_hosp = st.sidebar.number_input(
    "Hospital Fatality (%)", 0.0, 100.0, value=4.0 ,step=0.1, format="%f")/100.0

death_days = st.sidebar.number_input(
    "Days person remains in critical care or dies", 0, 20, value=4 ,step=1, format="%f")

crit_lag = st.sidebar.number_input(
    "Days person takes to go to critical care", 0, 20, value=4 ,step=1, format="%f")

hosp_los = st.sidebar.number_input("Hospital Length of Stay", value=10, step=1, format="%i")
icu_los = st.sidebar.number_input("ICU Length of Stay", value=11, step=1, format="%i")
vent_los = st.sidebar.number_input("Ventilator Length of Stay", value=6, step=1, format="%i")

regional_hosp_share = (
   st.sidebar.number_input(
       "Hospital Bed Share (%)", 0.0, 100.0, value=100.0, step=1.0, format="%f")
   / 100.0
)

S = st.sidebar.number_input(
  "Regional Population", value=S_default, step=100000, format="%i")

initial_infections = st.sidebar.number_input(
    "Currently Known Regional Infections (only used to compute detection rate - does not change projections)", value=known_infections, step=10, format="%i")

total_infections = current_hosp / regional_hosp_share / hosp_rate
detection_prob = initial_infections / total_infections

intrinsic_growth_rate = 2 ** (1 / doubling_time) - 1
# (0.12 + 0.07)/

recovered = 0.0

# mean recovery rate, gamma, (in 1/days).
gamma = 1 / recovery_days

# Contact rate, beta
beta = (
    intrinsic_growth_rate + gamma
) / S * (1-relative_contact_rate) # {rate based on doubling time} / {initial S}

r_t = beta / gamma * S # r_t is r_0 after distancing
r_naught = (intrinsic_growth_rate + gamma) / gamma
doubling_time_t = 1/np.log2(beta*S - gamma +1) # doubling time after distancing

# Contact rate,  beta for SEIR
beta2 = (
    intrinsic_growth_rate + (1/infectious_period)
) / S * (1-relative_contact_rate)
alpha = 1/incubation_period

# Contact rate,  beta for SEIR with phase adjusted R0
beta3 = (
(alpha+intrinsic_growth_rate)*(intrinsic_growth_rate + (1/infectious_period))
) / (alpha*S) *(1-relative_contact_rate)

## converting beta to intrinsic growth rate calculation
# https://www.sciencedirect.com/science/article/pii/S2468042719300491
beta4 = (
    (alpha+intrinsic_growth_rate)*(intrinsic_growth_rate + (1/infectious_period))
) / (alpha*S) 


# for SEIJRD
gamma_hosp = 1 / hosp_los
icu_days = 1 / icu_los

###################### Main Tittle ###################
st.title("COVID-19 Disease Model - Erie County, NY")

###################### First Graph ###################
# Erie cases Graph
erie_cases_bar = alt.Chart(erie_df).mark_bar(color='lightgray').encode(
    x='Date:T',
    y='Cases:Q',
    tooltip=[alt.Tooltip("Cases:Q", format=".0f", title="Cases")])
erie_admit_line = alt.Chart(erie_df).mark_line(color='red', point=True).encode(
    x='Date:T',
    y='Admissions:Q')
erie_icu_line = alt.Chart(erie_df).mark_line(color='orange', point=True).encode(
    x='Date:T',
    y='ICU:Q')

# Bar graph - Removing this one to show new graph below. 
#st.altair_chart(alt.layer(erie_cases_bar + erie_admit_line + erie_icu_line), use_container_width=True)

# Slider and Date
n_days = st.slider("Number of days to project", 30, 200, 120, 1, "%i")
as_date = st.checkbox(label="Present result as dates", value=False)


st.header("""Erie County: Reported Cases, Census and Admissions""")
# Erie Graph of Cases # Lines of cases
def erie_chart(
    projection_admits: pd.DataFrame) -> alt.Chart:
    """docstring"""
    
    projection_admits = projection_admits.rename(columns={"Admissions": "Census Inpatient", 
                                                            "ICU":"Census Intensive", 
                                                            "Ventilated":"Census Ventilated",
                                                            "New_admits":"New Admissions",
                                                            "New_discharge":"New Discharges",
                                                            })
    
    return(
        alt
        .Chart(projection_admits)
        .transform_fold(fold=["Census Inpatient", 
                                "Census Intensive", 
                                "Census Ventilated",
                                "New Admissions",
                                "New Discharges"
                                ])
        .mark_line(strokeWidth=3, point=True)
        .encode(
            x=alt.X("Date", title="Date"),
            y=alt.Y("value:Q", title="Erie County Census"),
            color="key:N",
            tooltip=[alt.Tooltip("value:Q", format=".0f"),"key:N"]
        )
        .interactive()
    )

  
#Erie Graph of Cases # Lines of cases # Inpatient Census

if as_date:
    erie_df = add_date_column(erie_df)
    day_date = 'date:T'
    def erie_inpatient(projection_admits: pd.DataFrame) -> alt.Chart:
        """docstring"""
    
        projection_admits = projection_admits.rename(columns={"Admissions": "Erie County Inpatient"})
    
        return(
            alt
            .Chart(projection_admits)
            .transform_fold(fold=["Erie County Inpatient"])
            .mark_line(strokeWidth=3, strokeDash=[2,3], point=True)
            .encode(
                x=alt.X(day_date),
                y=alt.Y("value:Q", title="Census"),
                color="key:N",
                tooltip=[alt.Tooltip("value:Q", format=".0f"),"key:N"]
            )
            .interactive()
        )
else:
    day_date = 'day'
    def erie_inpatient(
        projection_admits: pd.DataFrame) -> alt.Chart:
        """docstring"""
    
        projection_admits = projection_admits.rename(columns={"Admissions": "Erie County Inpatient"})
    
        return(
            alt
            .Chart(projection_admits)
            .transform_fold(fold=["Erie County Inpatient"])
            .mark_line(strokeWidth=3, strokeDash=[2,3], point=True)
            .encode(
                x=alt.X(day_date),
                y=alt.Y("value:Q", title="Census"),
                color="key:N",
                tooltip=[alt.Tooltip("value:Q", format=".0f"),"key:N"]
            )
            .interactive()
        )

# Erie Graph of Cases # Lines of cases # ICU Census
if as_date:
    erie_df = add_date_column(erie_df)
    day_date = 'date:T'
    def erie_icu(projection_admits: pd.DataFrame) -> alt.Chart:
        """docstring"""
        
        projection_admits = projection_admits.rename(columns={"ICU": "Erie County ICU"})
        
        return(
            alt
            .Chart(projection_admits)
            .transform_fold(fold=["Erie County ICU"])
            .mark_line(strokeWidth=3, strokeDash=[2,3], point=True)
            .encode(
                x=alt.X(day_date),
                y=alt.Y("value:Q", title="Census"),
                color="key:N",
                tooltip=[alt.Tooltip("value:Q", format=".0f"),"key:N"]
            )
            .interactive()
        )
else:
    day_date = 'day'
    def erie_icu(projection_admits: pd.DataFrame) -> alt.Chart:
        """docstring"""
        
        projection_admits = projection_admits.rename(columns={"ICU": "Erie County ICU"})
        
        return(
            alt
            .Chart(projection_admits)
            .transform_fold(fold=["Erie County ICU"])
            .mark_line(strokeWidth=3, strokeDash=[2,3], point=True)
            .encode(
                x=alt.X(day_date),
                y=alt.Y("value:Q", title="Census"),
                color="key:N",
                tooltip=[alt.Tooltip("value:Q", format=".0f"),"key:N"]
            )
            .interactive()
        )
    
# Erie Graph of Cases # Lines of cases # Ventilator Census
if as_date:
    erie_df = add_date_column(erie_df)
    day_date = 'date:T'
    def erie_vent(projection_admits: pd.DataFrame) -> alt.Chart:
        """docstring"""
        
        projection_admits = projection_admits.rename(columns={"Ventilated": "Erie County Ventilated"})
          
        return(
            alt
            .Chart(projection_admits)
            .transform_fold(fold=["Erie County Ventilated"])
            .mark_line(strokeWidth=3, strokeDash=[2,3], point=True)
            .encode(
                x=alt.X(day_date),
                y=alt.Y("value:Q", title="Census"),
                color="key:N",
                tooltip=[alt.Tooltip("value:Q", format=".0f"),"key:N"]
            )
            .interactive()
        )
else:
    day_date = 'day'
    def erie_vent(projection_admits: pd.DataFrame) -> alt.Chart:
        """docstring"""
        
        projection_admits = projection_admits.rename(columns={"Ventilated": "Erie County Ventilated"})
          
        return(
            alt
            .Chart(projection_admits)
            .transform_fold(fold=["Erie County Ventilated"])
            .mark_line(strokeWidth=3, strokeDash=[2,3], point=True)
            .encode(
                x=alt.X(day_date),
                y=alt.Y("value:Q", title="Census"),
                color="key:N",
                tooltip=[alt.Tooltip("value:Q", format=".0f"),"key:N"]
            )
            .interactive()
        )


erie_lines = erie_chart(erie_df)
erie_lines_ip = erie_inpatient(erie_df)
erie_lines_icu = erie_icu(erie_df)
erie_lines_vent = erie_vent(erie_df)

# Bar chart of Erie cases with layer of HERDS DAta Erie
st.altair_chart(erie_cases_bar + erie_lines, use_container_width=True)
# Line chart of HERDS Data for Erie
#st.altair_chart(erie_lines_ip, use_container_width=True)

beta_decay = 0.0
#s, i, r = sim_sir(S, I, R, beta, gamma, n_days)


RateLos = namedtuple("RateLos", ("rate", "length_of_stay"))
hospitalized=RateLos(hosp_rate, hosp_los)
icu=RateLos(icu_rate, icu_los)
ventilated=RateLos(vent_rate, vent_los)


rates = tuple(each.rate for each in (hospitalized, icu, ventilated))
lengths_of_stay = tuple(each.length_of_stay for each in (hospitalized, icu, ventilated))


#############
### SIR model
s_v, i_v, r_v = sim_sir(S-2, 1, 1 ,beta, gamma, n_days)
susceptible_v, infected_v, recovered_v = s_v, i_v, r_v

i_hospitalized_v, i_icu_v, i_ventilated_v = get_dispositions(i_v, rates, regional_hosp_share)

r_hospitalized_v, r_icu_v, r_ventilated_v = get_dispositions(r_v, rates, regional_hosp_share)

dispositions = (
            i_hospitalized_v + r_hospitalized_v,
            i_icu_v + r_icu_v,
            i_ventilated_v + r_ventilated_v)

hospitalized_v, icu_v, ventilated_v = (
            i_hospitalized_v,
            i_icu_v,
            i_ventilated_v)


##############
### SEIR model
gamma2=1/infectious_period
exposed2=beta4*S*total_infections
S2=S-exposed2-total_infections

s_e, e_e, i_e, r_e = sim_seir(S-2, 1 ,1, recovered, beta3, gamma2, alpha, n_days)

susceptible_e, exposed_e, infected_e, recovered_e = s_e, e_e, i_e, r_e

i_hospitalized_e, i_icu_e, i_ventilated_e = get_dispositions(i_e, rates, regional_hosp_share)

r_hospitalized_e, r_icu_e, r_ventilated_e = get_dispositions(r_e, rates, regional_hosp_share)

dispositions_e = (
            i_hospitalized_e + r_hospitalized_e,
            i_icu_e + r_icu_e,
            i_ventilated_e + r_ventilated_e)

hospitalized_e, icu_e, ventilated_e = (
            i_hospitalized_e,
            i_icu_e,
            i_ventilated_e)


#####################################
## SEIR model with phase adjusted R_0

s_R, e_R, i_R, r_R = sim_seir_decay(S-2, 1 ,1, 0.0, beta4, gamma2,alpha, n_days, decay1, decay2, decay3)

susceptible_R, exposed_R, infected_R, recovered_R = s_R, e_R, i_R, r_R

i_hospitalized_R, i_icu_R, i_ventilated_R = get_dispositions(i_R, rates, regional_hosp_share)

r_hospitalized_R, r_icu_R, r_ventilated_R = get_dispositions(r_R, rates, regional_hosp_share)

dispositions_R = (
            i_hospitalized_R + r_hospitalized_R,
            i_icu_R + r_icu_R,
            i_ventilated_R + r_ventilated_R)

hospitalized_R, icu_R, ventilated_R = (
            i_hospitalized_R,
            i_icu_R,
            i_ventilated_R)


##################################################################
## SEIR model with phase adjusted R_0 and Disease Related Fatality

s_D, e_D, i_D, r_D, d_D = sim_seird_decay(S-2, 1, 1 , 0.0, 0.0, beta4, gamma2,alpha, n_days, decay1, decay2, decay3, fatal)

susceptible_D, exposed_D, infected_D, recovered_D = s_D, e_D, i_D, r_D

i_hospitalized_D, i_icu_D, i_ventilated_D = get_dispositions(i_D, rates, regional_hosp_share)

r_hospitalized_D, r_icu_D, r_ventilated_D = get_dispositions(r_D, rates, regional_hosp_share)

dispositions_D = (
            i_hospitalized_D + r_hospitalized_D,
            i_icu_D + r_icu_D,
            i_ventilated_D + r_ventilated_D)

hospitalized_D, icu_D, ventilated_D = (
            i_hospitalized_D,
            i_icu_D,
            i_ventilated_D)

## SEIJCRD model with phase adjusted R_0 and Disease Related Fatality

# Graphs of new admissions for Erie
# st.subheader("New Admissions: SIR Model")
# st.markdown("Projected number of **daily** COVID-19 admissions for Erie County")

# Projection days
plot_projection_days = n_days - 10

#############
# # SIR Model
# # New cases
projection_admits = build_admissions_df(dispositions)
# # Census Table
census_table = build_census_df(projection_admits)
# ############################

############
# SEIR Model
# New cases
projection_admits_e = build_admissions_df(dispositions_e)
# Census Table
census_table_e = build_census_df(projection_admits_e)

#############
# SEIR Model with phase adjustment
# New cases
projection_admits_R = build_admissions_df(dispositions_R)
# Census Table
census_table_R = build_census_df(projection_admits_R)

#############
# SEIR Model with phase adjustment and Disease Fatality
# New cases
projection_admits_D = build_admissions_df(dispositions_D)
# Census Table
census_table_D = build_census_df(projection_admits_D)

# Erie Graph of Cases: SIR, SEIR
# Admissions Graphs
# Erie Graph of Cases
def regional_admissions_chart(
    projection_admits: pd.DataFrame, 
    plot_projection_days: int,
    as_date:bool = False) -> alt.Chart:
    """docstring"""
    
    projection_admits = projection_admits.rename(columns={"hosp": "Hospitalized", "icu": "ICU", "vent": "Ventilated"})
    
    tooltip_dict = {False: "day", True: "date:T"}
    if as_date:
        projection_admits = add_date_column(projection_admits)
        x_kwargs = {"shorthand": "date:T", "title": "Date"}
    else:
        x_kwargs = {"shorthand": "day", "title": "Days from initial infection"}
    
    return (
        alt
        .Chart(projection_admits.head(plot_projection_days))
        .transform_fold(fold=["Hospitalized", "ICU", "Ventilated"])
        .mark_line(point=False)
        .encode(
            x=alt.X(**x_kwargs),
            y=alt.Y("value:Q", title="Daily admissions"),
            color="key:N",
            tooltip=[
                tooltip_dict[as_date],
                alt.Tooltip("value:Q", format=".0f", title="Admissions"),
                "key:N",
            ],
        )
        .interactive()
    )

# , scale=alt.Scale(domain=[0, 3250])

#st.altair_chart(alt.layer(altair_chart(regional_admissions_chart(projection_admits).mark_line()))
 #+ alt.layer(regional_admissions_chart(projection_admits_e).mark_line()), use_container_width=True)    


# Comparison of Single line graph - Hospitalized, ICU, Vent and All
if model_options == "Inpatient":
    columns_comp = {"hosp": "Hospitalized"}
    fold_comp = ["Hospitalized"]
    #capacity_col = {"expanded_beds":"Expanded IP Beds (50%)", "expanded_beds2":"Expanded IP Beds (100%)"}
    #capacity_fol = ["Expanded IP Beds (50%)", "Expanded IP Beds (100%)"]
    capacity_col = {"total_county_beds":"Inpatient Beds"}
    capacity_fol = ["Inpatient Beds"]
if model_options == "ICU":
    columns_comp = {"icu": "ICU"}
    fold_comp = ["ICU"]
    #capacity_col = {"expanded_icu_beds": "Expanded ICU Beds (50%)", "expanded_icu_beds2": "Expanded ICU Beds (100%)"}
    #capacity_fol = ["Expanded ICU Beds (50%)", "Expanded ICU Beds (100%)"]
    capacity_col = {"total_county_icu": "ICU Beds"}
    capacity_fol = ["ICU Beds"]
if model_options == "Ventilated":
    columns_comp = {"vent": "Ventilated"}
    fold_comp = ["Ventilated"]
    capacity_col = {"expanded_vent_beds": "Expanded Ventilators (50%)", "expanded_vent_beds2": "Expanded Ventilators (100%)"}
    capacity_fol = ["Expanded Ventilators (50%)", "Expanded Ventilators (100%)"]

def ip_chart(
    projection_admits: pd.DataFrame, 
    plot_projection_days: int,
    as_date:bool = False) -> alt.Chart:
    """docstring"""
    
    projection_admits = projection_admits.rename(columns=columns_comp|capaity_col)
    
    tooltip_dict = {False: "day", True: "date:T"}
    if as_date:
        projection_admits = add_date_column(projection_admits)
        x_kwargs = {"shorthand": "date:T", "title": "Date"}
    else:
        x_kwargs = {"shorthand": "day", "title": "Days from initial infection"}
    
    return (
        alt
        .Chart(projection_admits.head(plot_projection_days))
        .transform_fold(fold=fold_comp+capacity_fol)
        .mark_line(point=False)
        .encode(
            x=alt.X(**x_kwargs),
            y=alt.Y("value:Q", title="Daily admissions"),
            color="key:N",
            tooltip=[
                tooltip_dict[as_date],
                alt.Tooltip("value:Q", format=".0f", title="Admissions"),
                "key:N",
            ],
        )
        .interactive()
    )

#, scale=alt.Scale(domain=[0, 100])
# alt.value('orange')

# Erie County Admissions Chart: SIR
# st.altair_chart(
    # regional_admissions_chart(projection_admits, 
        # plot_projection_days, 
        # as_date=as_date), 
    # use_container_width=True)


#if st.checkbox("Show Graph for Erie County Projected Admissions: SEIR Model with Adjusted R_0 and Case Fatality"):
    # Erie Graph of Cases: SEIR with phase adjustment with phase adjustment and case fatality
    # st.subheader("New Admissions: SEIR Model")

##############################
#4/3/20 First Projection Graph - Admissions
#############
st.header("""Projected Admissions Models for Erie County""")
st.subheader("Projected number of **daily** COVID-19 admissions for Erie County: SEIR - Phase Adjusted R_0 with Case Fatality")
st.altair_chart(
    regional_admissions_chart(projection_admits_D, 
        plot_projection_days, 
        as_date=as_date), 
    use_container_width=True)


if st.checkbox("Show more info about this tool"):
    st.subheader(
     "[Deterministic SEIR model](https://www.tandfonline.com/doi/full/10.1080/23737867.2018.1509026)")
    st.markdown(
    """The model consists of individuals who are either _Susceptible_ ($S$), _Exposed_ ($E$), _Infected_ ($I$), _Recovered_ ($R$), or _Fatal_ ($D$).
The epidemic proceeds via a growth and decline process. This is the core model of infectious disease spread and has been in use in epidemiology for many years."""
)
    st.markdown("""The system of differential equations are given by the following 5 equations.""")

    st.latex(r'''\frac{ds}{dt}=-\rho_t \beta SI/N''')
    st.latex(r'''\frac{de}{dt}=\rho_t \beta SI/N - \alpha E''')
    st.latex(r'''\frac{di}{dt}= \alpha E - \gamma I''')
    st.latex(r'''\frac{dr}{dt}=(1-f) \gamma I''')
    st.latex(r'''\frac{dd}{dt}=f \gamma I''')

    st.markdown(
    """where $\gamma$ is $1/mean\ infectious\ rate$, $$\\alpha$$ is $1/mean\ incubation\ period$, $$\\rho$$ is the rate of social distancing at time $t$,
and $$\\beta$$ is the rate of transmission. More information, including parameter specifications and reasons for model choice can be found [here]("https://github.com/gabai/stream_KH/wiki).""")




sir = regional_admissions_chart(projection_admits, plot_projection_days, as_date=as_date)
seir = regional_admissions_chart(projection_admits_e, plot_projection_days, as_date=as_date)
seir_r = regional_admissions_chart(projection_admits_R, plot_projection_days, as_date=as_date)
seir_d = regional_admissions_chart(projection_admits_D, plot_projection_days, as_date=as_date)


if st.checkbox("Show Graph of Erie County Projected Admissions with Model Comparison of Social Distancing"):
    st.subheader("Projected number of **daily** COVID-19 admissions for Erie County: Model Comparison (Left: 0% Social Distancing, Right: Step-Wise Social Distancing)")
    st.altair_chart(
        alt.layer(seir.mark_line())
        + alt.layer(seir_d.mark_line())
        , use_container_width=True)


# if st.checkbox("Show Graph of Erie County Projected Admissions: SEIR Model with no social distancing"):
    # # Erie Graph of Cases: SEIR Model
    # # st.subheader("New Admissions: SEIR Model")
    # st.subheader("Projected number of **daily** COVID-19 admissions for Erie County without social distancing using the SEIR model")
    # # Erie Graph of Cases: SEIR
    # st.altair_chart(
        # regional_admissions_chart(projection_admits_e, 
            # plot_projection_days, 
            # as_date=as_date), 
        # use_container_width=True)

##if st.checkbox("Show Graph for Erie County Projected Admissions: SEIR Model with Adjusted R_0"):
##    # Erie Graph of Cases: SEIR with phase adjustment
##    # st.subheader("New Admissions: SEIR Model")
##    st.subheader("Projected number of **daily** COVID-19 admissions for Erie County: Phase Adjusted R_0")
##    st.altair_chart(
##        regional_admissions_chart(projection_admits_R, 
##            plot_projection_days, 
##            as_date=as_date), 
##        use_container_width=True)

########### Tabular DATA ########    
# if st.checkbox("Show Projected Admissions in tabular form: SEIR"):
    # admits_table_e = projection_admits_e[np.mod(projection_admits_e.index, 7) == 0].copy()
    # admits_table_e["day"] = admits_table_e.index
    # admits_table_e.index = range(admits_table_e.shape[0])
    # admits_table_e = admits_table_e.fillna(0).astype(int)
    
    # st.dataframe(admits_table_e)

# if st.checkbox("Show Projected Admissions in tabular form: SEIR with Phase Adjusted R_0"):
    # admits_table_R = projection_admits_R[np.mod(projection_admits_R.index, 7) == 0].copy()
    # admits_table_R["day"] = admits_table_R.index
    # admits_table_R.index = range(admits_table_R.shape[0])
    # admits_table_R = admits_table_R.fillna(0).astype(int)
    
    # st.dataframe(admits_table_R)
    
# if st.checkbox("Show Projected Admissions in tabular form: SEIR with Phase Adjusted R_0 and Case Fatality"):
    # admits_table_D = projection_admits_D[np.mod(projection_admits_D.index, 7) == 0].copy()
    # admits_table_D["day"] = admits_table_D.index
    # admits_table_D.index = range(admits_table_D.shape[0])
    # admits_table_D = admits_table_D.fillna(0).astype(int)
    
    # st.dataframe(admits_table_D)
    


################################################
################################################
#############    Census Graphs        ##########
################################################
################################################
st.header("""Projected Census Models for Erie County""")
# def admitted_patients_chart(
    # census: pd.DataFrame,
    # plot_projection_days: int,
    # as_date:bool = False) -> alt.Chart:
    # """docstring"""
    # census = census.rename(columns={"hosp": "Hospital Census", "icu": "ICU Census", "vent": "Ventilated Census", 
    # "expanded_beds_county":"Expanded IP Beds", "expanded_icu_county": "Expanded ICU Beds"})

    # tooltip_dict = {False: "day", True: "date:T"}
    # if as_date:
        # census = add_date_column(census.head(plot_projection_days))
        # x_kwargs = {"shorthand": "date:T", "title": "Date"}
    # else:
        # x_kwargs = {"shorthand": "day", "title": "Days from initial infection"}

    # return (
        # alt
        # .Chart(census)
        # .transform_fold(fold=["Hospital Census", "ICU Census", "Ventilated Census", "Expanded IP Beds", "Expanded ICU Beds"])
        # .mark_line(point=False)
        # .encode(
            # x=alt.X(**x_kwargs),
            # y=alt.Y("value:Q", title="Census"),
            # color="key:N",
            # tooltip=[
                # tooltip_dict[as_date],
                # alt.Tooltip("value:Q", format=".0f"),
                # "key:N",
            # ],
        # )
        # .interactive()
    # )


# Comparison of Census Single line graph - Hospitalized, ICU, Vent
if model_options == "Inpatient":
    #columns_comp_census = {"hosp": "Hospital Census", "expanded_beds_county":"Expanded IP Beds (50%)", "expanded_beds_county2":"Expanded IP Beds (100%)"}
    #fold_comp_census = ["Hospital Census", "Expanded IP Beds (50%)", "Expanded IP Beds (100%)"]
    columns_comp_census = {"hosp": "Hospital Census", "total_county_beds":"Inpatient Beds"}
    fold_comp_census = ["Hospital Census", "Inpatient Beds"]
    graph_selection = erie_lines_ip
if model_options == "ICU":
    #columns_comp_census = {"icu": "ICU Census", "expanded_icu_county": "Expanded ICU Beds (50%)", "expanded_icu_county2": "Expanded ICU Beds (100%)"}
    #fold_comp_census = ["ICU Census", "Expanded ICU Beds (50%)", "Expanded ICU Beds (100%)"]
    columns_comp_census = {"icu": "ICU Census", "total_county_icu": "ICU Beds"}
    fold_comp_census = ["ICU Census", "ICU Beds"]
    graph_selection = erie_lines_icu
if model_options == "Ventilated":
    columns_comp_census = {"vent": "Ventilated Census"}
    fold_comp_census = ["Ventilated Census"]
    graph_selection = erie_lines_vent

def ip_census_chart(
    census: pd.DataFrame,
    plot_projection_days: int,
    as_date:bool = False) -> alt.Chart:
    """docstring"""
    census = census.rename(columns=columns_comp_census)

    tooltip_dict = {False: "day", True: "date:T"}
    if as_date:
        census = add_date_column(census.head(plot_projection_days))
        x_kwargs = {"shorthand": "date:T", "title": "Date"}
    else:
        x_kwargs = {"shorthand": "day", "title": "Days from initial infection"}

    return (
        alt
        .Chart(census)
        .transform_fold(fold=fold_comp_census)
        .mark_line(point=False)
        .encode(
            x=alt.X(**x_kwargs),
            y=alt.Y("value:Q", title="Census"),
            color="key:N",
            tooltip=[
                tooltip_dict[as_date],
                alt.Tooltip("value:Q", format=".0f", title="Census"),
                "key:N",
            ],
        )
        .interactive()
    )


################# Add 0% 10% 20% SD graph of SEIR MODEL ###################

    #, scale=alt.Scale(domain=[0, 40000])
    # scale=alt.Scale(domain=[-5, 9000])
    


#sir_ip_c = ip_census_chart(census_table, plot_projection_days, as_date=as_date)
seir_ip_c = ip_census_chart(census_table_e, plot_projection_days, as_date=as_date)
#seir_r_ip_c = ip_census_chart(census_table_R, plot_projection_days, as_date=as_date)
seir_d_ip_c = ip_census_chart(census_table_D, plot_projection_days, as_date=as_date)
###

# st.subheader("Projected **census** of COVID-19 patients for Erie County: Model Comparison")

# st.altair_chart(
   # #alt.layer(sir_ip_c.mark_line())
   # #+
   # alt.layer(seir_ip_c.mark_line())
   # + alt.layer(seir_r_ip_c.mark_line())
   # + alt.layer(seir_d_ip_c.mark_line())
   # , use_container_width=True)

# Chart of Model Comparison for SEIR and Adjusted with Erie County Data
st.subheader("Comparison of COVID-19 admissions for Erie County: Data vs Model")
st.altair_chart(
    alt.layer(seir_ip_c.mark_line())
    + alt.layer(seir_d_ip_c.mark_line())
    + alt.layer(graph_selection)
    , use_container_width=True)



# 4/6/20 New Model with added 10-20 SD
# Chart of Model Comparison for SEIR and Adjusted with Erie County Data
# st.subheader("Comparison of COVID-19 admissions for Erie County: Data vs Model")
# st.altair_chart(
    # alt.layer(seir_ip_c.mark_line())
    # + alt.layer(seir_ip_c10.mark_line())
    # + alt.layer(seir_ip_c20.mark_line())
    # + alt.layer(seir_d_ip_c.mark_line())
    # + alt.layer(graph_selection)
    # , use_container_width=True)

#if st.checkbox("Show Graph for Erie County Projected Census: SEIR Model"):
    # Erie County Census Graph - SEIR Model
    # st.subheader("Projected **census** of COVID-19 patients for Erie County, accounting for arrivals and discharges: **SEIR Model**")
    # st.altair_chart(
        # admitted_patients_chart(
            # census_table_e,
            # plot_projection_days,
            # as_date=as_date),
        # use_container_width=True)

##if st.checkbox("Show Graph for Erie County Projected Census: SEIR Model with adjusted R_0"):
##    # Erie County Census Graph - SEIR Model
##    st.subheader("Projected **census** of COVID-19 patients for Erie County, accounting for arrivals and discharges: **SEIR Model**")
##    st.altair_chart(
##        admitted_patients_chart(
##            census_table_R,
##            plot_projection_days,
##            as_date=as_date),
##        use_container_width=True)
##
### , scale=alt.Scale(domain=[0, 30000])


# if st.checkbox("Show Graph for Erie County Projected Census: SEIR Model with adjusted R_0 and Case Fatality"):
    # # Erie County Census Graph - SEIR Model
    # st.subheader("Projected **census** of COVID-19 patients for Erie County, accounting for arrivals and discharges: **SEIR Model**")
    # st.altair_chart(
        # admitted_patients_chart(
            # census_table_D,
            # plot_projection_days,
            # as_date=as_date),
        # use_container_width=True)

# , scale=alt.Scale(domain=[0, 30000])

# Erie Graph of Beds
def bed_lines(
    projection_admits: pd.DataFrame) -> alt.Chart:
    """docstring"""
    
    projection_admits = projection_admits.rename(columns={"total_county_icu": "ICU Beds", 
                                                            "total_county_beds":"Inpatient Beds", 
                                                            "expanded_icu_county":"Expanded ICU 50%",
                                                            "expanded_beds_county":"Expanded Inpatient 50%",
                                                            "expanded_icu_county2":"Expanded ICU 100%",
                                                            "expanded_beds_county2":"Expanded Inpatient 100%"
                                                            })
    
    return(
        alt
        .Chart(projection_admits)
        .transform_fold(fold=["ICU Beds", 
                                "Inpatient Beds", 
                                "Expanded ICU 50%",
                                "Expanded Inpatient 50%",
                                "Expanded ICU 100%",
                                "Expanded Inpatient 100%"
                                ])
        .mark_line(point=False)
        .encode(
            x=alt.X("day", title="Date"),
            y=alt.Y("value:Q", title="Erie County Bed Census"),
            color="key:N",
            tooltip=[alt.Tooltip("value:Q", format=".0f"),"key:N"]
        )
        .interactive()
    )

# beds = bed_lines(census_table)
# st.altair_chart(beds, use_container_width=True)











##########################################################
##########################################################
###########            PPE            ####################
##########################################################
##########################################################
st.header("Projected PPE Needs for Erie County")
def ppe_chart(
    census: pd.DataFrame,
    as_date:bool = False) -> alt.Chart:
    """docstring"""
    census = census.rename(columns={'ppe_mean_mild': 'Mean PPE needs - mild cases', 'ppe_mean_severe': 'Mean PPE needs - severe cases'})
    tooltip_dict = {False: "day", True: "date:T"}
    if as_date:
        census = add_date_column(census)
        x_kwargs = {"shorthand": "date:T", "title": "Date"}
    else:
        x_kwargs = {"shorthand": "day", "title": "Days from initial infection"}

    return (
        alt
        .Chart(census)
        .transform_fold(fold=['Mean PPE needs - mild cases', 'Mean PPE needs - severe cases'])
        .mark_line(point=False)
        .encode(
            x=alt.X(**x_kwargs),
            y=alt.Y("value:Q", title="Projected PPE needs per day"),
            color="key:N",
            tooltip=[
                tooltip_dict[as_date],
                alt.Tooltip("value:Q", format=".0f", title="PPE Needs"),
                "key:N",
            ],
        )
        .interactive()
    )

# , scale=alt.Scale(domain=[0, 450000])

#SIR Model with PPE predictions
# st.subheader("Projected personal protective equipment needs for mild and severe cases of COVID-19: SIR Model")
# st.altair_chart(
    # ppe_chart(
    # census_table,
    # as_date=as_date),
    # use_container_width=True)
    
# # SEIR Model with PPE predictions
# st.subheader("Projected personal protective equipment needs for mild and severe cases of COVID-19: SEIR Model")
# st.altair_chart(
    # ppe_chart(
    # census_table_e,
    # as_date=as_date),
    # use_container_width=True)

# # SEIR Model with adjusted R_0 with - PPE predictions
# st.subheader("Projected personal protective equipment needs for mild and severe cases of COVID-19: SEIR Model with Adjuted R_0")
# st.altair_chart(
    # ppe_chart(
    # census_table_R,
    # as_date=as_date),
    # use_container_width=True)
    
# SEIR Model with adjusted R_0 with Case Fatality - PPE predictions
st.subheader("Projected personal protective equipment needs for mild and severe cases of COVID-19: SEIR Model with Adjutes R_0 and Case Fatality")
st.altair_chart(
    ppe_chart(
    census_table_D,
    as_date=as_date),
    use_container_width=True)

#############################
# PPE needs summary variables
def get_key(dic, val): 
    for key, value in dic.items(): 
         if val == value: 
             return key 

float_formatter = "{:.2f}".format

# ppe_7d_mild_lower = census_table['ppe_mild_d'][1:8]
# ppe_7d_mild_upper = census_table['ppe_mild_u'][1:8]
# ppe_7d_severe_lower = census_table['ppe_severe_d'][1:8]
# ppe_7d_severe_upper = census_table['ppe_severe_u'][1:8]

# st.table(ppe_7d_mild_lower)
# st.table(ppe_7d_mild_upper)
# st.table(ppe_7d_severe_lower)
# st.table(ppe_7d_severe_upper)


# # One day
# ppe_1d_mild_lower = current_hosp * 14.0
# ppe_1d_mild_upper = current_hosp * 15.0
# ppe_1d_severe_lower = current_hosp *15.0
# ppe_1d_severe_upper = current_hosp *24.0
# # 7 Days
# ppe_7d_mild_lower = ppe_1d_mild_lower+sum(census_table['ppe_mild_d'][1:8])
# ppe_7d_mild_upper = ppe_1d_mild_upper+sum(census_table['ppe_mild_u'][1:8])
# ppe_7d_severe_lower = ppe_1d_severe_lower+sum(census_table['ppe_severe_d'][1:8])
# ppe_7d_severe_upper = ppe_1d_severe_upper+sum(census_table['ppe_severe_u'][1:8])
# # 2nd week
# ppe_14d_mild_lower = sum(census_table['ppe_mild_d'][1:15])
# ppe_14d_mild_upper = sum(census_table['ppe_mild_u'][1:15])
# ppe_14d_severe_lower = sum(census_table['ppe_severe_d'][1:15])
# ppe_14d_severe_upper = sum(census_table['ppe_severe_u'][1:15])
# # 3rd week
# ppe_21d_mild_lower = sum(census_table['ppe_mild_d'][1:22])
# ppe_21d_mild_upper = sum(census_table['ppe_mild_u'][1:22])
# ppe_21d_severe_lower = sum(census_table['ppe_severe_d'][1:22])
# ppe_21d_severe_upper = sum(census_table['ppe_severe_u'][1:22])
# # Month one
# ppe_1m_mild_lower = sum(census_table['ppe_mild_d'][1:29])
# ppe_1m_mild_upper = sum(census_table['ppe_mild_u'][1:29])
# ppe_1m_severe_lower = sum(census_table['ppe_severe_d'][1:29])
# ppe_1m_severe_upper = sum(census_table['ppe_severe_u'][1:29])


# SEIR Model

# # One day
# ppe_1d_mild_lower = current_hosp * 14.0
# ppe_1d_mild_upper = current_hosp * 15.0
# ppe_1d_severe_lower = current_hosp *15.0
# ppe_1d_severe_upper = current_hosp *24.0
# # 7 Days
# ppe_7d_mild_lower = ppe_1d_mild_lower+sum(census_table_e['ppe_mild_d'][1:8])
# ppe_7d_mild_upper = ppe_1d_mild_upper+sum(census_table_e['ppe_mild_u'][1:8])
# ppe_7d_severe_lower = ppe_1d_severe_lower+sum(census_table_e['ppe_severe_d'][1:8])
# ppe_7d_severe_upper = ppe_1d_severe_upper+sum(census_table_e['ppe_severe_u'][1:8])
# # 2nd week
# ppe_14d_mild_lower = sum(census_table_e['ppe_mild_d'][1:15])
# ppe_14d_mild_upper = sum(census_table_e['ppe_mild_u'][1:15])
# ppe_14d_severe_lower = sum(census_table_e['ppe_severe_d'][1:15])
# ppe_14d_severe_upper = sum(census_table_e['ppe_severe_u'][1:15])
# # 3rd week
# ppe_21d_mild_lower = sum(census_table_e['ppe_mild_d'][1:22])
# ppe_21d_mild_upper = sum(census_table_e['ppe_mild_u'][1:22])
# ppe_21d_severe_lower = sum(census_table_e['ppe_severe_d'][1:22])
# ppe_21d_severe_upper = sum(census_table_e['ppe_severe_u'][1:22])
# # Month one
# ppe_1m_mild_lower = sum(census_table_e['ppe_mild_d'][1:29])
# ppe_1m_mild_upper = sum(census_table_e['ppe_mild_u'][1:29])
# ppe_1m_severe_lower = sum(census_table_e['ppe_severe_d'][1:29])
# ppe_1m_severe_upper = sum(census_table_e['ppe_severe_u'][1:29])

# st.markdown("""The estimated **daily** PPE needs for the currently admitted COVID-19 patients is **{ppe_1d_mild_lower:.0f}**-**{ppe_1d_mild_upper:.0f}** for mild cases of COVID-19, 
            # and **{ppe_1d_severe_lower:.0f}**-**{ppe_1d_severe_upper:.0f}** for severe cases. The estimated PPE needs for the **first month** of COVID-19 patients is
            # **{ppe_1m_mild_lower:.0f}**-**{ppe_1m_mild_upper:.0f}** for mild cases of COVID-19, and **{ppe_1m_severe_lower:.0f}**-**{ppe_1m_severe_upper:.0f}** for severe cases.""".format(
                # ppe_1d_mild_lower = ppe_1d_mild_lower,
                # ppe_1d_mild_upper = ppe_1d_mild_upper,
                # ppe_1d_severe_lower = ppe_1d_severe_lower,
                # ppe_1d_severe_upper = ppe_1d_severe_upper,
                # ppe_1m_mild_lower = ppe_1m_mild_lower,
                # ppe_1m_mild_upper = ppe_1m_mild_upper,
                # ppe_1m_severe_lower = ppe_1m_severe_lower,
                # ppe_1m_severe_upper = ppe_1m_severe_upper
# ))

# # PPE Needs by day, 7d, 14, 21, 28d
# data = {
    # 'PPE Needs': ['Today', 'Week 1', 'Week2', 'Week3', 'First Month'],
    # 'Mild Cases' : [
        # float_formatter(np.mean([ppe_1d_mild_lower, ppe_1d_mild_upper])), 
        # float_formatter(np.mean([ppe_7d_mild_lower, ppe_7d_mild_upper])), 
        # float_formatter(np.mean([ppe_14d_mild_lower, ppe_14d_mild_upper])), 
        # float_formatter(np.mean([ppe_21d_mild_lower, ppe_21d_mild_upper])), 
        # float_formatter(np.mean([ppe_1m_mild_lower, ppe_1m_mild_upper]))
        # ],
    # 'Severe Cases': [
        # float_formatter(np.mean([ppe_1d_severe_lower, ppe_1d_severe_upper])), 
        # float_formatter(np.mean([ppe_7d_severe_lower, ppe_7d_severe_upper])), 
        # float_formatter(np.mean([ppe_14d_severe_lower, ppe_14d_severe_upper])),
        # float_formatter(np.mean([ppe_21d_severe_lower, ppe_21d_severe_upper])), 
        # float_formatter(np.mean([ppe_1m_severe_lower, ppe_1m_severe_upper]))
        # ], 
# }
# ppe_needs = pd.DataFrame(data)
# st.table(ppe_needs)

### Recovered/Infected table
##st.subheader("The number of infected and recovered individuals in the region at any given moment")
##
##def additional_projections_chart(i: np.ndarray, r: np.ndarray) -> alt.Chart:
##    dat = pd.DataFrame({"Infected": i, "Recovered": r})
##
##    return (
##        alt
##        .Chart(dat.reset_index())
##        .transform_fold(fold=["Infected", "Recovered"])
##        .mark_line(point=False)
##        .encode(
##            x=alt.X("index", title="Days from today"),
##            y=alt.Y("value:Q", title="Case Volume"),
##            tooltip=["key:N", "value:Q"], 
##            color="key:N"
##        )
##        .interactive()
##    )
##
##st.altair_chart(additional_projections_chart(i_R, r_R), use_container_width=True)
##



# # Recovered/Infected/Hospitalized/Fatality table
# st.subheader("The number of infected,recovered, and fatal individuals in the region at any given moment")

# def additional_projections_chart(i: np.ndarray, j: np.ndarray, c: np.ndarray,r: np.ndarray, d: np.ndarray) -> alt.Chart:
    # dat = pd.DataFrame({"Infected": i, "Hospitalized":j, "ICU":c, "Recovered": r, "Fatal":d})

    # return (
        # alt
        # .Chart(dat.reset_index())
        # .transform_fold(fold=["Infected", "Hospitalized", "ICU", "Recovered", "Fatal"])
        # .mark_line(point=False)
        # .encode(
            # x=alt.X("index", title="Days from today"),
            # y=alt.Y("value:Q", title="Case Volume"),
            # tooltip=["key:N", "value:Q"], 
            # color="key:N"
        # )
        # .interactive()
    # )

# st.altair_chart(additional_projections_chart(i_H, j_H, c_H, r_H, d_H), use_container_width=True)



##if st.checkbox("Show Additional Information"):
##
## st.subheader("Guidance on Selecting Inputs")
## st.markdown(
##     """* **Hospitalized COVID-19 Patients:** The number of patients currently hospitalized with COVID-19. This number is used in conjunction with Hospital Market Share and Hospitalization % to estimate the total number of infected individuals in your region.
##     * **Currently Known Regional Infections**: The number of infections reported in your hospital's catchment region. This input is used to estimate the detection rate of infected individuals. 
##     * **Doubling Time (days):** This parameter drives the rate of new cases during the early phases of the outbreak. The American Hospital Association currently projects doubling rates between 7 and 10 days. This is the doubling time you expect under status quo conditions. To account for reduced contact and other public health interventions, modify the _Social distancing_ input. 
##     * **Social distancing (% reduction in person-to-person physical contact):** This parameter allows users to explore how reduction in interpersonal contact & transmission (hand-washing) might slow the rate of new infections. It is your estimate of how much social contact reduction is being achieved in your region relative to the status quo. While it is unclear how much any given policy might affect social contact (eg. school closures or remote work), this parameter lets you see how projections change with percentage reductions in social contact.
##     * **Hospitalization %(total infections):** Percentage of **all** infected cases which will need hospitalization.
##     * **ICU %(total infections):** Percentage of **all** infected cases which will need to be treated in an ICU.
##     * **Ventilated %(total infections):** Percentage of **all** infected cases which will need mechanical ventilation.
##     * **Hospital Length of Stay:** Average number of days of treatment needed for hospitalized COVID-19 patients. 
##     * **ICU Length of Stay:** Average number of days of ICU treatment needed for ICU COVID-19 patients.
##     * **Vent Length of Stay:**  Average number of days of ventilation needed for ventilated COVID-19 patients.
##     * **Hospital Market Share (%):** The proportion of patients in the region that are likely to come to your hospital (as opposed to other hospitals in the region) when they get sick. One way to estimate this is to look at all of the hospitals in your region and add up all of the beds. The number of beds at your hospital divided by the total number of beds in the region times 100 will give you a reasonable starting estimate.
##     * **Regional Population:** Total population size of the catchment region of your hospital(s). 
##     """)

# # Show data
# days = np.array(range(0, n_days + 1))
# data_list = [days, s_R, e_R, i_R, r_R]
# data_dict = dict(zip(["day", "susceptible", "exposed","infections", "recovered"], data_list))
# projection_area = pd.DataFrame.from_dict(data_dict)
# infect_table = (projection_area.iloc[::7, :]).apply(np.floor)
# infect_table.index = range(infect_table.shape[0])

# if st.checkbox("Show Raw SEIR Similation Data"):
    # st.dataframe(infect_table)

## # Show data
##days1 = np.array(range(0, n_days + 1))
##data_list1 = [days1, s_D, e_D, i_D, r_D, d_D]
##data_dict1 = dict(zip(["day", "susceptible", "exposed","infections", "recovered", "fatal"], data_list1))
##projection_area1 = pd.DataFrame.from_dict(data_dict1)
##infect_table1 = (projection_area1.iloc[::7, :]).apply(np.floor)
##infect_table1.index = range(infect_table1.shape[0])
##
##if st.checkbox("Show Raw SEIR Similation Data with case fatality" ):
##    st.dataframe(infect_table1)


# Show data
#days3 = np.array(range(0, n_days + 1))
#data_list3 = [days3, s_H, e_H, i_H, j_H, c_H, r_H, d_H]
#data_dict3 = dict(zip(["day", "susceptible", "exposed","infections", "hospitalized", "ICU", "recovered", "fatal"], data_list3))
#projection_area3 = pd.DataFrame.from_dict(data_dict3)
#infect_table3 = (projection_area3.iloc[::7, :]).apply(np.floor)
#infect_table3.index = range(infect_table3.shape[0])

#if st.checkbox("Show Raw SEIJCR Similation Data" ):
#    st.dataframe(infect_table3)


# Show data
#days2 = np.array(range(0, n_days + 1))
#data_list2 = [days2, s_v, i_v, r_v]
#data_dict2 = dict(zip(["day", "susceptible", "infections", "recovered"], data_list2))
#projection_area2 = pd.DataFrame.from_dict(data_dict2)
#infect_table2 = (projection_area2.iloc[::7, :]).apply(np.floor)
#infect_table2.index = range(infect_table2.shape[0])

#if st.checkbox("Show Raw SIR Similation Data with case fatality" ):
#    st.dataframe(infect_table2)



# st.subheader("References & Acknowledgements")
# st.markdown(
    # """
    # We appreciate the great work done by Predictive Healthcare team (http://predictivehealthcare.pennmedicine.org/) at
# Penn Medicine who created the predictive model used.
    # https://www.worldometers.info/coronavirus/coronavirus-incubation-period/
# Lauer SA, Grantz KH, Bi Q, et al. The Incubation Period of Coronavirus Disease 2019 (COVID-19) From Publicly Reported Confirmed Cases: Estimation and Application. Ann Intern Med. 2020; [Epub ahead of print 10 March 2020]. doi: https://doi.org/10.7326/M20-0504
# http://www.centerforhealthsecurity.org/resources/COVID-19/index.html
# http://www.centerforhealthsecurity.org/resources/fact-sheets/pdfs/coronaviruses.pdf'
# https://coronavirus.jhu.edu/
# https://www.who.int/emergencies/diseases/novel-coronavirus-2019/situation-reports
# https://www.worldometers.info/coronavirus/coronavirus-age-sex-demographics/
    # """
# )



# Recovered/Infected/Fatality table
st.subheader("Infected,recovered,and fatal individuals in the **region** across time")

def additional_projections_chart(i: np.ndarray, r: np.ndarray, d: np.ndarray) -> alt.Chart:
    dat = pd.DataFrame({"Infected": i, "Recovered": r, "Fatal":d})

    return (
        alt
        .Chart(dat.reset_index())
        .transform_fold(fold=["Infected"])
        .mark_line(point=False)
        .encode(
            x=alt.X("index", title="Days from initial infection"),
            y=alt.Y("value:Q", title="Case Volume"),
            tooltip=["key:N", "value:Q"], 
            color="key:N"
        )
        .interactive()
    )

recov_infec = additional_projections_chart(i_D, r_D, d_D)


def death_chart(i: np.ndarray, r: np.ndarray, d: np.ndarray) -> alt.Chart:
    dat = pd.DataFrame({"Infected": i, "Recovered": r, "Fatal":d})

    return (
        alt
        .Chart(dat.reset_index())
        .transform_fold(fold=["Fatal"])
        .mark_bar()
        .encode(
            x=alt.X("index", title="Days from initial infection"),
            y=alt.Y("value:Q", title="Case Volume"),
            tooltip=["key:N", "value:Q"], 
            color=alt.value('red')
        )
        .interactive()
    )

deaths = death_chart(i_D, r_D, d_D)

st.altair_chart(deaths + recov_infec, use_container_width=True)



total_fatalities=max(d_D)
infection_total_t=max(d_D)+max(r_D)
st.markdown(
    """There is a projected number of **{total_fatalities:.0f}** fatalities due to COVID-19.""".format(
        total_fatalities=total_fatalities 
    ))

st.markdown("""There is a projected number of **{infection_total_t:.0f}** infections due to COVID-19.""".format(
        infection_total_t=infection_total_t
    )
            )
AAA=beta4*(1/gamma2)*1500000
R2=AAA*(1-decay2)
R3=AAA*(1-decay3)

st.markdown("""The initial $R_0$ is **{AAA:.1f}** the $R_0$ after 2 weeks is **{R2:.1f}** and the $R_0$ after 3 weeks is **{R3:.1f}**.  This is based on a doubling rate of **{doubling_time:.0f}** and the calculation of the [basic reproduction number](https://www.sciencedirect.com/science/article/pii/S2468042719300491).  """.format(
        AAA=AAA,
        R2=R2,
        R3=R3,
        doubling_time=doubling_time
    )
            )
