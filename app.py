# COVID-19 
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


###########################
# Models and base functions
###########################
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
    census_df = census_df[["day", "hosp", "icu", "vent", 
    ]]
    
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
    decay1:float, decay2:float, decay3: float, decay4: float, end_delta: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate the SIR model forward in time."""
    s, e, i, r = (float(v) for v in (s, e, i, r))
    n = s + e + i + r
    s_v, e_v, i_v, r_v = [s], [e], [i], [r]
    for day in range(n_days):
        if start_day<=day<=int1_delta:
            beta_decay=beta*(1-decay1)
        elif int1_delta<=day<=int2_delta:
            beta_decay=beta*(1-decay2)
        elif int2_delta<=day<=end_delta:
            beta_decay=beta*(1-decay3)
        else:
            beta_decay=beta*(1-decay4)
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
    decay1:float, decay2:float, decay3: float, decay4: float, end_delta: int, fatal: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate the SIR model forward in time."""
    s, e, i, r, d= (float(v) for v in (s, e, i, r, d))
    n = s + e + i + r + d
    s_v, e_v, i_v, r_v, d_v = [s], [e], [i], [r], [d]
    for day in range(n_days):
        if start_day<=day<=int1_delta:
            beta_decay=beta*(1-decay1)
        elif int1_delta<=day<=int2_delta:
            beta_decay=beta*(1-decay2)
        elif int2_delta<=day<=end_delta:
            beta_decay=beta*(1-decay3)
        else:
            beta_decay=beta*(1-decay4)
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
    decay1:float, decay2:float, decay3: float, decay4: float, end_delta: int, fatal_hosp: float, hosp_rate: float, icu_rate: float, icu_days:float, crit_lag: float, death_days:float
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
        elif 29<=day<=end_delta: 
            beta_decay=beta*(1-decay3)
        else:
            beta_decay=beta*(1-decay4)
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

# Less complicated

def seijcrd2(
    s: float, e: float, i: float, j:float, r: float, d: float, beta: float, gamma: float, alpha: float, n: float, fatal: float, fatal_hosp: float, hosp_rate:float,
    hosp_day_rate:float, l:float
    ) -> Tuple[float, float, float, float, float,float]:
    """The SIR model, one time step."""
    s_n = -beta*s*(i + (l*j)) +s
    e_n = beta*s*(i + (l*j)) - alpha * e + e
    i_n = alpha * e - (hosp_rate + gamma) * i + i 
    j_n = hosp_rate * i - hosp_day_rate*j +j
    r_n = gamma * (1-fatal)*i + ((1-fatal_hosp) * hosp_day_rate * j) +r
    d_n = gamma * (fatal)*i + ((fatal_hosp)*hosp_day_rate*j) +d
    if s_n < 0.0:
        s_n = 0.0
    if e_n < 0.0:
        e_n = 0.0
    if i_n < 0.0:
        i_n = 0.0
    if j_n < 0.0:
        j_n = 0.0
    if r_n < 0.0:
        r_n = 0.0
    if d_n < 0.0:
        d_n = 0.0

    scale = n / (s_n + e_n+ i_n + j_n+ r_n + d_n)
    return s_n * scale, e_n * scale, i_n * scale, j_n* scale, r_n * scale, d_n * scale
# 

def sim_seijcrd_decay2(
    s: float, e:float, i: float, j:float, r: float, d: float, beta: float, gamma: float, alpha: float, n_days: int,
    decay2:float, decay3: float, fatal: float, fatal_hosp: float, hosp_rate: float, hosp_day_rate:float, l:float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,np.ndarray, np.ndarray]:
    """Simulate the SIR model forward in time."""
    s, e, i, j, r, d= (float(v) for v in (s, e, i, j, r, d))
    n = s + e + i + j+r + d
    s_v, e_v, i_v, j_v, r_v, d_v = [s], [e], [i], [j], [r], [d]
    for day in range(n_days):
        if 0<=day<=21:
            beta_decay=beta
        elif 22<=day<=28:
            beta_decay=beta*(1-decay2)
        else: 
            beta_decay=beta*(1-decay3)
        s, e, i,j, r,d = seijcrd2(s, e, i,j, r, d, beta_decay, gamma, alpha, n, fatal, fatal_hosp, hosp_rate,hosp_day_rate, l)
        s_v.append(s)
        e_v.append(e)
        i_v.append(i)
        j_v.append(j)
        r_v.append(r)
        d_v.append(d)

    return (
        np.array(s_v),
        np.array(e_v),
        np.array(i_v),
        np.array(j_v),
        np.array(r_v),
        np.array(d_v)
    )
 

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
    start = start_date
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

    
# PPE Values
ppe_mild_val_lower = 14
ppe_mild_val_upper = 15
ppe_severe_val_lower = 15
ppe_severe_val_upper = 24

# List of Groups
groups = ['hosp', 'icu', 'vent']

# Widgets
# model_options = st.sidebar.radio(
    # "Service", ('Inpatient', 'ICU', 'Ventilated'))

location_option = st.sidebar.radio(
    "Location", ('United States', 'New York State', 'Erie County, NY'))

if location_option =='United States':
    S = 328000000
    first_case_date = datetime(2020,1,20)
if location_option =='New York State':
    S = 19450000
    first_case_date = datetime(2020,3,1)
if location_option =='Erie County, NY':
    S = 1500000
    first_case_date = datetime(2020,3,16)

# Populations and Infections
population = S
cases = 1000.0
S_default = population
known_infections = 1000.0
known_cases = 120.0
regional_hosp_share = 1.0
current_hosp =known_cases

##current_hosp = st.sidebar.number_input(
##    "Total Hospitalized Cases", value=known_cases, step=1.0, format="%f")

doubling_time = st.sidebar.number_input(
    "Doubling Time (days)", value=5.0, step=1.0, format="%f")

start_date = st.sidebar.date_input(
    "Suspected first contact", first_case_date)
start_day = 1

##relative_contact_rate = st.sidebar.number_input(
##    "Social distancing (% reduction in social contact) Unadjusted Model", 0, 100, value=0, step=5, format="%i")/100.0
relative_contact_rate=0
decay1 = st.sidebar.number_input(
    "Social distancing (% reduction in social contact) in Week 0-2", 0, 100, value=0, step=5, format="%i")/100.0

intervention1 = st.sidebar.date_input(
    "Date of change Social Distancing - School Closure", datetime(2020,3,18))
int1_delta = (intervention1 - start_date).days
    
decay2 = st.sidebar.number_input(
    "Social distancing (% reduction in social contact) in Week 3 - School Closure", 0, 100, value=15, step=5, format="%i")/100.0

intervention2 = st.sidebar.date_input(
    "Date of change in Social Distancing - Closure Businesses, Shelter in Place", datetime(2020,3,25))
int2_delta = (intervention2 - start_date).days

decay3 = st.sidebar.number_input(
    "Social distancing (% reduction in social contact) from Week 3 to change in SD - After Business Closure%", 0, 100, value=40 ,step=5, format="%i")/100.0

end_date = st.sidebar.date_input(
    "End date or change in social distancing", datetime(2020,5,15))
# Delta from start and end date for decay4
end_delta = (end_date - start_date).days

decay4 = st.sidebar.number_input(
    "Social distancing after end date", 0, 100, value=20 ,step=5, format="%i")/100.0

hosp_rate = (
    st.sidebar.number_input("Hospitalization %", 0.0, 100.0, value=2.5, step=0.50, format="%f")/ 100.0)

icu_rate = (
    st.sidebar.number_input("ICU %", 0.0, 100.0, value=1.25, step=0.25, format="%f") / 100.0)

vent_rate = (
    st.sidebar.number_input("Ventilated %", 0.0, 100.0, value=1.0, step=0.25, format="%f")/ 100.0)

incubation_period =(
    st.sidebar.number_input("Incubation Period", 0.0, 12.0, value=5.8, step=0.1, format="%f"))

recovery_days =(
    st.sidebar.number_input("Recovery Period", 0.0, 21.0, value=14.0 ,step=0.1, format="%f"))

infectious_period =(
    st.sidebar.number_input("Infectious Period", 0.0, 18.0, value=3.0,step=0.1, format="%f"))

fatal = st.sidebar.number_input(
    "Overall Fatality (%)", 0.0, 100.0, value=0.6 ,step=0.1, format="%f")/100.0

fatal_hosp = st.sidebar.number_input(
    "Hospital Fatality (%)", 0.0, 100.0, value=4.0 ,step=0.1, format="%f")/100.0

##death_days = st.sidebar.number_input(
##    "Days person remains in critical care or dies", 0, 20, value=6,step=1, format="%f")
##
##crit_lag = st.sidebar.number_input(
##    "Days person takes to go to critical care", 0, 20, value=4 ,step=1, format="%f")
##
##R_0_j=(
##    st.sidebar.number_input("R0", 0.0, 18.0, value=2.3,step=0.1, format="%f"))

hosp_los = st.sidebar.number_input("Hospital Length of Stay", value=5, step=1, format="%i")
icu_los = st.sidebar.number_input("ICU Length of Stay", value=9, step=1, format="%i")
vent_los = st.sidebar.number_input("Ventilator Length of Stay", value=6, step=1, format="%i")

# regional_hosp_share = (
   # st.sidebar.number_input(
       # "Hospital Bed Share (%)", 0.0, 100.0, value=100.0, step=1.0, format="%f")
   # / 100.0
# )

S = st.sidebar.number_input(
  "Regional Population", value=S_default, step=100000, format="%i")

##initial_infections = st.sidebar.number_input(
##    "Currently Known Regional Infections (only used to compute detection rate - does not change projections)", value=known_infections, step=10.0, format="%f")
initial_infections=known_cases
total_infections = current_hosp / regional_hosp_share / hosp_rate
detection_prob = initial_infections / total_infections


#S, I, R = S, initial_infections / detection_prob, 0

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
#doubling_time_t = 1/np.log2(beta*S - gamma +1) # doubling time after distancing

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
gamma2=1/infectious_period
l=0.8
#beta5= R_0_j * ((1/((gamma2)+(alpha)))+((alpha/(+(alpha)))*(l/gamma_hosp)))**(-1)
#beta5= R_0_j * ((1/(gamma2+alpha) )+ ((l/(gamma_hosp))*(alpha/ (gamma2+alpha))))**(-1)
##beta5=(R_0_j * 1/((1/(gamma2+alpha)+alpha/(gamma2+alpha)*l/gamma_hosp)))/S
st.title("COVID-19 Disease Model")


# Slider and Datealpha
n_days = st.slider("Number of days to project", 30, 200, 180, 1, "%i")
as_date = st.checkbox(label="Present result as dates", value=False)


st.header("""Reported Cases, Census and Admissions""")

# Graph of Cases # Lines of cases
# def cases_chart(
    # projection_admits: pd.DataFrame) -> alt.Chart:
    # """docstring"""
    
    # projection_admits = projection_admits.rename(columns={"Admissions": "Census Inpatient", 
                                                            # "ICU":"Census Intensive", 
                                                            # "Ventilated":"Census Ventilated",
                                                            # "New_admits":"New Admissions",
                                                            # "New_discharge":"New Discharges",
                                                            # })
    
    # return(
        # alt
        # .Chart(projection_admits)
        # .transform_fold(fold=["Census Inpatient", 
                                # "Census Intensive", 
                                # "Census Ventilated",
                                # "New Admissions",
                                # "New Discharges"
                                # ])
        # .mark_line(strokeWidth=3, point=True)
        # .encode(
            # x=alt.X("Date", title="Date"),
            # y=alt.Y("value:Q", title="Census"),
            # color="key:N",
            # tooltip=[alt.Tooltip("value:Q", format=".0f"),"key:N"]
        # )
        # .interactive()
    # )

beta_decay = 0.0

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

s_e, e_e, i_e, r_e = sim_seir(S-11, 1 ,10, recovered, beta3, gamma2, alpha, n_days)

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

s_R, e_R, i_R, r_R = sim_seir_decay(S-2, 1 ,1, 0.0, beta4, gamma2,alpha, n_days, decay1, decay2, decay3, decay4, end_delta)

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


####################################################################
#### SEIR model with phase adjusted R_0 and Disease Related Fatality
##
##s_D, e_D, i_D, r_D, d_D = sim_seird_decay(S-2, 1, 1 , 0.0, 0.0, beta4, gamma2,alpha, n_days, decay1, decay2, decay3, decay4, end_delta, fatal)
##
##susceptible_D, exposed_D, infected_D, recovered_D = s_D, e_D, i_D, r_D
##
##i_hospitalized_D, i_icu_D, i_ventilated_D = get_dispositions(i_D, rates, regional_hosp_share)
##
##r_hospitalized_D, r_icu_D, r_ventilated_D = get_dispositions(r_D, rates, regional_hosp_share)
##
##dispositions_D = (
##            i_hospitalized_D + r_hospitalized_D,
##            i_icu_D + r_icu_D,
##            i_ventilated_D + r_ventilated_D)
##
##hospitalized_D, icu_D, ventilated_D = (
##            i_hospitalized_D,
##            i_icu_D,
##            i_ventilated_D)
##
##
### Projection days
##plot_projection_days = n_days - 10



##################################################################
## SEIJR model with phase adjusted R_0 and Disease Related Fatality 

s_D, e_D, i_D, r_D, d_D = sim_seird_decay(S-150, 100.0, 50.0 , 0.0, 0.0, beta4, gamma2,alpha, n_days, decay1, decay2, decay3, decay4, end_delta, fatal)

susceptible_D, exposed_D, infected_D, recovered_D = s_D, e_D, i_D, r_D

i_hospitalized_D, i_icu_D, i_ventilated_D = get_dispositions(i_D, rates, regional_hosp_share)

r_hospitalized_D, r_icu_D, r_ventilated_D = get_dispositions(r_D, rates, regional_hosp_share)
d_hospitalized_D, d_icu_D, d_ventilated_D = get_dispositions(d_D, rates, regional_hosp_share)

dispositions_D = (
            i_hospitalized_D + r_hospitalized_D+d_hospitalized_D,
            i_icu_D + r_icu_D+d_icu_D,
            i_ventilated_D + r_ventilated_D+ d_icu_D)

hospitalized_D, icu_D, ventilated_D = (
            i_hospitalized_D,
            i_icu_D,
            i_ventilated_D)


# Projection days
plot_projection_days = n_days - 10

SEIRD_max=max(hospitalized_D)

########## no social distancing

s_D2, e_D2, i_D2, r_D2, d_D2 = sim_seird_decay(S-150, 100.0, 50.0 , 0.0, 0.0, beta4, gamma2,alpha, n_days, 0, 0, 0, 0,
                                               end_delta, fatal)

susceptible_D2, exposed_D2, infected_D2, recovered_D2 = s_D2, e_D2, i_D2, r_D2

i_hospitalized_D2, i_icu_D2, i_ventilated_D2 = get_dispositions(i_D2, rates, regional_hosp_share)

r_hospitalized_D2, r_icu_D2, r_ventilated_D2 = get_dispositions(r_D2, rates, regional_hosp_share)

d_hospitalized_D2,d_icu_D2,d_ventilated_D2 = get_dispositions(d_D2, rates, regional_hosp_share)
dispositions_D2 = (
            i_hospitalized_D2 + r_hospitalized_D2+d_hospitalized_D2,
            i_icu_D2 + r_icu_D2+d_icu_D2,
            i_ventilated_D2 + r_ventilated_D2, d_ventilated_D2)

hospitalized_D2, icu_D2, ventilated_D2 = (
            i_hospitalized_D2,
            i_icu_D2,
            i_ventilated_D2)

SEIRDno_max=max(hospitalized_D2)

Difference_admissions_height=SEIRDno_max-SEIRD_max





###################################################################
#### SEIJR model with phase adjusted R_0 and Disease Related Fatality
##
##hosp_day_rate=1/hosp_los
##
##s_H2, e_H2, i_H2, j_H2, r_H2, d_H2 = sim_seijcrd_decay2(S-2, 1.0, 1.0, 0.0, 0.0, 0.0, beta5, gamma2,alpha, n_days,decay2,decay3, fatal, fatal_hosp, hosp_day_rate, hosp_rate, l)
##

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

#############
# SEIR Model with phase adjustment and Disease Fatality NO SOCIAL DISTANCING
# New cases
projection_admits_D2 = build_admissions_df(dispositions_D2)
# Census Table
census_table_D2 = build_census_df(projection_admits_D2)


## Confirmed cases graphs

url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv'
df = pd.read_csv(url)
url2='https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
df2=pd.read_csv(url2)
is_US=df2['Country/Region']=='US'
df_US=df2[is_US]

is_NY =  df['Province_State']=='New York'
df_NY = df[is_NY]
is_Erie = df_NY['Admin2']=='Erie'
df_Erie = df_NY[is_Erie]
list(df_NY.columns)
df_NY = df_NY.drop(['UID','iso2','iso3','code3',
 'FIPS','Admin2','Country_Region', 'Province_State','Lat','Long_','Combined_Key','1/22/20','1/23/20','1/24/20','1/25/20','1/26/20','1/27/20',
 '1/28/20','1/29/20','1/30/20','1/31/20','2/1/20','2/2/20','2/3/20','2/4/20','2/5/20','2/6/20','2/7/20',
 '2/8/20','2/9/20','2/10/20','2/11/20','2/12/20','2/13/20','2/14/20','2/15/20','2/16/20','2/17/20','2/18/20',
 '2/19/20','2/20/20','2/21/20','2/22/20','2/23/20','2/24/20','2/25/20','2/26/20', '2/27/20','2/28/20','2/29/20'], axis=1)
df_NY_all=df_NY.agg([sum])

df_Erie = df_Erie.drop(['UID','iso2','iso3','code3',
 'FIPS','Admin2', 'Lat','Long_','Combined_Key','Province_State','1/22/20','1/23/20','1/24/20','1/25/20','1/26/20','1/27/20',
 '1/28/20','1/29/20','1/30/20','1/31/20','2/1/20','2/2/20','2/3/20','2/4/20','2/5/20','2/6/20','2/7/20',
 '2/8/20','2/9/20','2/10/20','2/11/20','2/12/20','2/13/20','2/14/20','2/15/20','2/16/20','2/17/20','2/18/20',
 '2/19/20','2/20/20','2/21/20','2/22/20','2/23/20','2/24/20','2/25/20','2/26/20', '2/27/20','2/28/20','2/29/20'], axis=1)

list(df_US.columns)
df_US = df_US.drop(['Province/State','Lat','Country/Region',
 'Long','1/22/20','1/23/20', '1/24/20','1/25/20','1/26/20','1/27/20','1/28/20', '1/29/20','1/30/20','1/31/20',
 '2/1/20','2/2/20','2/3/20','2/4/20','2/5/20','2/6/20','2/7/20','2/8/20', '2/9/20','2/10/20','2/11/20','2/12/20','2/13/20','2/14/20',
 '2/15/20','2/16/20', '2/17/20', '2/18/20', '2/19/20', '2/20/20', '2/21/20', '2/22/20', '2/23/20', '2/24/20', 
'2/25/20', '2/26/20', '2/27/20','2/28/20','2/29/20'],axis=1)


df_US['Region']='US'
df_Erie['Region']='Erie'
df_NY_all['Region']="NY"

df_US2=pd.melt(df_US, id_vars=['Region'],value_vars=['3/1/20','3/2/20','3/3/20','3/4/20','3/5/20','3/6/20','3/7/20','3/8/20','3/9/20','3/10/20','3/11/20','3/12/20',
 '3/13/20','3/14/20','3/15/20','3/16/20','3/17/20','3/18/20','3/19/20','3/20/20','3/21/20','3/22/20',
 '3/23/20','3/24/20','3/25/20','3/26/20','3/27/20','3/28/20', '3/29/20', '3/30/20','3/31/20','4/1/20',
 '4/2/20','4/3/20','4/4/20','4/5/20','4/6/20','4/7/20','4/8/20', '4/9/20','4/10/20', '4/11/20','4/12/20', '4/13/20', '4/14/20',
                                                     '4/15/20','4/16/20','4/17/20','4/18/20','4/19/20','4/20/20','4/21/20',
                                                     '4/22/20','4/23/20','4/24/20','4/25/20','4/26/20','4/27/20','4/28/20', '4/29/20'],
                        var_name='Date',value_name='US')

df_NY2=pd.melt(df_NY_all, id_vars=['Region'],value_vars=['3/1/20','3/2/20','3/3/20','3/4/20','3/5/20','3/6/20','3/7/20','3/8/20','3/9/20','3/10/20','3/11/20','3/12/20',
 '3/13/20','3/14/20','3/15/20','3/16/20','3/17/20','3/18/20','3/19/20','3/20/20','3/21/20','3/22/20',
 '3/23/20','3/24/20','3/25/20','3/26/20','3/27/20','3/28/20', '3/29/20', '3/30/20','3/31/20','4/1/20',
 '4/2/20','4/3/20','4/4/20','4/5/20','4/6/20','4/7/20','4/8/20', '4/9/20','4/10/20', '4/11/20','4/12/20', '4/13/20', '4/14/20',
                                                     '4/15/20','4/16/20','4/17/20','4/18/20','4/19/20','4/20/20','4/21/20',
                                                     '4/22/20','4/23/20','4/24/20','4/25/20','4/26/20','4/27/20','4/28/20', '4/29/20'],
                        var_name='Day2',value_name='NY')

df_Erie2=pd.melt(df_Erie, id_vars=['Region'],value_vars=['3/1/20','3/2/20','3/3/20','3/4/20','3/5/20','3/6/20','3/7/20','3/8/20','3/9/20','3/10/20','3/11/20','3/12/20',
 '3/13/20','3/14/20','3/15/20','3/16/20','3/17/20','3/18/20','3/19/20','3/20/20','3/21/20','3/22/20',
 '3/23/20','3/24/20','3/25/20','3/26/20','3/27/20','3/28/20', '3/29/20', '3/30/20','3/31/20','4/1/20',
 '4/2/20','4/3/20','4/4/20','4/5/20','4/6/20','4/7/20','4/8/20', '4/9/20','4/10/20', '4/11/20','4/12/20', '4/13/20', '4/14/20',
                                                     '4/15/20','4/16/20','4/17/20','4/18/20','4/19/20','4/20/20','4/21/20',
                                                     '4/22/20','4/23/20','4/24/20','4/25/20','4/26/20','4/27/20','4/28/20', '4/29/20'],
                        var_name='Day3',value_name='Erie')

df_US2=df_US2.drop(['Region'], axis=1)
df_NY2=df_NY2.drop(['Region'], axis=1)
df_Erie2=df_Erie2.drop(['Region'], axis=1)

result = pd.concat([df_US2, df_NY2, df_Erie2], axis=1, sort=False)
result=result.drop(['Day2', 'Day3'],axis=1)
result['day'] = np.arange(len(result))
result['US']=(result['US']/328000000)*100
result['NY']=(result['NY']/19450000)*100
result['Erie']=(result['Erie']/1500000)*100
#st.dataframe(result)

st.subheader("""Confirmed Cases for the US, New York, and Erie County""")
# Erie Graph of Cases # Lines of cases
def confirmed_chart(
    projection: pd.DataFrame,
    as_date:bool = False) -> alt.Chart:
    """docstring"""
    
    tooltip_dict = {False: "day", True: "Date:T"}
    if as_date:
        #projection = add_date_column(projection)
        x_kwargs = {"shorthand": "Date:T", "title": "Date"}
    else:
        x_kwargs = {"shorthand": "day", "title": "Days from initial infection"}
    
    return(
        alt
        .Chart(projection)
        .transform_fold(fold=["US", 
                               "NY", 
                                "Erie",
                                ])
        .mark_line(strokeWidth=2, point=True)
        .encode(
            x=alt.X(**x_kwargs),
            y=alt.Y("value:Q", title="% Confirmed Cases Per Region Population"),
            color="key:N",
            tooltip=[
                tooltip_dict[as_date],
                alt.Tooltip("value:Q", format=".0f", title="COVID-19 Cases"),
                "key:N",
            ],
        )
        .interactive()
    )

 # Bar chart of Erie cases with layer of HERDS DAta Erie
st.altair_chart(confirmed_chart(result, as_date=as_date), use_container_width=True)

st.markdown(
    """This chart shows the percent daily [confirmed](https://coronavirus.jhu.edu/map.html) cases per region population. These numbers are highly influenced by testing rates and testing practices in each geographic location. """
)
#cols = [2,4]
#result.drop(result.columns[cols],axis=1,inplace=True)








st.header("""Model Projections""")
# Admissions Graphs
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


# Comparison of Single line graph - Hospitalized, ICU, Vent and All
# if model_options == "Inpatient":
    # columns_comp = {"hosp": "Hospitalized"}
    # fold_comp = ["Hospitalized"]
    # capacity_col = {"total_county_beds":"Inpatient Beds"}
    # capacity_fol = ["Inpatient Beds"]
# if model_options == "ICU":
    # columns_comp = {"icu": "ICU"}
    # fold_comp = ["ICU"]
    # capacity_col = {"total_county_icu": "ICU Beds"}
    # capacity_fol = ["ICU Beds"]
# if model_options == "Ventilated":
    # columns_comp = {"vent": "Ventilated"}
    # fold_comp = ["Ventilated"]
    # capacity_col = {"expanded_vent_beds": "Expanded Ventilators (50%)", "expanded_vent_beds2": "Expanded Ventilators (100%)"}
    # capacity_fol = ["Expanded Ventilators (50%)", "Expanded Ventilators (100%)"]

# def ip_chart(
    # projection_admits: pd.DataFrame, 
    # plot_projection_days: int,
    # as_date:bool = False) -> alt.Chart:
    # """docstring"""
    
    # projection_admits = projection_admits.rename(columns=columns_comp)
    
    # tooltip_dict = {False: "day", True: "date:T"}
    # if as_date:
        # projection_admits = add_date_column(projection_admits)
        # x_kwargs = {"shorthand": "date:T", "title": "Date"}
    # else:
        # x_kwargs = {"shorthand": "day", "title": "Days from initial infection"}
    
    # return (
        # alt
        # .Chart(projection_admits.head(plot_projection_days))
        # .transform_fold(fold=fold_comp+capacity_fol)
        # .mark_line(point=False)
        # .encode(
            # x=alt.X(**x_kwargs),
            # y=alt.Y("value:Q", title="Daily admissions"),
            # color="key:N",
            # tooltip=[
                # tooltip_dict[as_date],
                # alt.Tooltip("value:Q", format=".0f", title="Admissions"),
                # "key:N",
            # ],
        # )
        # .interactive()
    # )



#, scale=alt.Scale(domain=[0, 100])
# alt.value('orange')


###################### Vertical Lines Graph ###################
# Schools 18th
# Non-essential business 22nd
vertical = pd.DataFrame({'day': [int1_delta, int2_delta, end_delta]})

def vertical_chart(
    projection_admits: pd.DataFrame, 
    as_date:bool = False) -> alt.Chart:
    """docstring"""
    
    tooltip_dict = {False: "day", True: "date:T"}
    if as_date:
        projection_admits = add_date_column(projection_admits)
        x_kwargs = {"shorthand": "date:T", "title": "Date"}
    else:
        x_kwargs = {"shorthand": "day", "title": "Days from initial infection"}
    
    return (
        alt
        .Chart(projection_admits)
        .mark_rule(color='gray')
        .encode(
            x=alt.X(**x_kwargs),
            tooltip=[
                tooltip_dict[as_date]],
        )
    )

vertical1 = vertical_chart(vertical, as_date=as_date)



##############################
#4/3/20 First Projection Graph - Admissions
#############
st.subheader("Projected number of **daily** COVID-19 admissions")
admits_graph = regional_admissions_chart(projection_admits_D, 
        plot_projection_days, 
        as_date=as_date)

st.altair_chart(admits_graph, use_container_width=True)

#+ vertical1,

st.markdown(
    """This model shows the number of daily admissions projected for the chosen time period. """
)

#st.dataframe(projection_admits)
if st.checkbox("Show more info about the model specification and assumptions"):
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
    """where $\gamma$ is $1/mean\ infectious\ rate$, $$f$$ is the fatality rate, $$\\alpha$$ is $1/mean\ incubation\ period$, $$\\rho$$ is one minus the rate of social distancing at time $t$,
and $$\\beta$$ is the rate of transmission.

Note that a number of assumptions are made with deterministic compartmental models. First, we are assuming a large, closed population with no births or deaths.
Second, within the time period, immunity to the disease is acquired. Third, the susceptible and infected subpopulations are dispersed homogeneously in geographic space.
In addition to the model assumptions noted here, the model is limited by uncertainty related to parameter choice.
Parameters are measured independently from the model, which is hard to do in the midst of an outbreak.
Early reports from other geographic locations have allowed us to estimate this model.
However, parameters can be different depending on population characteristics and can vary over periods of the outbreak.
Therefore, interpreting the results can be difficult. """)


sir = regional_admissions_chart(projection_admits, plot_projection_days, as_date=as_date)
seir = regional_admissions_chart(projection_admits_e, plot_projection_days, as_date=as_date)
seir_r = regional_admissions_chart(projection_admits_R, plot_projection_days, as_date=as_date)
seir_d = regional_admissions_chart(projection_admits_D, plot_projection_days, as_date=as_date)
seir_d2 = regional_admissions_chart(projection_admits_D2, plot_projection_days, as_date=as_date)

Max_hosp_admissions=max(projection_admits_D['hosp'].dropna())
Max_hosp_admissions_nosoc=max(projection_admits_D2['hosp'].dropna())
Max_diff=Max_hosp_admissions_nosoc-Max_hosp_admissions


st.subheader("Projected number of **daily** COVID-19 admissions: Model Comparison (Left: 0% Social Distancing, Right: Step-Wise Social Distancing)")
st.altair_chart(
    alt.layer(seir_d2.mark_line())
    + alt.layer(seir_d.mark_point())
    + alt.layer(vertical1.mark_rule())
    , use_container_width=True)

st.markdown(
    """In the above graph, the curves to the left (indicated by the solid lines) represent projections if government implemented
social distancing (e.g. New York State on PAUSE) had not gone into effect. The second set of curves (denoted by points) represent projections with social distancing.
The percent of social distancing can be chosen by the user."""
    )

st.markdown(
    """Compared to a projection with no social distancing (letting the virus run it's natural course with no government shut-down),
    there are **{Max_diff:.0f}** fewer admissions (daily hospitalizations) at the peak of the epidemic curve. Therefore,
    we are flattening the curve.""".format(
        Max_diff=Max_diff
    ))



################################################
################################################
#############    Census Graphs        ##########
################################################
################################################
st.header("""Projected Census Models""")

# def ip_census_chart(
    # census: pd.DataFrame,
    # plot_projection_days: int,
    # as_date:bool = False) -> alt.Chart:
    # """docstring"""
    # census = census.rename(columns=columns_comp)

    # tooltip_dict = {False: "day", True: "date:T"}
    # if as_date:
        # census = add_date_column(census.head(plot_projection_days))
        # x_kwargs = {"shorthand": "date:T", "title": "Date"}
    # else:
        # x_kwargs = {"shorthand": "day", "title": "Days from initial infection"}

    # return (
        # alt
        # .Chart(census)
        # .transform_fold(fold=fold_comp)
        # .mark_line(point=False)
        # .encode(
            # x=alt.X(**x_kwargs),
            # y=alt.Y("value:Q", title="Census"),
            # color="key:N",
            # tooltip=[
                # tooltip_dict[as_date],
                # alt.Tooltip("value:Q", format=".0f", title="Census"),
                # "key:N",
            # ],
        # )
        # .interactive()
    # )




def admitted_patients_chart(
    census: pd.DataFrame,
    plot_projection_days: int,
    as_date=False) -> alt.Chart:
    """docstring"""
    
    census = census.rename(columns={"hosp": "Hospital Census", 
        "icu": "ICU Census", 
        "vent": "Ventilated Census"})
    
    tooltip_dict = {False: "day", True: "date:T"}
    if as_date:
        census = add_date_column(census)
        x_kwargs = {"shorthand": "date:T", "title": "Date"}
    else:
        x_kwargs = {"shorthand": "day", "title": "Days from initial infection"}
    
    return (
        alt
        .Chart(census.head(plot_projection_days))
        .transform_fold(fold=["Hospital Census", "ICU Census", "Ventilated Census"])
        .mark_line(point=False)
        .encode(
            x=alt.X(**x_kwargs),
            y=alt.Y("value:Q", title="Census"),
            color="key:N",
            tooltip=["day", "key:N"]
        )
        .interactive()
    )

#SEIR w/ adjusted R_0 and deaths
st.altair_chart(admitted_patients_chart(census_table_D, 
    plot_projection_days, 
    as_date=as_date),
    use_container_width=True)


# Version with single line
#sir_ip_c = ip_census_chart(census_table, plot_projection_days, as_date=as_date)
#seir_ip_c = ip_census_chart(census_table_e, plot_projection_days, as_date=as_date)
#seir_r_ip_c = ip_census_chart(census_table_R, plot_projection_days, as_date=as_date)
#seir_d_ip_c = ip_census_chart(census_table_D, plot_projection_days, as_date=as_date)
###

# Version with single line
# st.altair_chart(
    # alt.layer(seir_d_ip_c.mark_line())
    # #+ alt.layer(graph_selection)
 # #   +alt.layer(seir_ip_c.mark_line())
    # + alt.layer(vertical1)
    # , use_container_width=True)

st.markdown(
    """This model shows the daily census for projected occupied hospital beds. """
)

################# Add 0% 10% 20% SD graph of SEIR MODEL ###################

    #, scale=alt.Scale(domain=[0, 40000])
    # scale=alt.Scale(domain=[-5, 9000])
    


#sir_ip_c = ip_census_chart(census_table, plot_projection_days, as_date=as_date)
#seir_ip_c = ip_census_chart(census_table_e, plot_projection_days, as_date=as_date)
#seir_r_ip_c = ip_census_chart(census_table_R, plot_projection_days, as_date=as_date)
#seir_d_ip_c = ip_census_chart(census_table_D, plot_projection_days, as_date=as_date)
###
# Added SEIR 10, 20 SD
#seir_ip_c10 = ip_census_chart(census_table_e10, plot_projection_days, as_date=as_date)
#seir_ip_c20 = ip_census_chart(census_table_e20, plot_projection_days, as_date=as_date)


##########################################################
##########################################################
###########            PPE            ####################
##########################################################
##########################################################
#st.header("Projected PPE Needs")
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
    
### SEIR Model with adjusted R_0 with Case Fatality - PPE predictions
##st.subheader("Projected personal protective equipment needs for mild and severe cases of COVID-19: SEIR Model with Adjutes R_0 and Case Fatality")
##
##ppe_graph = ppe_chart(census_table_D, as_date=as_date)
##
##st.altair_chart(alt.layer(ppe_graph.mark_line()) + alt.layer(vertical1), use_container_width=True)
##
# Recovered/Infected/Fatality table
st.header("Projected infected and fatal individuals in the region across time")

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



##st.markdown("""There is a projected number of **{infection_total_t:.0f}** infections due to COVID-19.""".format(
##        infection_total_t=infection_total_t
##    )
##            )
AAA=beta4*(1/gamma2)*S
R2=AAA*(1-decay2)
R3=AAA*(1-decay3)
R4=AAA*(1-decay4)

st.markdown("""The initial $R_0$ is **{AAA:.1f}** the $R_e$ after 2 weeks is **{R2:.1f}** and the $R_e$ after 3 weeks to end of social distancing is **{R3:.1f}**.
After reducing social distancing, the $R_e$ is **{R4:.1f}**.
            This is based on a doubling rate of **{doubling_time:.0f}** and the calculation of the [basic reproduction number](https://www.sciencedirect.com/science/article/pii/S2468042719300491).  """.format(
        AAA=AAA,
        R2=R2,
        R3=R3,
        R4=R4,
        doubling_time=doubling_time
    )
            )



### Recovered/Infected/Hospitalized/Fatality table
##st.header("Estimating Hospitalization within the model")
##st.subheader("The number of infected,recovered, and fatal individuals in the region at any given moment")
##
##def additional_projections_chart(e:np.ndarray, i: np.ndarray, j: np.ndarray, d: np.ndarray) -> alt.Chart:
##    dat = pd.DataFrame({"Exposed":e,"Infected": i, "Hospitalized":j, "Fatal":d})
##
##    return (
##        alt
##        .Chart(dat.reset_index())
##        .transform_fold(fold=["Exposed","Infected", "Hospitalized", "Fatal"])
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
##st.altair_chart(additional_projections_chart(j_H2, e_H2, i_H2, d_H2), use_container_width=True)
##
##

st.subheader("Acknowledgments")
st.markdown("""The SEIR model and application were developed by the University at Buffalo's Clinical Informatics lab in the
[Department of Biomedical Informatics](http://medicine.buffalo.edu/departments/biomedical-informatics.html) (Gabriel Anaya, Sarah Mullin,
Jinwei Hu, Brianne Mackenzie, Arlen Brickman, and [Peter Elkin](http://medicine.buffalo.edu/faculty/profile.html?ubit=elkinp)) with collaboration from [Matthew Bonner](http://sphhp.buffalo.edu/epidemiology-and-environmental-health/faculty-and-staff/faculty-directory/mrbonner.html) in the Department of Epidemiology and Environmental Health, [Greg Wilding](http://sphhp.buffalo.edu/biostatistics/faculty-and-staff/faculty-directory/gwilding.html) in the Department of Biostatistics, and [Great Lakes Healthcare](https://www.greatlakeshealth.com) with [Peter Winkelstein](http://medicine.buffalo.edu/faculty/profile.html?ubit=pwink). 
            Building off of the core application from the [CHIME model](https://github.com/CodeForPhilly/chime/), our model adds compartments for _Exposed_ and _Death_ and creates a step-wise social distancing adjusted model.
            Documentation of parameter choices and model choices can be found in the department github Wiki.  For questions, please email [Gabriel Anaya](ganaya@buffalo.edu) or [Sarah Mullin](sarahmul@buffalo.edu).  """)

st.markdown(
    """This work has been supported in part by grants from NIH NLM T15LM012495, NIAA R21AA026954, and NCATS UL1TR001412. This study was funded in part by the Department of Veterans Affairs."""
)
