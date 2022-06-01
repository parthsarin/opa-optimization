import pandas as pd
import numpy as np
import seaborn as sns
from bisect import bisect
import matplotlib.pyplot as plt
from functools import cache

##################
# DO NOTHING
##################

# print('SCENARIO: DO NOTHING')
# downpayment = 0.20     # percent of price
# interest = 0.055       # percent of price
# mortgage_duration = 30 # years
# rofr = 0               # percent reduction in sales price
# spillover_amt = 0      # percent of sales that are affected by spillover
# spillover_mag = 0      # reduction of sale price due to spillover
# off_market = 0.25      # percent of sales off markett
# dti = 0.36             # debt-to-income ratio that's required to get the mortgage approved
# ami = 149_600          # average median income
# sfh = 1                # include single family homes in OPA? 1 = yes, 0 = no


##################
# FULL OPA
##################

# print('SCENARIO: FULL OPA')
# downpayment = 0.20     # percent of price
# interest = 0.055       # percent of price
# mortgage_duration = 30 # years
# rofr = 0.1             # percent reduction in sales price
# spillover_amt = 0.4    # percent of sales that are affected by spillover
# spillover_mag = 0.1    # reduction of sale price due to spillover
# off_market = 0.1       # percent of sales off markett
# dti = 0.36             # debt-to-income ratio that's required to get the mortgage approved
# ami = 149_600          # average median income
# sfh = 1                # include single family homes in OPA? 1 = yes, 0 = no


##################
# OPA wo ROFR
##################

# print('SCENARIO: OPA WITHOUT ROFR')
# downpayment = 0.20     # percent of price
# interest = 0.055       # percent of price
# mortgage_duration = 30 # years
# rofr = 0               # percent reduction in sales price
# spillover_amt = 0.2    # percent of sales that are affected by spillover
# spillover_mag = 0.05   # reduction of sale price due to spillover
# off_market = 0.1       # percent of sales off markett
# dti = 0.36             # debt-to-income ratio that's required to get the mortgage approved
# ami = 149_600          # average median income
# sfh = 1                # include single family homes in OPA? 1 = yes, 0 = no

##################
# OPA WITH MODIFIED ELIGIBILITY
##################

print('SCENARIO: OPA WITH MODIFIED ELIGIBILITY')
downpayment = 0.20     # percent of price
interest = 0.055       # percent of price
mortgage_duration = 30 # years
rofr = 0.1             # percent reduction in sales price
spillover_amt = 0.05   # percent of sales that are affected by spillover
spillover_mag = 0.1    # reduction of sale price due to spillover
off_market = 0.1       # percent of sales off markett
dti = 0.36             # debt-to-income ratio that's required to get the mortgage approved
ami = 149_600          # average median income
sfh = 0                # include single family homes in OPA? 1 = yes, 0 = no


df = pd.read_csv('final_results_1000.csv')
df.columns = [c.lower().replace(' ', '_') for c in df.columns]

n = len(df)
sim_ids = set(df.simulation)
num_sims = len(sim_ids)
num_per_sim = n / num_sims


##################
# COMPUTATION
##################
def get_category_coarse(x):
    idx = bisect([80, 160], x)
    cats = ['Low Income', 'Middle Income', 'Upper Income']
    return cats[idx]


df['adj_price'] = df.price
df['adj_price'] = df.adj_price - (np.random.random(n) < spillover_amt) * spillover_mag * df.price

df['p_ami'] = df.income / ami * 100
df['mortgage'] = (1 - downpayment) * ((1 + interest) ** mortgage_duration) * (1 - rofr) * df.adj_price / mortgage_duration
df['viable'] = (df.mortgage / df.income <= dti) 

# 0 = not eligible, 1 = still eligible
df['eligible_market'] = 1 - (np.random.random(n) < off_market)
df['eligible_sfh'] = np.ones(n) if sfh == 1 else (df.property_indicator != 10) * 1

df['viable_eligible'] = df.eligible_market * df.eligible_sfh * df.viable
df['income_category'] = df.p_ami.apply(get_category_coarse)


##################
# OPTIMIZATION
##################
def count_sum_to_thresh_thanos(a, thresh, full_df):
    for i in range(len(a)):
        if sum(a[:i]) > thresh:
            break
    num_helped = i - 1

    return num_helped, full_df.iloc[:num_helped]


def thanos(funding):
    potential = df[df.eligible_market * df.eligible_sfh == 1]
    potential = potential[potential.viable_eligible == 0]
    potential['gap'] = potential.mortgage / dti - potential.income

    num_helped = []
    people_helped = []

    for sid in sim_ids:
        x = potential[potential.simulation == sid]
        x = x.sample(frac=0.5)  # thanos
        x = x.sort_values('gap')
        a = x.gap
        a = a / 0.63  # https://web.archive.org/web/20200321115234id_/https://www.innovations.harvard.edu/sites/default/files/hpd_0202_stegman.pdf

        nh, ph = count_sum_to_thresh_thanos(a, funding, x)

        num_helped.append(nh)
        people_helped.append(ph)

    num_helped = np.array(num_helped)
    people_helped = pd.concat(people_helped)

    return num_helped, people_helped


##################
# RESULTS
##################
def perc_low_income(people_helped, full_df):
    try:
        num_low_income_helped = people_helped.income_category.value_counts()['Low Income']
    except KeyError:
        num_low_income_helped = 0
        
    num_low_income_afford = sum((full_df.viable_eligible == 1) & (full_df.income_category == 'Low Income'))
    total_low_income = sum(full_df.income_category == 'Low Income')
    return (num_low_income_afford + num_low_income_helped) / total_low_income


def perc_poc(people_helped, full_df):
    try:
        num_poc_helped = len(people_helped) - people_helped.race.value_counts()['White']
    except KeyError:
        num_poc_helped = 0
    
    num_poc_afford = sum((full_df.viable_eligible == 1) &
                         (full_df.race != 'White'))
    total_poc = sum(full_df.race != 'White')
    return (num_poc_afford + num_poc_helped) / total_poc


funding_options = [0, 2_000_000, 5_000_000, 10_000_000, 20_000_000, 50_000_000]
for f in funding_options:
    nh, ph = thanos(f)
    nh_baseline = df.groupby('simulation').viable_eligible.sum().mean()
    print(nh.mean() + nh_baseline, end=', ')
