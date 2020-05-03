# Import packages
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import networkx as nx
import pandas as pd
import numpy as np
import scipy as sp
import datetime as dt
import community
from shapely.geometry import Polygon
import importlib

import DDOM # Data Driven Occupational Mobility
import model_cal

import random
import math

import simpy

import cmocean as cmo


def cal_algo(params):
    parameters = params[0]
    empirical_data = params[1]
    # output variable
    calibration_output = True

    # Read the data from data_processing.ipynb
    sa_calibration_data = pd.read_csv('../Data_Labour/calibration_data.csv')
    employment_SSYK = pd.read_csv('../Data_Labour/occupational_employment.csv', sep = ',')
    SSYK_shock = pd.read_csv('../Data_Labour/occupation_shock.csv', sep = ',')

    G = nx.read_graphml('../Data_Labour/Occ_mob_sweden.graphml')

    e_vac_rate = empirical_data['sa_vac_rate']
    e_unemployed = empirical_data['u_trend']
    e_seq = [(u, e_vac_rate.iloc[i]) for i, u in enumerate(e_unemployed)]

    # Loop should give result = {dict with parameters:dict with outputs}
    results = []
    no_sim = len(parameters['timestep'])*len(parameters['delta_u'])*len(parameters['delta_ny'])*len(parameters['gamma_u'])*len(parameters['a'])
    print('Number of simulations -ish:', no_sim)
    simulation_index = 0
    for timestep in parameters['timestep']:
        for delta_u in parameters['delta_u']:
            for delta_ny in parameters['delta_ny']:
                for gamma_u in parameters['gamma_u']:
                    for a in parameters['a']: 
                        if delta_u <= delta_ny:
                            continue
                        print('Simulation index:', simulation_index)
                        print('a:', a, 'delta_u:', delta_u, 'delta_ny:', delta_ny, 'gamma_u:', gamma_u, 'timestep:', timestep)
                        employment = employment_SSYK[['SSYK', '2014']]
                        employment = {str(employment['SSYK'].iloc[i]):employment['2014'].iloc[i] for i in range(len(employment))}
                        node_names = list(G.nodes())

                        # setup network
                        employed = {str(name):e for name,e in employment.items() if str(name) in node_names}
                        unemployed = {name:0 for name in node_names}
                        vacancies = {name:0 for name in node_names}
                        applications = {name:[] for name in node_names}
                        target_demand = {str(name):e for name,e in employment.items() if str(name) in node_names}
                        of_data = SSYK_shock.groupby(by = ['ssyk3'], axis = 0).mean()
                        of_data = of_data.to_dict()['Computerisation Probability']
                        attributes = {'employed':employed, 'unemployed':unemployed, 'vacancies':vacancies, 'applications':applications,
                        'target_demand':target_demand, 'comp_prob':of_data}

                        # Save parameters
                        out_parameters = {'a':a, 'delta_u':delta_u, 'delta_ny':delta_ny, 'gamma_u':gamma_u, 'timestep':timestep}

                        # Run simulation
                        output = DDOM.deterministic_simulation(G, parameters['years'], timestep, delta_u, gamma_u, delta_ny, gamma_u, empirical_data,
                        parameters['t_0'], parameters['k'], parameters['L'], parameters['avg_hours_0'], a, parameters['T'], parameters['shock_start'],
                        attributes, calibration_output)
                        

                        print('cost:', output['cost'], 'A_e:', output['A_e'], 'A_m:', output['A_m'])

                        for key, val in out_parameters.items():
                            output[key] = val

                        results.append(output)
                        simulation_index += 1

