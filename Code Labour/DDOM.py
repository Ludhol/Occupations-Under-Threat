# Required packages (check which are required)
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import networkx as nx
import pandas as pd
import numpy as np
import scipy as sp
import datetime as dt
import community
from shapely.geometry import Polygon

import random
import math

import cmocean as cmo


# setup network
def set_attributes(G, data):
    '''
    Initialises the node attributes of the occupational mobility network.
    Parameters:
    -----------
    G (nx.graph): The occupational mobility network
    data (?): Data used to specify attribute values
    
    setting up data will have the following form:
    data (pd.DataFrame): Columns contain values for the attributes and rows are occupations

    The attributes are:
    employed (int): Number of employees in the occupation
    unemplyed (int): Number of unemployed in the occupation
    vacancies (int): Number of vacancies in the occupation
    applications (list): List where the elements are the name of the occupation where the application is coming from
    target_demand (int): The target demand of the occupation (exogenous and used for automation shock)
    risk_factor (float): A value between 0 and 1 indicating the risk of a proffession (could be automation or corona or whatever)

    Note that the current demand is implicit (=employed + vacancies)
    '''
    # Make dictionaries
    attributes = data.to_dict()

    # Add some variables
    # attributes['target_demand']
    
    # Set attrbutes
    for key, value in attributes.items():

        nx.set_node_attributes(G, value, str(key))

def make_vacancies(G, delta_ny, gamma_ny):
    '''
    Each occupation has a probability of making vacancies each time step that depends on the difference between
    current demand and target demand as well as an exogenous probability 

    Parameters:
    -----------
    G (nx.graph): The occupational mobility network
    other parameters
    '''
    # The attributes we need to calculate new vacancies
    vac_dict = nx.get_node_attributes(G, 'vacancies')
    emp_dict = nx.get_node_attributes(G, 'employed')
    td_dict = nx.get_node_attributes(G, 'target_demand')

    # loop through all the occupations and create vacancies as described in the paper
    for occupation in vac_dict.keys():
        # The occupation specificattribute values
        target_demand = td_dict[occupation]
        employed = emp_dict[occupation]
        current_demand = vac_dict[occupation] + employed
        

        # Calculate the probability of creating new vacancies (equation 10)
        demand_diff = max(0, target_demand - current_demand)

        # Zero Steady State rule
        if employed == vac_dict[occupation] == 0 and target_demand > 0:
            vacancies += 1
        else:
            if demand_diff == 0 or employed == 0:
                alpha_ny = 0
            else:
                alpha_ny = gamma_ny*demand_diff/employed

            p_ny = delta_ny + alpha_ny - delta_ny*alpha_ny

            # Expected number of new vacancies - wrong
            # new_vacancies = round(p_ny*employed
            new_vacancies = np.random.binomial(employed, p_ny)
            # Add the new vacancies to total number of vacancies
            vac_dict[occupation] += new_vacancies

    nx.set_node_attributes(G, vac_dict, 'vacancies')


def separate_workers(G, delta_u, gamma_u):
    '''
    Each occupation has a probability of seperating workers each time step that depends on the difference between
    current demand and target demand as well as an exogenous probability 

    Parameters:
    -----------
    G (nx.graph): The occupational mobility network
    other parameters
    '''

    # The attributes we need to calculate number of seperated workers
    unemp_dict = nx.get_node_attributes(G, 'unemployed')
    emp_dict = nx.get_node_attributes(G, 'employed')
    td_dict = nx.get_node_attributes(G, 'target_demand')
    vac_dict = nx.get_node_attributes(G, 'vacancies')

    # loop through all the occupations and seperate workers as described in the paper
    for occupation in unemp_dict.keys():
        # The occupation specificattribute values
        target_demand = td_dict[occupation]
        employed = emp_dict[occupation]
        vacancies = vac_dict[occupation]

        current_demand = vacancies + employed

        # Calculate the probability of creating new vacancies (equation 9)
        demand_diff = max(0, current_demand - target_demand)
        if demand_diff == 0 or employed == 0:
            alpha_u = 0
        else:
            alpha_u = gamma_u*demand_diff/employed
        # probability calculation (equation 11)
        p_u =  delta_u + alpha_u - delta_u*alpha_u
        

        # This is the expected number of seperated workers - WRONG!!!
        # separated = round(p_u*unemployed)
        separated = np.random.binomial(employed, p_u)

        unemp_dict[occupation] += separated
        emp_dict[occupation] -= separated
    nx.set_node_attributes(G, unemp_dict, 'unemployed')
    nx.set_node_attributes(G, emp_dict, 'employed')

def make_applications(G):
    '''
    Each unemployed worker send out an application to a neighbouring occupation proportional to the edge weight
    between the occupations

    Parameters:
    -----------
    G (nx.graph): The occupational mobility network
    '''
    # loop through every occupation and every unemployed person in the occupation and add applications in the 
    # applications attribute of the occupation that the worker applies to
    app_dict = nx.get_node_attributes(G, 'applications')
    unem_dict = nx.get_node_attributes(G, 'unemployed')
    vac_dict = nx.get_node_attributes(G, 'vacancies')

    A_dict = nx.get_edge_attributes(G, 'weight')

    for occupation in app_dict:
        unemployed = unem_dict[occupation]
        neighbors = [n for n in G.neighbors(occupation)]

        # Calculate probabilities (equation 7)
        vac_A_list = [vac_dict[n]* A_dict[(occupation, n)] for n in neighbors]
        vac_A_sum = sum(vac_A_list)
        if vac_A_sum == 0:
            continue
        p_list = [vac_A/vac_A_sum for vac_A in vac_A_list]

        # Each unemployed person chooses an occupation to apply to
        for _ in range(unemployed):
            # Choice is weighted by probabilities calculated above
            apply_to = np.random.choice(neighbors, p = p_list)
            # Make the application
            app_dict[apply_to].append(occupation)
    nx.set_node_attributes(G, app_dict, 'applications')


def handle_applications(G):
    '''
    Randomly accept N applications in the application attrbute where N is the number of vacancies in the occupation

    Parameters:
    -----------
    G (nx.graph): The occupational mobility network
    '''
    # The attributes we need
    app_dict = nx.get_node_attributes(G, 'applications')
    vac_dict = nx.get_node_attributes(G, 'vacancies')
    emp_dict = nx.get_node_attributes(G, 'employed')
    unemp_dict = nx.get_node_attributes(G, 'unemployed')

    # loop through all the occupations and try to accept the same number of applications as there are vacancies
    for occupation in app_dict:
        applications = app_dict[occupation]
        vacancies = vac_dict[occupation]

        filled_vacancies = 0
        for j in range(vacancies):
            if len(applications) == 0:
                # if vacancies > len(applications) not all vacancies are filled
                break
            # A random application is chosen (and removed from the list of applications)
            accepted = applications.pop(random.randrange(len(applications)))

            # Increase the amount of employed people in the occupation by the number of filled vacancies
            emp_dict[occupation] += 1

            # Decrease the amount of unemployed people in the occupations where applications were accepted from
            unemp_dict[accepted] -= 1
            # Number of filled vacancies
            filled_vacancies = j + 1
    
        # Decrease the amount of vacancies in the occupation
        vac_dict[occupation] = vacancies - filled_vacancies
        # Update the applications list
        app_dict[occupation] = applications
    
    # Update the network node attributes
    nx.set_node_attributes(G, app_dict, 'applications')
    nx.set_node_attributes(G, vac_dict, 'vacancies')
    nx.set_node_attributes(G, emp_dict, 'employed')
    nx.set_node_attributes(G, unemp_dict, 'unemployed')

def update_target_demand(G, demand_0, t, T, a):
    '''
    function that updates the target demand. Used for calibrating the the model paramters

    Parameters:
    -----------
    G (nx.Graph): The occupational mobility network
    demand_0 (int): Initial demand
    t (int): Timestep of simulation
    T (float): The duration of a business cycle
    a (float): the amplitude of the business cycle sin wave
    '''

    t = t - T

    target_demand = {}
    for node in G.nodes():
        target_demand[node] = round(demand_0[node]*(1 - a*np.sin(t*2*np.pi/T)))

    nx.set_node_attributes(G, target_demand, 'target_demand')


def shock(G, demand_0, final_demand, t, t_0, k):
    '''
    function that implements the labour demand shock

    Parameters:
    -----------
    G (nx.Graph): The occupational mobility network
    demand_0 (int): Initial demand
    t (int): Timestep of simulation
    t_0 (int): Midpoint of shock (where sigmoid goes from exponential increase to exponential decrease)
    k (float): 

    '''
    
    employed = nx.get_node_attributes(G, 'employed')
    demand_shock = {}
    target_demand = {}

    for key in employed.keys():
        demand_shock[key] = (final_demand[key] - demand_0[key])/(1 + np.exp(k*(t-t_0)))
        target_demand[key] = demand_0[key] + demand_shock[key]

    nx.set_node_attributes(G, target_demand, 'target demand')


def calibration_calculation(empirical_data, model_data, A_e, period):
    '''
    This is the function to be minimised during calibration.abs

    Parameters:
    empirical_data (dict): timeseries of vacancies and unemployment
    model_data (dict): timeseries of vacancies and unemployment
    '''

    start = round(period*1.5)
    end = -1

    m_vacancies = [sum(model_data['vacancies'][i].values()) for i in range(len(model_data['vacancies']))]
    m_employed = [sum(model_data['employment'][i].values()) for i in range(len(model_data['employment']))]


    m_vac_rate = [m_vacancies[i]*100/(m_vacancies[i] + e) for i, e in enumerate(m_employed)]

    m_unemployed = [sum(model_data['unemployment'][i].values()) for i in range(len(model_data['unemployment']))]
    m_unemployed = [u*100/(m_employed[i]+ u) for i, u in enumerate(m_unemployed)]

   
    fig, ax = plt.subplots()
    cols = ["#69b243", "#ad5ec7"]
    # ax_bounds = [4, 11, 0.2, 4]
    # ax.axis(ax_bounds)
    # ax.xaxis.set_ticks(np.arange(ax_bounds[0], ax_bounds[1], 1))
    # ax.yaxis.set_ticks(np.arange(ax_bounds[2], ax_bounds[3], 0.5))
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax.set_title("Simulated & Empirical Beveridge Curve", fontsize=16)
    ax.set_xlabel('Unemployment (%)')
    ax.set_ylabel('Vacancy rate (%)')



    plt.plot(empirical_data['u_trend'], empirical_data['sa_vac_rate'], color = cols[1], ls = '-', marker = 'o', linewidth = 1, markersize = 2)
    plt.plot(m_unemployed, m_vac_rate, color = cols[0], ls = '-', marker = 'o', linewidth = 1, markersize = 2)
    # plt.savefig('../Graphs/Beveridge_curve.pdf', dpi=425, bbox_inches='tight')
    plt.show()


    m_vac_rate = m_vac_rate[start:end]
    m_unemployed = m_unemployed[start:end]

    vac_max = np.max(m_vac_rate)
    vac_min = np.min(m_vac_rate)
    u_max = np.max(m_unemployed)
    u_min = np.min(m_unemployed)
    m_seq = [(u, m_vac_rate[i]) for i, u in enumerate(m_unemployed)]

    if Polygon(m_seq).is_valid == True:
        A_m = Polygon(m_seq).buffer(0)
        m_area = A_m.area
    else:
        print('A_m is not valid')
        m_area = 'N/A'
        cost = 'N/A'
        intersection_area = 'N/A'
        union_area = 'N/A'
        return {'cost':cost, 'intersection':intersection_area, 'union':union_area, 'A_m': m_area, 'm_u_max':u_max, 'm_u_min':u_min, 'm_vac_max':vac_max, 'm_vac_min':vac_min}
    try:
        union_area = A_m.union(A_e).area
    except:
        print('union is not found')
        union_area = 'N/A'
        cost = 'N/A'
    try:
        intersection_area = A_m.intersection(A_e).area
    except:
        print('intersection is not found')
        intersection_area = 'N/A'
        cost = 'N/A'
    try:
        if A_m.union(A_e).is_valid == True and A_m.intersection(A_e).is_valid == True:
            cost = union_area - intersection_area
            print('Cost:', cost, 'Union area:', union_area, 'Intersection_area:', intersection_area)
    except:
        cost = 'N/A'
        print('Cost not calculated')

    cal_output = {'cost':cost, 'intersection':intersection_area, 'union':union_area, 'A_m': m_area, 'm_u_max':u_max, 'm_u_min':u_min, 'm_vac_max':vac_max, 'm_vac_min':vac_min}
    print(cal_output)
    return cal_output


def simulation(G, years, timestep, delta_u, gamma_u, delta_ny, gamma_ny, empirical_data, t_0, k, avg_hours_0, a, T, shock_start, attributes, calibration_output = False):
    '''
    Set attribute data of occupational mobility netowrk and carry out the simulation for a specified number of timesteps

    Parameters:
    -----------
    G (nx.Graph): The occupational mobility network
    data (pd.DataFrame): DataFrame containing data
    years (int): Number of years for the simulation
    timestep (float): size of each timestep in terms of weeks
    delta_u (float): probability of spontanous seperation
    delta_ny (float): probability of spontanously opneing vacancy
    gamma_u (float): speed of unemployed adjustment towards target demand
    gamma_ny (float): speed of vacancy adjustment towards target demand
    '''

    #set_attributes(G, data)
    # This needs to be put into the network (used as starting point)
    time_0 = dt.datetime.now()
    for key, value in attributes.items():
        nx.set_node_attributes(G, value, str(key))

    timesteps = round(years*52/timestep)
    T_steps = round(T*52/timestep)

    vacancies = nx.get_node_attributes(G, 'vacancies')
    employed = nx.get_node_attributes(G, 'employed')

    demand_0 = {}

    for key in vacancies.keys():
        demand_0[key] = vacancies[key] + employed[key] 

    vac_data = []
    emp_data = []
    unemp_data = []

    # Variables to calculate the post shock demand
    risk_factor = nx.get_node_attributes(G, 'comp_prob')
    # Need more data for average hours worked in a year
    average_hours_worked_0 = avg_hours_0
    
    # Empirical data
    e_vac_rate = empirical_data['sa_vac_rate']
    e_unemployed = empirical_data['u_trend']
    e_seq = [(u, e_vac_rate.iloc[i]) for i, u in enumerate(e_unemployed)]
    A_e = Polygon(e_seq)

    final_hours_worked = {}

    for occupation in risk_factor.keys():
        final_hours_worked[occupation] = average_hours_worked_0*employed[occupation]*(1-risk_factor[occupation])

    # L should be the workforce at steady state (employed + unemployed) and should be calculated in the loop
    L = sum(employed.values())

    final_average_hours_worked = sum(final_hours_worked.values())/L

    # Post shock demand
    final_demand = {occupation:hours/final_average_hours_worked for occupation, hours in final_hours_worked.items()}

    for t in range(timesteps):
        handle_applications(G)
        make_applications(G)
        make_vacancies(G, delta_ny, gamma_ny)
        separate_workers(G, delta_u, gamma_u)

        # order should be checked and changed
        # if t > shock_start:
        #    shock(G, demand_0, final_demand, t, t_0, k)

        vac_data.append(nx.get_node_attributes(G, 'vacancies'))
        unemp_data.append(nx.get_node_attributes(G, 'unemployed'))
        emp_data.append(nx.get_node_attributes(G, 'employed'))

        update_target_demand(G, demand_0, t, T_steps, a)
    

    model_data = {'vacancies': vac_data, 'unemployment': unemp_data, 'employment':emp_data}
    cost = calibration_calculation(empirical_data, model_data, A_e, t_0)

    vac_data = pd.DataFrame(vac_data)
    unemp_data = pd.DataFrame(unemp_data)
    emp_data = pd.DataFrame(emp_data)

    if calibration_output == True:
        return cost
    else:
        return {'vacancy_data': vac_data, 'unemployment_data': unemp_data, 'employment_data': emp_data, 'cost': cost}

def deterministic_simulation(G, years, timestep, delta_u, gamma_u, delta_ny, gamma_ny, empirical_data, t_0, k, L, avg_hours_0, a, T, shock_start, attributes, calibration_output = False):
    #set_attributes(G, data)
    # This needs to be put into the network (used as starting point)

    for key, value in attributes.items():
        nx.set_node_attributes(G, value, str(key))

    timesteps = round(T*52/timestep)*3
    T_steps = round(T*52/timestep)
    t_0 = round(t_0/timestep)

    vacancies = nx.get_node_attributes(G, 'vacancies')
    employed = nx.get_node_attributes(G, 'employed')

    demand_0 = {}

    for key in vacancies.keys():
        demand_0[key] = vacancies[key] + employed[key] 

    vac_data = []
    emp_data = []
    unemp_data = []

    # Variables to calculate the post shock demand
    risk_factor = nx.get_node_attributes(G, 'comp_prob')
    average_hours_worked_0 = avg_hours_0
    

    final_hours_worked = {}

    for occupation in risk_factor.keys():
        final_hours_worked[occupation] = average_hours_worked_0*employed[occupation]*(1-risk_factor[occupation])

    final_average_hours_worked = sum(final_hours_worked.values())/L

    # Post shock demand
    final_demand = {occupation:hours/final_average_hours_worked for occupation, hours in final_hours_worked.items()}



    occupations = list(G.nodes())
    time = dt.datetime.now()
    print('Simulation started at: ', time)
    for t in range(timesteps):
        ny = nx.get_node_attributes(G, 'vacancies')
        u = nx.get_node_attributes(G, 'unemployed')
        e = nx.get_node_attributes(G, 'employed')
        A = nx.get_edge_attributes(G, 'weight')

        s = {}
        f = {}
        for j in occupations:
            s[j] = []
            for i in G.predecessors(j):
                ny_A_sum = sum([ny[k]*A[(i,k)] for k in G.neighbors(i)])
                if ny_A_sum == 0:
                    s[j].append(0)
                else:
                    s[j].append(u[i]*ny[j]*A[(i,j)]/ny_A_sum)

            s[j] = sum(s[j])
            for i in G.predecessors(j):
                ny_A_sum = sum([ny[k]*A[(i,k)] for k in G.neighbors(i)])
                if s[j]*ny_A_sum == 0:
                    f[(i,j)] = 0
                else:
                    f[(i,j)] = u[i]*(ny[j]**(2))*A[(i,j)]*(1 - math.exp(-s[j]/ny[j]))/(s[j]*ny_A_sum)

        new_e = {}
        new_u = {}
        new_ny = {}
        
        target_demand = nx.get_node_attributes(G, 'target_demand')
        current_demand = {}

        for i in occupations:
            current_demand[i] = ny[i] + e[i]
            demand_diff = max(0, current_demand[i] - target_demand[i])

            f_i = sum([f[(j,i)] for j in G.predecessors(i)])

            new_e[i] = e[i] - delta_u*e[i] + (1 - delta_u)*gamma_u*demand_diff + f_i

            f_j = sum([f[(i,j)] for j in G.successors(i)])

            new_u[i] = u[i] + delta_u*e[i] + (1 - delta_u)*gamma_u*demand_diff - f_j

            demand_diff = max(0, target_demand[i]-current_demand[i])

            new_ny[i] = ny[i] + delta_ny*e[i] + (1-delta_ny)*gamma_u*demand_diff - f_i

        nx.set_node_attributes(G, new_ny, 'vacancies')
        nx.set_node_attributes(G, new_e, 'employed')
        nx.set_node_attributes(G, new_u, 'unemployed')

        vac_data.append(nx.get_node_attributes(G, 'vacancies'))
        unemp_data.append(nx.get_node_attributes(G, 'unemployed'))
        emp_data.append(nx.get_node_attributes(G, 'employed'))
        if T_steps == t:
            ss_e = new_e
            ss_demand = {}
            for occ, e in ss_e.items():
                ss_demand[occ] = target_demand[occ] - (delta_u - delta_ny)*e/(gamma_u*(1-delta_ny))
            

        if T_steps < t:
            update_target_demand(G, demand_0, t, T_steps, a)

        # order should be checked and changed
        # if t > shock_start:
        #    shock(G, demand_0, final_demand, t, t_0, k)

    model_data = {'vacancies': vac_data, 'unemployment': unemp_data, 'employment':emp_data}
    # Empirical data
    e_vac_rate = empirical_data['sa_vac_rate']
    e_unemployed = empirical_data['u_trend']
    e_seq = [(u, e_vac_rate.iloc[i]) for i, u in enumerate(e_unemployed)]#
    A_e = Polygon(e_seq).buffer(0)

    cost = calibration_calculation(empirical_data, model_data, A_e, t_0)
    cost['A_e'] = A_e.area

    vac_data = pd.DataFrame(vac_data)
    unemp_data = pd.DataFrame(unemp_data)
    emp_data = pd.DataFrame(emp_data)
    time = dt.datetime.now()- time
    print('Simulation took: ', time)
    cost['time'] = time
    if calibration_output == True:
        return cost
    else:
        return {'vacancy_data': vac_data, 'unemployment_data': unemp_data, 'employment_data': emp_data, 'cost': cost}

