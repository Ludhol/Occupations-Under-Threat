# Required packages
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import datetime as dt
from shapely.geometry import Polygon
import random
import math

'''
This package is used to implement a Data Driven Network Model of the labour market. It's intended use is to analyse an Occupational Mobility Network under an automation shock. However, it may be used for similar analysis of any labour flow network and any exogenous labour demand shock. The code is entirely based on the methods and theories outlined in "Automation and Occupational Mobility: A data-driven network model" by R. Maria del Rio-Chanona et. al., which may be found here: https://arxiv.org/pdf/1906.04086.pdf.

Author: Ludvig Holmér
'''

def make_vacancies(G, delta_nu, gamma_ny):
    '''
    Each occupation has a probability of making vacancies each time step that depends on the difference between
    current demand and target demand as well as an exogenous probability 
    Each vacancy is a list since workers apply to specific vacancies within an occupation
    If a worker applies to a vacancy, the occupation code which it came from is appended to the list

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
        current_demand = len(vac_dict[occupation]) + employed

        # Calculate the probability of creating new vacancies (equation 10)
        demand_diff = max(0, target_demand - current_demand)

        # Zero Steady State rule
        if employed == len(vac_dict[occupation]) == 0 and target_demand > 0:
            vac_dict[occupation].append([])
        else:
            if demand_diff == 0 or employed == 0:
                alpha_ny = 0
            else:
                alpha_ny = gamma_ny*demand_diff/employed

            p_ny = delta_nu + alpha_ny - delta_nu*alpha_ny

            new_vacancies = np.random.binomial(employed, p_ny)
            # Add the new vacancies to total number of vacancies
            for _ in range(new_vacancies):
                vac_dict[occupation].append([])

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
        vacancies = len(vac_dict[occupation])

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
    Each unemployed worker send out an application to a neighbouring occupation with probability proportional to the edge weight
    between the occupations
    Once an occupation is chosen, a random vacancy within that occupation is applied to

    Parameters:
    -----------
    G (nx.graph): The occupational mobility network
    '''
    # loop through every occupation and every unemployed person in the occupation and add applications in the 
    # applications attribute of the occupation that the worker applies to
    vac_dict = nx.get_node_attributes(G, 'vacancies')
    unem_dict = nx.get_node_attributes(G, 'unemployed')

    A_dict = nx.get_edge_attributes(G, 'weight')

    for occupation in vac_dict.keys():
        unemployed = unem_dict[occupation]
        neighbors = [n for n in G.neighbors(occupation)]

        # Calculate probabilities (equation 7)
        vac_A_list = [len(vac_dict[n])* A_dict[(occupation, n)] for n in neighbors]
        vac_A_sum = sum(vac_A_list)
        if vac_A_sum == 0:
            continue
        p_list = [vac_A/vac_A_sum for vac_A in vac_A_list]

        # Each unemployed person chooses an occupation to apply to
        for _ in range(unemployed):
            # Choice is weighted by probabilities calculated above
            apply_to_occ = np.random.choice(neighbors, p = p_list)
            # When occupation chosen a random vacancy within that occupation is applied for
            apply_to_vac = np.random.randint(len(vac_dict[apply_to_occ]))

            # Make the application
            vac_dict[apply_to_occ][apply_to_vac].append(occupation)

    nx.set_node_attributes(G, vac_dict, 'vacancies')


def handle_applications(G):
    '''
    Randomly accept N applications in the application attrbute where N is the number of vacancies in the occupation

    Parameters:
    -----------
    G (nx.graph): The occupational mobility network
    '''
    # The attributes we need
    vac_dict = nx.get_node_attributes(G, 'vacancies')
    emp_dict = nx.get_node_attributes(G, 'employed')
    unemp_dict = nx.get_node_attributes(G, 'unemployed')
    
    inflow_dict = {}

    # loop through all the occupations and try to accept the same number of applications as there are vacancies
    for occupation in vac_dict.keys():
        vacancies = vac_dict[occupation]
        vacs = len(vacancies)
        filled = 0
        # Here vac is a list of applicants and vacancies is a list of lists
        for vac in vacancies:
            if len(vac) == 0:
                # if no one applied for this vacancy -> continue to the next one
                continue
            # A random application is chosen (and removed from the list of applications)
            accepted = np.random.choice(vac)
            # Increase the amount of employed people in the occupation by the number of filled vacancies
            emp_dict[occupation] += 1
            # Decrease the amount of unemployed people in the occupations where applications were accepted from
            unemp_dict[accepted] -= 1
            # One more vacancy is filled
            filled += 1
            
        # Save inflow of workers for long-term unemployment calculation
        inflow_dict[occupation] = filled

        # Create a new list of vacancies based on how many were there in the begininng and how many were filled
        # Note: vacs >= filled is always true because of loop above
        if vacs > filled:
            # Not all vacancies filled:
            vac_dict[occupation] = [[]]*(vacs - filled)
        else:
            # All vacancies filled:
            vac_dict[occupation] = []

    # Update the network node attributes
    nx.set_node_attributes(G, vac_dict, 'vacancies')
    nx.set_node_attributes(G, emp_dict, 'employed')
    nx.set_node_attributes(G, unemp_dict, 'unemployed')
    nx.set_node_attributes(G, inflow_dict, 'inflow')

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

def calculate_long_term_unemployment(G, delta_u, gamma_u, t, unemp_data, f_i_data, timestep, e_0):
    new_lt_u = {}
    # Used to calculate long term unemployment (time-consuming)
    l = round(27/timestep)
    vacancies = {occ:len(vac) for occ, vac in nx.get_node_attributes(G, 'vacancies').items()}
    employed = nx.get_node_attributes(G, 'employed')
    target_demand = nx.get_node_attributes(G, 'target_demand')
    current_demand = {occ:vac + employed[occ] for occ, vac in vacancies.items()}
    for i in employed.keys():
        if 500 >= l:
            demand_diff = round(np.max([0, current_demand[i] - target_demand[i]]))
            omega = e_0[i] * (delta_u + (1 - delta_u)*gamma_u*demand_diff/employed[i])
            temp_sum = 0
             # Calculate long term unemployment
            while l <= t:
                lt_u_k = long_term_u(unemp_data, f_i_data, omega, l - 1, t-1, i)*(1 - f_i_data[t-1][i]/unemp_data[t-1][i])

                l +=1
                temp_sum += lt_u_k
            new_lt_u[i] = temp_sum
    return new_lt_u


def long_term_u(u, f_i, omega, k, t, i):
    '''
    Function that calculates the long term unemployment of an occupation i
    This function must be used within a simulation loop
    u and f_i are lists containing dicts for each time step with unemployment and inflow of workers for each occupation
    '''
    
    if t == 1 and k == 1:
        return omega
    elif u[t][i] == 0:
        return 0
    elif t == 1:
        return long_term_u(u, f_i, omega, k - 1, 1, i)*(1 - f_i[1][i]/u[1][i])
    elif k == 1:
        return long_term_u(u, f_i, omega, 1, t - 1, i)*(1 - f_i[t-1][i]/u[t-1][i])
    else:
        return long_term_u(u, f_i, omega, k - 1, t - 1, i)*(1 - f_i[t-1][i]/u[t-1][i])


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

    demand_shock = {}
    target_demand = {}

    for key in demand_0.keys():
        demand_shock[key] = (final_demand[key] - demand_0[key])/(1 + math.exp(-k*(t-t_0)))
        target_demand[key] = round(demand_0[key] + demand_shock[key])

    nx.set_node_attributes(G, target_demand, 'target_demand')


def calibration_calculation(empirical_data, model_data, A_e, period):
    '''
    This is the function to be minimised during calibration.abs

    Parameters:
    empirical_data (dict): timeseries of vacancies and unemployment
    model_data (dict): timeseries of vacancies and unemployment
    '''
    start = round(period*2)

    m_vacancies = [sum(model_data['vacancies'][i].values()) for i in range(len(model_data['vacancies']))]
    m_employed = [sum(model_data['employment'][i].values()) for i in range(len(model_data['employment']))]
    m_vac_rate = [m_vacancies[i]*100/(m_vacancies[i] + e) for i, e in enumerate(m_employed)]
    m_unemployed = [sum(model_data['unemployment'][i].values()) for i in range(len(model_data['unemployment']))]
    m_unemployed = [u*100/(m_employed[i]+ u) for i, u in enumerate(m_unemployed)]

   
    fig, ax = plt.subplots()
    cols = ["#69b243", "#ad5ec7"]
    ax.set_title("Simulated & Empirical Beveridge Curve", fontsize=16)
    ax.set_xlabel('Unemployment (%)')
    ax.set_ylabel('Vacancy rate (%)')



    plt.plot(empirical_data['u_trend'], empirical_data['sa_vac_rate'], color = cols[1], ls = '-', marker = 'o', linewidth = 1,
             markersize = 2, 
             label = 'Empirical data from \n' + str(empirical_data['date'][0]) + ' to ' + str(empirical_data['date'][-1]))
    plt.plot(m_unemployed, m_vac_rate, color = cols[0], ls = '-', marker = 'o', linewidth = 1, markersize = 2, 
             label = 'model output')
    
    ax.grid(linewidth=2, color='#999999', alpha=0.4)
    ax.tick_params(axis='both', which='both', labelsize=13,
                labelbottom=True, bottom=True, labelleft=True, left=True)
    plt.legend(bbox_to_anchor=(0.99, 0.99), loc='upper right', borderaxespad=0.)
    # plt.savefig('../Graphs/Bev_sim_ex.pdf', dpi=425, bbox_inches='tight')
    plt.show()

    u_ss = m_unemployed[-1]
    vac_ss = m_vac_rate[-1]
    m_vac_rate = m_vac_rate[start:]
    m_unemployed = m_unemployed[start:]

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
        return {'cost':cost, 'intersection':intersection_area, 'union':union_area, 'A_m': m_area, 
        'm_u_max':u_max, 'm_u_min':u_min, 'm_vac_max':vac_max, 'm_vac_min':vac_min, 'm_u_ss':u_ss, 'm_vac_ss': vac_ss}
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
            cost = 1 - intersection_area/union_area
            print('Cost:', cost, 'Union area:', union_area, 'Intersection_area:', intersection_area)
    except:
        cost = 'N/A'
        print('Cost not calculated')

    cal_output = {'cost':cost, 'intersection':intersection_area, 'union':union_area, 'A_m': m_area, 
    'm_u_max':u_max, 'm_u_min':u_min, 'm_u_ss':u_ss, 'm_vac_max':vac_max, 'm_vac_min':vac_min, 'm_vac_ss':vac_ss}
    # print(cal_output)
    return cal_output


def simulation(G, delta_u, delta_nu, gamma_u, gamma_nu, timestep, period, shock_period, 
                k, avg_hours_0, t_0, attributes, long_term = False, complete_network = False):
    '''
    Set attribute data of occupational mobility network and carry out the simulation for a specified number of timesteps

    Parameters:
    -----------
    G (nx.Graph): The occupational mobility network
    data (pd.DataFrame): DataFrame containing data
    years (int): Number of years for the simulation
    timestep (float): size of each timestep in terms of weeks
    delta_u (float): probability of spontanous seperation
    delta_nu (float): probability of spontanously opneing vacancy
    gamma_u (float): speed of unemployed adjustment towards target demand
    gamma_ny (float): speed of vacancy adjustment towards target demand
    '''

    if complete_network == True:
        G = nx.complete_graph(G, nx.DiGraph)
        edges = G.edges()
        weights = {edge: 1/len(G) for edge in edges}
        nx.set_edge_attributes(G, weights, 'weight')

    for key, value in attributes.items():
        nx.set_node_attributes(G, value, str(key))
    
    pre_steps = int(period*52/timestep)*2 # Steps before the automation shock
    shock_steps = int(shock_period*52/timestep) # Steps during the automation shock
    post_steps = int(period*52/timestep)*2 # Steps after the automation shock

    timesteps = pre_steps + shock_steps + post_steps
    
    # Data used to calculate the post shock demand
    comp_prob = nx.get_node_attributes(G, 'comp_prob')
    average_hours_worked_0 = avg_hours_0

    vacancies = nx.get_node_attributes(G, 'vacancies')
    employed = nx.get_node_attributes(G, 'employed')

    vac_data = []
    emp_data = []
    unemp_data = []
    td_data = []
    if long_term == True:
        lt_unemp_data = []
        f_i_data = []
        e_0 = {key:val for key, val in employed.items()}

    demand_0 = {occ:len(vacancies[occ]) + employed[occ] for occ in vacancies.keys()} 

    # Calculate the post shock demand for each occupation
    L = sum(demand_0.values())
    final_hours_worked = {occ : average_hours_worked_0*employed[occ]*(1-prob) for occ, prob in comp_prob.items()}
    final_average_hours_worked = sum(final_hours_worked.values())/L

    # Post shock demand
    final_demand = {occupation: round(hours/final_average_hours_worked) for occupation, hours in final_hours_worked.items()}

    # Print when the simulation starts
    time = dt.datetime.now()
    print('Simulation started at: ', time)

    for t in range(timesteps):
        # Conduct the simulation steps:
        # First, accept applications from previous timestep
        handle_applications(G)

        # Then, let the (unemployed) workers that were not accepted or did not apply last timestep, apply to vacancies
        make_applications(G)

        # Create new vacancies that may be applied for in the next timestep
        make_vacancies(G, delta_nu, gamma_nu)

        # Separate workers that may apply in the next timestep
        separate_workers(G, delta_u, gamma_u)

        # Calculate long-term unemployment
        if long_term == True:
            new_lt_u = calculate_long_term_unemployment(G, delta_u, gamma_u, t, unemp_data, f_i_data, timestep, e_0)
            lt_unemp_data.append(new_lt_u)
            new_f_i = nx.get_node_attributes(G, 'inflow')
            f_i_data.append(new_f_i)
        
        # Implement automation shock
        if pre_steps < t and t < shock_steps + pre_steps:
            shock(G, demand_0, final_demand, t*timestep/52, pre_steps*timestep/52 + t_0, k)
        # Save the data
        # For vacancies we are interested in the number of open vacancies - not the list of lists
        vac_total = {key:len(val) for key, val in nx.get_node_attributes(G, 'vacancies').items()}
        vac_data.append(vac_total)
        unemp_data.append(nx.get_node_attributes(G, 'unemployed'))
        emp_data.append(nx.get_node_attributes(G, 'employed'))
        td_data.append(nx.get_node_attributes(G, 'target_demand'))



    # The simulation is completed
    print('Simulation completed', dt.datetime.now())
    time = dt.datetime.now() - time
    print('Simulation took: ', time)
    if long_term == True:
        return {'vacancy_data': vac_data, 'unemployment_data': unemp_data, 'employment_data': emp_data, 
                'target_demand_data': td_data, 'lt_unemp_data': lt_unemp_data}
    else:
        return {'vacancy_data': vac_data, 'unemployment_data': unemp_data, 'employment_data': emp_data,
                'target_demand_data': td_data}

    
def deterministic_simulation(G, delta_u, delta_nu, gamma_u, gamma_nu, timestep, period, attributes, 
                             t_0 = None, avg_hours_0 = None, k = None, shock_period = None, long_term = False, 
                             complete_network = False, calibration = False, steady_state = False, empirical_data = None, a = 0.03):

    # if calibration == True and empirical_data == None:
    #    print('EEROR: Empirical data must be input for calibration run')
    #    return
    if long_term == calibration == True:
        print('ERROR: Should not calculate long term unemployment during a calibration run')
        return
    if complete_network == calibration == True:
        print('ERROR: Should not use complete network during calibration run')
        return
    
    if complete_network == True:
        G = nx.complete_graph(G, nx.DiGraph)
        edges = G.edges()
        weights = {edge: 1/len(G) for edge in edges}
        nx.set_edge_attributes(G, weights, 'weight')

    for key, value in attributes.items():
        nx.set_node_attributes(G, value, str(key))

    # Translate period (in years) into timesteps where timestep is the number of weeks between steps
    if calibration == False and steady_state == False:
        pre_steps = int(period*52/timestep)*2 # Steps before the automation shock
        shock_steps = int(shock_period*52/timestep) # Steps during the automation shock
        post_steps = int(period*52/timestep)*2 # Steps after the automation shock
        timesteps = pre_steps + shock_steps + post_steps
    elif steady_state == True:
        timesteps = int(period*52/timestep)
    else:
        pre_steps = int(period*52/timestep) # Steps before the automation shock
        timesteps = pre_steps*3
    

    # Calculate intial demand
    vacancies = nx.get_node_attributes(G, 'vacancies')
    employed = nx.get_node_attributes(G, 'employed')

    e_0 = {key:val for key, val in employed.items()}

    demand_0 = {occ:vacancies[occ] + employed[occ] for occ in vacancies.keys()} 

    vac_data = []
    emp_data = []
    unemp_data = []
    td_data = []
    if long_term == True:
        lt_unemp_data = []
        f_i_data = []

    # Data to calculate the post shock demand
    comp_prob = nx.get_node_attributes(G, 'comp_prob')
    average_hours_worked_0 = avg_hours_0

    # Calculate post shock demand
    L = sum(demand_0.values())
    final_hours_worked = {occ : average_hours_worked_0*employed[occ]*(1-prob) for occ, prob in comp_prob.items()}
    final_average_hours_worked = sum(final_hours_worked.values())/L

    # Post shock demand
    final_demand = {occupation: round(hours/final_average_hours_worked) for occupation, hours in final_hours_worked.items()}

    occupations = G.nodes()
    time = dt.datetime.now()

    # The occupational mobility network does not change and may be set here
    A = nx.get_edge_attributes(G, 'weight')
    # The beginning of the simulation
    print('Simulation started at: ', time)
    for t in range(timesteps):
        # Initial values from previous timesteps
        nu = nx.get_node_attributes(G, 'vacancies')
        u = nx.get_node_attributes(G, 'unemployed')
        e = nx.get_node_attributes(G, 'employed')

        # s is the expected number of applicants to an occupation, j
        s = {}
        # f is the expected flow to and from an occupation, j
        f = {}
        for j in occupations:
            # Begin by calculating expected applicants since this determines flow
            s[j] = []
            # Loop over the occupations that have an edge to j
            for i in G.predecessors(j):
                # Workers in each of these occupations may apply to any occupation which they have an edge to 
                # (weighted by edge weight)
                nu_A_sum = np.sum([nu[k]*A[(i,k)] for k in G.neighbors(i)])
                # Workers may only apply to occupations in which there are vacancies
                if nu_A_sum == 0:
                    s[j].append(0)
                else:
                    s[j].append(u[i]*nu[j]*A[(i,j)]/nu_A_sum)

            # Sum over the expected applicants from all occupations that have an edge to j
            s[j] = sum(s[j])
            # Calculate the expected flow of workers from i to j
            for i in G.predecessors(j):
                # This is done via the expression arrived at by Taylor expansion
                nu_A_sum = np.sum([nu[k]*A[(i,k)] for k in G.neighbors(i)])
                if s[j]*nu_A_sum == 0:
                    f[(i,j)] = 0
                else:
                    f[(i,j)] = u[i]*(nu[j]**(2))*A[(i,j)]*(1 - math.exp(-s[j]/nu[j]))/(s[j]*nu_A_sum)

        new_e = {}
        new_u = {}
        new_nu = {}
        if long_term == True:
            new_f_i = {}
            new_lt_u = {}
        
        target_demand = nx.get_node_attributes(G, 'target_demand')
        current_demand = {}
        for i in occupations:
            # Set the current demand of the occupation
            current_demand[i] = nu[i] + e[i]
            demand_diff = round(np.max([0, current_demand[i] - target_demand[i]]))

            # Calculate the inflow of employees to the occupation 
            f_i = round(np.sum([f[(j,i)] for j in G.predecessors(i)]))
            # saved since timeseries is required to calculate long term unemployed
            if long_term == True:
                new_f_i[i] = f_i

            # Calculate new amount of employees
            new_e[i] = round(e[i] - delta_u*e[i] - (1 - delta_u)*gamma_u*demand_diff + f_i)

            # Calculate outflow of unemployed workers
            f_j = round(np.sum([f[(i,j)] for j in G.successors(i)]))

            # Calculate new amount of unemployed workers
            new_u[i] = round(u[i] + delta_u*e[i] + (1 - delta_u)*gamma_u*demand_diff - f_j)
            
            # Calculate new vacancies
            demand_diff = round(np.max([0, target_demand[i]-current_demand[i]]))
            new_nu[i] = round(nu[i] + delta_nu*e[i] + (1-delta_nu)*gamma_nu*demand_diff - f_i) 

            # Used to calculate long term unemployment (time-consuming)
            l = round(27/timestep)
            if long_term == True and t >=l:
                demand_diff = round(np.max([0, current_demand[i] - target_demand[i]]))
                omega = e_0[i] * (delta_u + (1 - delta_u)*gamma_u*demand_diff/e[i])
                 # Calculate long term unemployment

                temp_sum = 0
                while l <= 500:
                    lt_u_k = long_term_u(unemp_data, f_i_data, omega, l - 1, t-1, i)*(1 - f_i_data[t-1][i]/u[i])
                    l +=1
                    temp_sum += lt_u_k
                new_lt_u[i] = round(temp_sum)

           
        # Update network with new values
        nx.set_node_attributes(G, new_nu, 'vacancies')
        nx.set_node_attributes(G, new_e, 'employed')
        nx.set_node_attributes(G, new_u, 'unemployed')
        if long_term == True:
            nx.set_node_attributes(G, new_lt_u, 'lt_unemployed')
            nx.set_node_attributes(G, new_f_i, 'f_i')

        # Save values for output
        vac_data.append(new_nu)
        unemp_data.append(new_u)
        emp_data.append(new_e)
        td_data.append(target_demand)
        if long_term == True:
            lt_unemp_data.append(new_lt_u)
            f_i_data.append(new_f_i)

        # Implement automation shock
        if steady_state == True:
            continue
        elif calibration == False:
            if pre_steps < t and t < shock_steps + pre_steps:
                shock(G, demand_0, final_demand, t*timestep/52, t_0 + pre_steps*timestep/52, k)
        else:
            if pre_steps < t:
                update_target_demand(G, demand_0, t, pre_steps, a)
        
    if steady_state == True:
        time = dt.datetime.now()- time
        print('simulation took:', time)
        ss_vacs = sum([vac for vac in vac_data[-1].values()])
        ss_emp = sum([emp for emp in emp_data[-1].values()])
        ss_unemp = sum([unemp for unemp in unemp_data[-1].values()])
        ss_vac_rate = ss_vacs*100/(ss_vacs + ss_emp)
        ss_unemp_rate = ss_unemp*100/(ss_unemp + ss_emp)
        return [ss_unemp_rate, ss_vac_rate]
    
    if calibration == True:
        model_data = {'vacancies': vac_data, 'unemployment': unemp_data, 'employment':emp_data}

        # Empirical data
        e_vac_rate = empirical_data['sa_vac_rate']
        e_unemployed = empirical_data['u_trend']
        e_seq = [(u, e_vac_rate.iloc[i]) for i, u in enumerate(e_unemployed)]
        A_e = Polygon(e_seq).buffer(0)
        cost = calibration_calculation(empirical_data, model_data, A_e, pre_steps)
        cost['A_e'] = A_e.area
        time = dt.datetime.now()- time
        print('Simulation took: ', time)
        return cost

    
    time = dt.datetime.now()- time
    print('Simulation took: ', time)
    if long_term == True:
        return {'vacancy_data': vac_data, 'unemployment_data': unemp_data, 'employment_data': emp_data, 
                'target_demand_data': td_data, 'lt_unemp_data': lt_unemp_data}

    else:
        return {'vacancy_data': vac_data, 'unemployment_data': unemp_data, 'employment_data': emp_data, 
                'target_demand_data': td_data}