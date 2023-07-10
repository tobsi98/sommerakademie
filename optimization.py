import os
import csv
import numpy as np
import pandas as pd
import pyomo.environ as en
from pyomo.opt import SolverFactory
from pyomo.opt import TerminationCondition, SolverStatus
from matplotlib import pyplot as plt



def opt_dc(nr_time_steps, nr_cooling_machines=4, nr_fwp=2, nr_locations=3, cop=4, LOAD_STEPS_PER_HOUR=4):
    model = en.AbstractModel()
    # ############################## Sets #####################################
    model.T = en.Set(initialize=np.arange(nr_time_steps))
    model.KM = en.Set(initialize=np.arange(nr_cooling_machines))
    model.KM_CWP = en.Set(initialize=['Small', 'Large']) # Cold Water Pump
    model.FWP = en.Set(initialize=np.arange(nr_fwp))
    model.LOCA = en.Set(initialize=np.arange(nr_locations))
    

    # ############################## Parameters ###############################
    # min and max power in [kW]
    model.km_min_pow = en.Param(model.KM, mutable=True)
    model.km_max_pow = en.Param(model.KM, mutable=True)
    # load in [kW]
    model.cooling_load = en.Param(model.LOCA, model.T, mutable=True)
    # Water flow in [m3/h]
    model.water_flow = en.Param(model.LOCA, model.T, mutable=True)
    model.cop = en.Param(initialize=cop)
    # incentive in [EUR/MWh]
    model.incentive = en.Param(mutable=True)
    # Electricity price in [EUR/MWh]
    model.electricity_price = en.Param(model.T, mutable=True)
    model.maintenance_costs = en.Param(mutable=True)

    # ############################## Variables ################################
    model.km_status = en.Var(model.LOCA, model.KM, model.T, domain=en.Binary)
    model.km_cwp_status = en.Var(model.LOCA, model.KM, model.KM_CWP, model.T, domain=en.Binary)
    # cold water pump power [kW]
    model.cwpp = en.Var(model.LOCA, model.T, domain=en.NonNegativeReals)
    model.km_generation_power = en.Var(model.LOCA, model.KM, model.T, domain=en.NonNegativeReals)
    model.elec_load = en.Var(model.LOCA, model.T, domain=en.NonNegativeReals)
    model.fwp_power =en.Var(model.LOCA, model.FWP,model.T,domain=en.NonNegativeReals)

    # ############################## Constraints ##############################
    
    def cover_load_rule(model, loca, t):
        return sum(model.km_generation_power[loca, i, t] for i in model.KM) >= model.cooling_load[loca, t]
    model.cover_load = en.Constraint(model.LOCA, model.T, rule=cover_load_rule)

    def km_min_generation_rule(model, loca, i, t): #km = kältemaschine 
        return model.km_status[loca, i, t] * model.km_min_pow[i] <= model.km_generation_power[loca, i, t]
    model.km_min_generation = en.Constraint(model.LOCA, model.KM, model.T, rule=km_min_generation_rule)

    def km_max_generation_rule(model, loca, i, t):
        return model.km_status[loca, i, t] * model.km_max_pow[i] >= model.km_generation_power[loca, i, t]
    model.km_max_generation = en.Constraint(model.LOCA, model.KM, model.T, rule=km_max_generation_rule)


    def cwp_status_rule(model, loca, i, j, t): #cwp = cool water pump
        return model.km_cwp_status[loca, i, j, t] <= model.km_status[loca, i, t]
    model.cwp_status = en.Constraint(model.LOCA, model.KM, model.KM_CWP, model.T, rule=cwp_status_rule)

    def cwp_wf_rule(model, loca, t):
        return model.water_flow[loca, t] <= sum(model.km_cwp_status[loca, i, 'Small', t] * 300 + model.km_cwp_status[loca, i, 'Large', t] * 420 for i in model.KM)
    model.cwp_wf = en.Constraint(model.LOCA, model.T, rule=cwp_wf_rule)

    def cwp_power_rule(model, loca, t):
        return model.cwpp[loca, t] == sum(model.km_cwp_status[loca, i, 'Small', t] * 23 + model.km_cwp_status[loca, i, 'Large', t] * 40.1 for i in model.KM)
    model.cwp_power = en.Constraint(model.LOCA, model.T, rule=cwp_power_rule)
    
    def fwp1_power_rule(model, loca, t):
      s1= model.km_status[loca, 0,t]
      s2= model.km_status[loca, 1,t]
      return model.fwp_power[loca, 0,t] >= 18*s1+18*s2+(s1+s2-1)*6.95*s1+(s1+s2-1)*6.95*s2
    model.fwp1_power = en.Constraint(model.LOCA, model.T, rule=fwp1_power_rule)
    
    def fwp2_power_rule(model, loca, t):
      s1= model.km_status[loca, 2,t]
      s2= model.km_status[loca, 3,t]
      return model.fwp_power[loca, 1,t] >= 18*s1+18*s2+(s1+s2-1)*6.95*s1+(s1+s2-1)*6.95*s2
    model.fwp2_power = en.Constraint(model.LOCA, model.T, rule=fwp2_power_rule)
    

    def electricity_load_rule(model, loca, t):
        return model.elec_load[loca, t] >= sum(model.km_generation_power[loca, i, t] / model.cop for i in model.KM)
    model.electricity_load = en.Constraint(model.LOCA, model.T, rule=electricity_load_rule)
    
    # def loca_rule(model, loca):
    #     return model.LOCA == 0
    # model.LOCA = en.Constraint(model.LOCA, rule=loca_rule)
    
    def obj_rule(model):
        # OBJECTIVE FUNC: Minimize Elec Costs
        return sum(sum(((model.cwpp[loca, t] + model.fwp_power[loca, 0,t] + 
                         model.fwp_power[loca, 1,t]) * model.electricity_price[t] + 
                   model.elec_load[loca, t] * (model.electricity_price[t] + 
               model.maintenance_costs - model.incentive)) / LOAD_STEPS_PER_HOUR
                for t in model.T)for loca in model.LOCA)
    model.obj = en.Objective(rule=obj_rule, sense=en.minimize)
    return model.create_instance()

    # print("Solver Termination: ", results.solver.termination_condition)
    # print("Solver Status: ", results.solver.status)
    # term_cond = results.solver.termination_condition == TerminationCondition.optimal

if __name__ == "__main__":
    cop = 4
    nr_cooling_machines = 4
    # load for one location
    f_load = "./inputs/load_all.csv"
    df = pd.read_csv(f_load)
    df['Index'] = list(zip(df['utility'], df['t']))
    df.set_index('Index', inplace=True)
    df.drop(columns=['utility', 't'], inplace=True)
    
    # load for all locations
   # f_load = "./inputs/load_all.csv"
   # df = pd.read_csv(f_load)
   # data_location0 = df[(df['utility']==0)]
   # data_location1 = df[(df['utility']==1)]
   # data_location1.reset_index(inplace = True, drop=True)
   # data_location2 = df[(df['utility']==2)]
  #  data_location2.reset_index(inplace = True, drop=True)
   # data_location1.rename(columns={'load': 'load1'}, inplace=True)
    #data_location2.rename(columns={'load': 'load2'}, inplace=True)
    #data = pd.concat([data_location0['t'], data_location0['load'], data_location1['load1'], data_location2['load2']], axis=1)
    #df = data_location2

    
    df.plot(y=['load', 'water_flow'])
    #df_loc1 = df[:653]
    #df_loc2 = df[653:1306]
    #df_loc3 = df[1306:1960]
    #plt.show()
    nr_time_steps = 653
    #write inputs into params
    instance = opt_dc(nr_time_steps, nr_locations=3)
    instance.cooling_load.store_values(df['load'].to_dict())
    instance.water_flow.store_values(df['water_flow'].to_dict())

    instance.km_min_pow.store_values({i: value for i, value in enumerate([700]
                                                 * nr_cooling_machines)}) #kW
    instance.km_max_pow.store_values({i: value for i, value in enumerate([2800] 
                                                 * nr_cooling_machines)}) #kW
    instance.incentive.store_values({None: 0.2*1e-3}) #20ct/MWh
    instance.electricity_price.store_values({i: value for i, value in 
                                     enumerate([20*1e-3] * nr_time_steps)}) #2ct/kWh
    instance.maintenance_costs.store_values({None: 1.2*1e-3}) #1,2€/kWh
    solver = SolverFactory('gurobi')
    opt_results = solver.solve(instance, tee=True)
    print("Solver Termination: ", opt_results.solver.termination_condition)
    print("Solver Status: ", opt_results.solver.status)
    with open('optimization_log.txt', 'w') as output_file:
        instance.pprint(output_file)

    variable_data = []
    df1 = pd.DataFrame()
    i=0
    spalte=0
    for var in instance.component_objects(en.Var, descend_into=True):
        for index in var:
            i = i + 1
            name = var[index].getname().split('[')[0]
            value_list = [en.value(var[index])] if var[index].is_indexed() else [en.value(var[index])]
            variable_data.append((name, index) + tuple(value_list))
            if i % 653 == 0:
                spalte = spalte + 1
                newList = variable_data[i - 653: i]
                cstring = ""
                if type(index) is tuple:
                    for v in range(0,len(index) - 1):
                        cstring = cstring + " " + str(index[v])
                    df1[name + cstring] = [item[2] for item in newList]
                else:
                    df1[name] = [item[2] for item in newList]
    fig, axes = plt.subplots()
    df1.plot.area(y=['km_generation_power 0 0', 'km_generation_power 0 1',
                     'km_generation_power 0 2', 'km_generation_power 0 3'])
    plt.xlabel('15 Minuten Intervall über eine Woche')
    plt.ylabel('Leistung KMs Standort 1 [kW]')
    df1.plot.area(y=['fwp_power 0 0'])
    plt.show()
    df1.plot.area(y=['fwp_power 0 1'])
    plt.show()
    
    df1.plot.area(y=['km_generation_power 1 0', 'km_generation_power 1 1',
                     'km_generation_power 1 2', 'km_generation_power 1 3'])
    plt.xlabel('15 Minuten Intervall über eine Woche')
    plt.ylabel('Leistung KMs Standort 2 [kW]')
    df1.plot.area(y=['fwp_power 1 0'])
    plt.show()
    df1.plot.area(y=['fwp_power 1 1'])
    plt.show()
    
    df1.plot.area(y=['km_generation_power 2 0', 'km_generation_power 2 1',
                     'km_generation_power 2 2', 'km_generation_power 2 3'])
    plt.xlabel('15 Minuten Intervall über eine Woche')
    plt.ylabel('Leistung KMs Standort 3 [kW]')
    df1.plot.area(y=['fwp_power 2 0'])
    plt.show()
    df1.plot.area(y=['fwp_power 2 1'])
    plt.show()
    # Save variable names, indices, and values to a CSV file
    filename = 'variable_data.csv'
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(('Variable', 'Index', 'Value'))
        for data in variable_data:
            writer.writerow(data)
            

#    sns.set_theme(palette='muted')

    
# ax.stackplot(timesteps, 
#              PowerThermal.to_numpy(dtype = float).transpose(), wind, pv, 
#              values['w_pump'].to_numpy(dtype = float).transpose(),
#              values['w_turb'].to_numpy(dtype = float).transpose(),
#              labels=['Kohle', 'GuD', 'Gasturbine','Wind', 'PV','Pump','Turbinieren'], 
#              colors = ["grey", "blue", "red", "green", "yellow", "purple", "orange"])
# ax.set_title('Bsp3: Fossile + erneuerbare Kraftwerke + Speicher')
# ax.legend(loc='lower left')
# ax.set_ylabel('Erzeugung [MW]')
# ax.set_xlim(xmin=timesteps[0], xmax=timesteps[23])
# fig.tight_layout()
