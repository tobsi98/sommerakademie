import os
import csv
import numpy as np
import pandas as pd
import pyomo.environ as en
from pyomo.opt import SolverFactory
from pyomo.opt import TerminationCondition, SolverStatus
from matplotlib import pyplot as plt



def opt_dc(nr_time_steps, nr_cooling_machines=4, nr_fwp=2, nr_locations=3, nr_pipes=3, cop=4, LOAD_STEPS_PER_HOUR=4, SOC_start=0, eta_laden=0.95, eta_entladen=0.97, SOC_max=28660, SOC_ende=0, p_laden=2866, p_entladen=6000):
    
    model = en.AbstractModel()
    #model.setParam('MIPGap', 0.05)
    # ############################## Sets #####################################
    model.T = en.Set(initialize=np.arange(nr_time_steps))
    model.KM = en.Set(initialize=np.arange(nr_cooling_machines))
    model.KM_CWP = en.Set(initialize=['Small', 'Large']) # Cold Water Pump
    model.FWP = en.Set(initialize=np.arange(nr_fwp))
    model.LOCA = en.Set(initialize=np.arange(nr_locations))
   # model.PIPES = en.Set(initialize=np.arange(nr_pipes))
    

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
    # transfer loss param 
  #  model.transfer_losses = en.Param(model.LOCA, model.LOCA, mutable=True)
      

    # ############################## Variables ################################
    model.km_status = en.Var(model.LOCA, model.KM, model.T, domain=en.Binary)
    model.km_cwp_status = en.Var(model.LOCA, model.KM, model.KM_CWP, model.T, domain=en.Binary)
    # cold water pump power [kW]
    model.cwpp = en.Var(model.LOCA, model.T, domain=en.NonNegativeReals)
    model.km_generation_power = en.Var(model.LOCA, model.KM, model.T, domain=en.NonNegativeReals)
    model.elec_load = en.Var(model.LOCA, model.T, domain=en.NonNegativeReals)
    model.fwp_power = en.Var(model.LOCA, model.FWP,model.T,domain=en.NonNegativeReals)
    #model.import_level = en.Var(model.LOCA, model.T, domain=en.NonNegativeReals)
  #  model.surplus = en.Var(model.LOCA, model.T, domain=en.NonNegativeReals)
    # constraints: import level (in cover load), surplus
    model.speicher_laden = en.Var(model.T, domain=en.NonNegativeReals)
    model.speicher_entladen = en.Var(model.T, domain=en.NonNegativeReals)
    model.speicher_SOC = en.Var(model.T, domain=en.NonNegativeReals)

    # ############################## Constraints ##############################
    # def surplus_rule(model, loca, t):
    #     return model.surplus[loca,t] == sum(model.km_generation_power[loca,i,t] for i in model.KM) - model.cooling_load[loca,t]
    # model.surplus_power = en.Constraint(model.LOCA, model.T, rule=surplus_rule)
    
    # def import_level_rule(model, locai, t):
    #     return model.import_level[locai,t] <= sum(model.surplus[j,t] * model.transfer_losses[(locai, j)] for j in model.LOCA)
    # model.import_lvl = en.Constraint(model.LOCA, model.T, rule=import_level_rule)
    
    # def import_level_2_rule(model, locai, locaj, t):
    #     return model.import_level[locai, locaj]
    
    #### cover_load_rule anpassen:  generation + import_level
    
    # def pipe_rule(model, loca, t): #welche Leistung muss auf der Leitung fließen
    #     return model.pipes_power[] == (sum(model.km_generation_power[loca,i,t] for i in model.KM) - model.cooling_load[loca,t])*eta(loca)
    #model.cover_load = en.Constraint(model.T, rule=pipe_rule)
    # Location A0
    def cover_load0_rule(model, t):
        return sum(model.km_generation_power[0,i,t] +0.95*model.km_generation_power[1,i,t] + 0.9*model.km_generation_power[2,i,t] for i in model.KM) == model.cooling_load[0,t]+0.95*model.cooling_load[1,t]+0.90*model.cooling_load[2,t]+model.speicher_laden[t] - model.speicher_entladen[t]
    model.cover_load0 = en.Constraint(model.T, rule=cover_load0_rule)
    # Location A1
    def cover_load1_rule(model, t):
        return sum(0.95*model.km_generation_power[0,i,t] + model.km_generation_power[1,i,t] + 0.92*model.km_generation_power[2,i,t] for i in model.KM) >= model.cooling_load[1,t]+0.95*model.cooling_load[0,t]+0.92*model.cooling_load[2,t]
    model.cover_load1 = en.Constraint(model.T, rule=cover_load1_rule)
    # Location A2
    def cover_load2_rule(model, loca, t):
        return sum(0.9*model.km_generation_power[0,i,t] + 0.92*model.km_generation_power[1,i,t] + model.km_generation_power[2,i,t] for i in model.KM) >= model.cooling_load[2,t] + 0.9*model.cooling_load[0,t] + 0.92*model.cooling_load[1,t]
    model.cover_load2 = en.Constraint(model.LOCA, model.T, rule=cover_load2_rule)
    # def cover_load_rule(model, loca, t):  
    #     return sum(model.km_generation_power[loca, i, t] for i in model.KM) + model.import_level[loca,t] >= model.cooling_load[loca, t]
    #     #return sum(model.km_generation_power[loca, i, t] for i in model.KM) >= model.cooling_load[loca, t] - model.surplus[loca,t] 
    # model.cover_load = en.Constraint(model.LOCA, model.T, rule=cover_load_rule)

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
      return model.fwp_power[loca, 0,t] == 18*s1+18*s2+(s1+s2-1)*6.95*s1+(s1+s2-1)*6.95*s2
    model.fwp1_power = en.Constraint(model.LOCA, model.T, rule=fwp1_power_rule)
    
    def fwp2_power_rule(model, loca, t):
      s1= model.km_status[loca, 2,t]
      s2= model.km_status[loca, 3,t]
      return model.fwp_power[loca, 1,t] == 18*s1+18*s2+(s1+s2-1)*6.95*s1+(s1+s2-1)*6.95*s2
    model.fwp2_power = en.Constraint(model.LOCA, model.T, rule=fwp2_power_rule)
    

    def electricity_load_rule(model, loca, t):
        return model.elec_load[loca, t] == sum(model.km_generation_power[loca, i, t] / model.cop for i in model.KM)
    model.electricity_load = en.Constraint(model.LOCA, model.T, rule=electricity_load_rule)
   
    # def km_status_crash_rule(model, t):
    #     return model.km_status[2, 0, t] == 0
    # model.km_crash_status = en.Constraint(model.T, rule=km_status_crash_rule)
    
    #################### Constraints Speicher ######################
    # def speicher_soc_init_rule(model, t): #Ladezustand im ersten Zeitwert 
    #     return      
    # model.speicher_soc_init = en.Constraint(model.T, rule = speicher_soc_init_rule)
    
    def speicher_soc_rule(model, t): #Speicherladezustand
        if t!=0:
            return (model.speicher_SOC[t-1]*0.9975 
           + (model.speicher_laden[t]*eta_laden 
           - model.speicher_entladen[t]/eta_entladen)/4
           - model.speicher_SOC[t] == 0)
        else:
            return SOC_start + (model.speicher_laden[0] * eta_laden - model.speicher_entladen[0]/eta_entladen)/4 - model.speicher_SOC[0] == 0
    model.speicher_soc = en.Constraint(model.T, rule=speicher_soc_rule)
    
    def speicher_soc_end_rule(model): #Endzustand
        return model.speicher_SOC[653-1] == SOC_ende
    model.speicher_SOC_end = en.Constraint(rule=speicher_soc_end_rule)
    
    # def speicher_entladen_init_rule(model): #entladen init rule
    #     return 
    # model.speicher_entladen_con = en.Constraint(rule=speicher_entladen_init_rule)
    
    def speicher_entladen_rule(model, t):  
        if t!=0:
            return model.speicher_entladen[t] <= model.speicher_SOC[t-1]*4
        else:
            return model.speicher_entladen[0] <= SOC_start*4
    model.speicher_entladen_con = en.Constraint(model.T, rule=speicher_entladen_rule)

    def speicher_max_rule(model, t):
        return model.speicher_SOC[t] <= SOC_max
    model.speicher_max_con = en.Constraint(model.T, rule=speicher_max_rule)
    
    def speicher_maxladen_rule(model, t):
        return model.speicher_laden[t] <= p_laden
    model.speicher_maxladen_con = en.Constraint(model.T, rule=speicher_maxladen_rule)
    
    def speicher_maxentladen_rule(model, t):
        return model.speicher_entladen[t] <= p_entladen
    model.speicher_maxentladen_con = en.Constraint(model.T, rule=speicher_maxentladen_rule)

  ##################### Zielfunktion #########################################

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
    
    f_eprice = "./inputs/ElectricityPrice.xlsx"
    df_eprice = pd.read_excel(f_eprice)
    df_eprice['preis abs'] = df_eprice['strompreis [ct/kWh]']
    df_eprice['preis abs'] = df_eprice['preis abs'] / 100
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
    # fig, ax = plt.subplots()

    # twin1 = ax.twinx()
    # p1 = ax.plot(df.index[], df['load'], label='Last Standort A0',color="b")
    # p2 = twin1.plot(df.index[], df_eprice['preis abs'], label='strompreis',color='orange')

    # ax.set_xlabel("Zeitstempel in 1/4 h")
    # ax.set_ylabel("Speicherstand in kWh")
    # twin1.set_ylabel("Strompreis in €/kWh")
    # ax.yaxis.label.set_color('b')
    # twin1.yaxis.label.set_color('r')

    # ax.set_xlim(0, 652)
    # ax.set_ylim(0, 15000)
    # twin1.set_ylim(-0.6, 0.5)

    # plt.show()
    #df_loc1 = df[:653]
    #df_loc2 = df[653:1306]
    #df_loc3 = df[1306:1960]
    #plt.show()
    nr_time_steps = 653
    
    #write inputs into params
    instance = opt_dc(nr_time_steps, nr_locations=3, SOC_start=0, eta_laden=0.95, eta_entladen=0.97, SOC_max=28660, SOC_ende=0, p_laden=2866, p_entladen=6000)
    instance.cooling_load.store_values(df['load'].to_dict())
    instance.water_flow.store_values(df['water_flow'].to_dict())

    instance.km_min_pow.store_values({i: value for i, value in enumerate([700]
                                                 * nr_cooling_machines)}) #kW
    instance.km_max_pow.store_values({i: value for i, value in enumerate([2800] 
                                                 * nr_cooling_machines)}) #kW
    instance.incentive.store_values({None: 0.2*1e-3}) #0,00002€/kWh
    instance.electricity_price.store_values(df_eprice['preis abs'].to_dict())
   # instance.electricity_price.store_values({i: value for i, value in 
    #                                 enumerate([20*1e-3] * nr_time_steps)}) #0,02€/kWh
    instance.maintenance_costs.store_values({None: 1.2*1e-3}) #1,2€/kWh
   # instance.transfer_losses.store_values({(0,1): 0.95, (0,2): 0.9, (1,2): 0.92,
   #                                       (1,0): 0.95, (2,1): 0.92, (2,0): 0.9,
   #                                       (0,0): 0, (1,1): 0, (2,2): 0})
    solver = SolverFactory('gurobi')
    solver.options['mipgap'] = 0.001
    
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
    
    df1 = df1.abs()
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
                     'km_generation_power 1 2', 'km_generation_power 1 3'], ylim=(0,4000))
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
            
    df_load = pd.read_csv(f_load)
    df1['total_gen_A0'] = pd.DataFrame(df1['km_generation_power 0 0'] + df1['km_generation_power 0 1'] + df1['km_generation_power 0 2'] + df1['km_generation_power 0 3'])
    df1['total_gen_A1'] = pd.DataFrame(df1['km_generation_power 1 0'] + df1['km_generation_power 1 1'] + df1['km_generation_power 1 2'] + df1['km_generation_power 1 3'])
    df1['total_gen_A2'] = pd.DataFrame(df1['km_generation_power 2 0'] + df1['km_generation_power 2 1'] + df1['km_generation_power 2 2'] + df1['km_generation_power 2 3'])
    load_A0 = pd.DataFrame(df_load.loc[df_load['utility']==0]['load'])
    load_A1 = pd.DataFrame(df_load.loc[df_load['utility']==1]['load'])
    load_A2 = pd.DataFrame(df_load.loc[df_load['utility']==2]['load'])
    load_A2.set_index(df1.index, inplace=True)
    load_A1.set_index(df1.index, inplace=True)
    surplus_A0 = pd.DataFrame(df1['total_gen_A0'] - load_A0['load'])
    surplus_A1 = pd.DataFrame(df1['total_gen_A1'] - load_A1['load'])
    surplus_A2 = pd.DataFrame(df1['total_gen_A2'] - load_A2['load'])
    
    surplus_A0.plot(title='Überschuss Standort A0', ylabel='Überschuss in kWh', xlabel=' Zeitstempel in 1/4 h', legend=False, grid=True, xlim=(0,652))
    surplus_A1.plot(title='Überschuss Standort A1', ylabel='Überschuss in kWh', xlabel=' Zeitstempel in 1/4 h', legend=False, grid=True, xlim=(0,652))
    surplus_A2.plot(title='Überschuss Standort A2', ylabel='Überschuss in kWh', xlabel=' Zeitstempel in 1/4 h', legend=False, grid=True, xlim=(0,652))
    
    plt.show()
    
    fig, ax = plt.subplots()

    twin1 = ax.twinx()
    p1 = ax.plot(df1.index, df1['speicher_SOC'], label='soc',color="b")
    p2 = twin1.plot(df_eprice.index, df_eprice['preis abs'], label='strompreis',color='r')

    ax.set_xlabel("Zeitstempel in 1/4 h")
    ax.set_ylabel("Speicherstand in kWh")
    twin1.set_ylabel("Strompreis in €/kWh")
    ax.yaxis.label.set_color('b')
    twin1.yaxis.label.set_color('r')

    ax.set_xlim(0, 652)
    ax.set_ylim(0, 15000)
    twin1.set_ylim(-0.6, 0.5)

    plt.show()
        