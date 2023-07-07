# sommerakademie
    df1 = pd.DataFrame()
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
                #if len(index)
                    df1[name + cstring] = [item[2] for item in newList]
                else:
                    df1[name] = [item[2] for item in newList]