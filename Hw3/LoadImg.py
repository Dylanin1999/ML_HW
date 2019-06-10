import numpy as np
import pandas as pd



def LoadData(filename):
    file = pd.read_csv(filename, encoding='utf-8')
    count = len(file['id'])
    data_list = []
    #for num in range(count):
      #  data_list.append([])
    for i in range(count):
        feature = file['feature'][i].split(' ')
        feature = list(map(lambda x: float(x), feature))
        data_list.append(feature)
        print("process: ", i)
    data_list = np.array(data_list)
    print(data_list)
    return data_list, file['id']

#LoadData('data/train.csv')


