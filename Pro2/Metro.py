

"""
    Data Preprocess & Generate
"""

import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np

from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler

def load_Metro():

    Metro_Attributes = ["time", "lineID", "stationID","deviceID",
                        "status", "userID", "payType"]
    Metro_target = []

    Metro_Attributes.remove("lineID")
    Metro_Attributes.remove("deviceID")
    Metro_Attributes.remove("userID")
    Metro_Attributes.remove("payType")

    # something is wrong with columns[27] -- 2200000
    data = pd.read_csv("Dataset/Metro_testA/testA_record_2019-01-28.csv")

    Metro = data[Metro_Attributes]
    num = Metro.shape[0]
    #for col in Metro_Attributes:
    #    print(col, Metro[col].dtype)

    print("num = ", num)
    print("Metro.shape:\t", Metro.shape)

    # some error occurs when one column is full with zeros
    #scatter_matrix(OS_slct, alpha=0.5, figsize=(16, 16*0.618))
    #plt.savefig("Figures/OS_PairCor.png")

    return Metro

def clean_Metro():

    Metro_stations = 81

    Metro_InOut = np.zeros((2, Metro_stations, int(24*60/10)), dtype=np.double)
    #Metro_in = np.zeros((Metro_stations, 24*60/10), dtype=np.int64)
    #Metro_out = np.zeros((Metro_stations, 24*60/10), dtype=np.int64)

    Metro_values = load_Metro().values
    num = Metro_values.shape[0]
    #print("", Metro_values[:, 0].dtype)
    Metro_time = Metro_values[:, 0]
    Metro_time = [ele.replace(' ', 'T') for ele in Metro_time]
    Metro_time = np.array(Metro_time, dtype='datetime64[s]')

    Metro_start = np.array(["2019-01-28T00:00:00"], dtype='datetime64[s]')
    print("Metro_start[0] = ", Metro_start[0])

    #print("", Metro_time[5] - Metro_start[0])
    #print("", int((Metro_time[1440] - Metro_start[0])/np.timedelta64(60*10, 's')))
    #print("", Metro_values[0, 2], Metro_values[0, 2] == 0)

    for i in range(num):
        timestap = Metro_time[i]
        stationID = int(Metro_values[i, 1])
        status = int(Metro_values[i, 2])
        
        time_diff = int((timestap - Metro_start)/np.timedelta64(60*10, 's'))

        Metro_InOut[status, stationID, time_diff] += 1

    print("", np.max(Metro_InOut))

    np.save("Dataset/Metro_testA/Stations81/Metro_InOut", Metro_InOut)

    return 0


# visualization, better considering the connection among 81 stations
def vis_Metro():

    Metro_InOut = np.load("Dataset/Metro_testA/Stations81/Metro_InOut.npy")
    #print("", np.sum(Metro_InOut))

    times = np.arange(Metro_InOut.shape[-1])

    stationID = 16

    fig = plt.figure()

    ax = fig.add_subplot(211)
    ax.plot(times, Metro_InOut[0, stationID], 'b')
    y = Metro_InOut[0, stationID]
    new_y = np.zeros_like(y)
    new_y[0] = y[0]
    alpha = 0.36
    for i in range(1, y.shape[0]):
        new_y[i] = alpha*new_y[i-1] + (1-alpha)*y[i]

    ax.plot(times, new_y, 'r')

    ax2 = fig.add_subplot(212)
    ax2.plot(times, Metro_InOut[1, stationID])

    plt.show()



    return 0


def generateMetro():

    Metro_InOut = np.load("Dataset/Metro_testA/Stations81/Metro_InOut.npy")
    #print("", np.sum(Metro_InOut))

    # smoothing
    alpha = 0.36
    for status in range(2):
        for stationID in range(81):
            y = Metro_InOut[status, stationID]
            y_new = np.zeros_like(y, dtype=np.double)
            y_new[0] = y[0]
            for i in range(1, y.shape[0]):
                y_new[i] = alpha*y_new[i-1] + (1-alpha)*y[i]

            Metro_InOut[status, stationID] = y_new


    testA = pd.read_csv("Dataset/Metro_testA/testA_submit_2019-01-29.csv")
    testA_Attribs = ["stationID", "startTime", "endTime", "inNums", "outNums"]


    #
    testA["inNums"] = Metro_InOut[1].reshape(-1)
    testA["outNums"] = Metro_InOut[0].reshape(-1)

    testA["stationID"] = testA["stationID"].astype(int)
    testA["startTime"] = testA["startTime"].astype(str)
    testA["endTime"] = testA["endTime"].astype(str)
    testA["inNums"] = testA["inNums"].astype(np.double)
    testA["outNums"] = testA["outNums"].astype(np.double)

    for col in testA_Attribs:
        print(col, type(testA[col][0]))

    print("", testA["inNums"][244])
    print("", testA["outNums"][244])
    print("", testA["endTime"][244])




    testA.to_csv("Dataset/Metro_testA/results_submit/testA_submit_2019-01-29.csv",
                 encoding='utf-8', index=False)


    return 0









if __name__ == "__main__":
    print("")
    #clean_Metro()
    #vis_Metro()
    generateMetro()




# END OF FILE