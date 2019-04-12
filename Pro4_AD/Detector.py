


"""
    Data Preprocess & Generate
"""

import pandas as pd
import matplotlib.pyplot as plt


def load_AD():

    AD_Attribs = ["timestamp", "value", "label", "KPI ID"]

    AD = pd.read_csv("Dataset/phase1/train.csv")
    for col in AD_Attribs:
        #print(col, type(AD[col][0]))
        pass

    grouped = AD.groupby(by="KPI ID")
    KPI_names = []
    for name, group in grouped:
        #print(name, group.shape)
        KPI_names.append(name)

    print("", grouped.get_group(KPI_names[0]).shape)
    print("", grouped.get_group(KPI_names[1]).columns)

    return grouped.get_group(KPI_names[0])

def vis_AD():

    KPI = load_AD()
    grouped = KPI.groupby(by="label")
    KPI_cat = []
    for name, group in grouped:
        #print(name, group.shape)
        KPI_cat.append(name)
        pass

    KPI_norm = grouped.get_group(KPI_cat[0])
    KPI_anomaly = grouped.get_group(KPI_cat[1])

    fig = plt.figure(figsize=(16, 16*0.618))
    ax = fig.add_subplot(111)
    ax.plot(KPI["timestamp"], KPI["value"], 'b.')
    ax.plot(KPI_anomaly["timestamp"], KPI_anomaly["value"], 'r.')
    plt.show()


    return 0








if __name__ == "__main__":
    print("")
    vis_AD()




# END OF FILE
