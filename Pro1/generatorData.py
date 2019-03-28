

"""
    Data Preprocess & Generate
"""

import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

import numpy as np


def load_data():

    OS_Attributes = ["3000000", "3000003", "3000004","3000005", "3000006",
                     "3000007", "3000008", "3000009", "3000010","3000200",
                     "3000201", "3000300", "3000301", "3000302"]
    OS_target = ["67"]

    # something is wrong with columns[27] -- 2200000
    data = pd.read_csv("Dataset/201812.csv", dtype={"2200000": object,
                                                    "3000003": object,
                                                    "3000007": np.float64,
                                                    "3000009": np.float64,
                                                    })


    #OS = data[OS_Attributes+OS_target]
    OS_Attributes_slct = []
    for col in OS_Attributes:
        print(col, "\t", data[col].dtype)
        if data[col].dtype == np.float64 and data[col].var():
            OS_Attributes_slct.append(col)
        else:
            continue
    # they are bad attributes
    OS_Attributes_slct.remove("3000301")
    OS_Attributes_slct.remove("3000302")
    print(OS_Attributes_slct+OS_target)

    OS = data[OS_Attributes_slct+OS_target]
    num = OS.shape[0]

    # delete the corresponding nan health score
    health_score = ["health_score"]
    index_mask = np.where(np.isnan(data[health_score].values) == False)[0]
    print("index_mask.shape", index_mask.shape)

    OS_slct = OS.loc[index_mask]
    num_slct = OS_slct.shape[0]
    print("num = ", num, "num_slct = ", num_slct)
    print("OS_slct.shape:\t", OS_slct.shape)

    # some error occurs when one column is full with zeros

    scatter_matrix(OS_slct, alpha=0.5, figsize=(16, 16*0.618))
    plt.savefig("Figures/OS_PairCor.pdf")

    return 0














if __name__ == "__main__":
    print("")
    load_data()