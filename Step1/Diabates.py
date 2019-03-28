


from sklearn.datasets import load_diabetes

import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt


# the points are as following
"""
    1. Imputing missing values before building an estimator
"""



"""
    information of description can be fetched in datasets/descr/
"""
def Diabetes2DataFrame():

    # return a Bunch,so data is a Bunch
    data = load_diabetes()
    X, y = data.data, data.target

    Attributes = ["Age", "Sex", "BMI", "ABP",
                  "S1", "S2", "S3", "S4", "S5", "S6"]
    Target = ["disease_p"]
    values = np.concatenate([X, y.reshape(-1,1)], axis=-1)

    Diabetes = pd.DataFrame(data=values, columns=Attributes+Target)

    return Diabetes

def DiabetesVisualization():

    Diabetes = Diabetes2DataFrame()

    scatter_matrix(Diabetes, alpha=0.5, figsize=(16, 16*0.618))
    plt.savefig("Figures/DiabetesPairCor.pdf")

    return 0












if __name__ == "__main__":

    print("")
    DiabetesVisualization()