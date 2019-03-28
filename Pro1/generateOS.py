

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

def load_OS():

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
        #print(col, "\t", data[col].dtype)
        if data[col].dtype == np.float64 and data[col].var():
            OS_Attributes_slct.append(col)
        else:
            continue
    # they are bad attributes
    OS_Attributes_slct.remove("3000301")
    OS_Attributes_slct.remove("3000302")
    #print(OS_Attributes_slct+OS_target)

    OS = data[OS_Attributes_slct+OS_target]
    num = OS.shape[0]

    # delete the corresponding nan health score
    health_score = ["health_score"]
    index_mask = np.where(np.isnan(data[health_score].values) == False)[0]
    #print("index_mask.shape", index_mask.shape)

    OS_slct = OS.loc[index_mask]
    num_slct = OS_slct.shape[0]
    #print("num = ", num, "num_slct = ", num_slct)
    #print("OS_slct.shape:\t", OS_slct.shape)

    # some error occurs when one column is full with zeros
    #scatter_matrix(OS_slct, alpha=0.5, figsize=(16, 16*0.618))
    #plt.savefig("Figures/OS_PairCor.png")

    return OS_slct

"""
    apply the Pipeline tech in sklearn
"""
from sklearn.base import BaseEstimator, TransformerMixin
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribs_names):
        self.attribs_names = attribs_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribs_names].values

def clean_OS():

    OS = load_OS()
    print(OS.columns)
    print(OS.values.shape)

    num_attribs = OS.columns[:-1]
    target = OS.columns[-1]
    OS_y = OS[target].values.reshape(-1, 1)

    cat_attribs = []

    num_pipeline = Pipeline([
        ("selector", DataFrameSelector(num_attribs)),
        ("imputer", Imputer(strategy="median")),
        ("std_scaler", StandardScaler()),
    ])

    full_pipe = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline)
    ])

    OS_X = full_pipe.fit_transform(OS)
    print("OS_X.shape = ", OS_X.shape)

    values = np.concatenate([OS_X, OS_y], axis=-1)
    OS = pd.DataFrame(values, columns=OS.columns)

    scatter_matrix(OS, alpha=0.5, figsize=(16, 16*0.618))
    plt.savefig("Figures/OS_PairCor_scaled.png")

    return OS












if __name__ == "__main__":
    print("")
    clean_OS()