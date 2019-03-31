

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

def load_IO():

    IO_Attributes = ["2184302", "2184303", "2184304", "2184305",
                     "2184306", "2189008", "2189010", "2189144"]
    IO_Attributes.remove("2189144")
    IO_target = ["70"]

    # something is wrong with columns[27] -- 2200000
    data = pd.read_csv("Dataset/201812.csv", dtype={"2200000": object,
                                                    })

    IO_Attributes_slct = []
    for col in IO_Attributes:
        #print(col, "\t", data[col].dtype)
        if data[col].dtype == np.float64 and data[col].var():
            IO_Attributes_slct.append(col)
        else:
            continue
    # they are bad attributes
    #OS_Attributes_slct.remove("3000301")
    #OS_Attributes_slct.remove("3000302")
    #print(OS_Attributes_slct+OS_target)

    IO = data[IO_Attributes_slct+IO_target]
    num = IO.shape[0]

    # delete the corresponding nan health score
    health_score = ["health_score"]
    index_mask = np.where(np.isnan(data[health_score].values) == False)[0]
    #print("index_mask.shape", index_mask.shape)

    IO_slct = IO.loc[index_mask]
    num_slct = IO_slct.shape[0]
    #print("num = ", num, "num_slct = ", num_slct)
    #print("OS_slct.shape:\t", OS_slct.shape)
    #print("IO_slct[IO_target] = ", IO_slct[IO_target].values.max(), "\t", IO_slct[IO_target].values.min())
    zeros_count = np.where(IO_slct[IO_target].values == 5)[0]
    print("zeros_count = ", zeros_count)
    #print(IO_slct[IO_target])

    # some error occurs when one column is full with zeros
    #scatter_matrix(IO_slct, alpha=0.5, figsize=(16, 16*0.618))
    #plt.savefig("Figures/IO_PairCor.png")

    return IO_slct

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

def clean_IO():

    IO = load_IO()
    print(IO.columns)
    print(IO.values.shape)

    num_attribs = IO.columns[:-1]
    target = IO.columns[-1]
    IO_y = IO[target].values.reshape(-1, 1)

    cat_attribs = []

    num_pipeline = Pipeline([
        ("selector", DataFrameSelector(num_attribs)),
        ("imputer", Imputer(strategy="median")),
        ("std_scaler", StandardScaler()),
    ])

    full_pipe = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline)
    ])

    IO_X = full_pipe.fit_transform(IO)
    print("IO_X.shape = ", IO_X.shape)

    values = np.concatenate([IO_X, IO_y], axis=-1)
    IO = pd.DataFrame(values, columns=IO.columns)

    print("IO[target] = ", IO[target].max(), "\t", IO[target].min())

    #scatter_matrix(IO, alpha=0.5, figsize=(16, 16*0.618), diagonal="kde")
    #plt.savefig("Figures/IO_PairCor_scaled.png")

    return IO












if __name__ == "__main__":
    print("")
    load_IO()