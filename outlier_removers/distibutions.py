import matplotlib.pyplot as plt
import scipy.stats
from scipy.stats import norm
import pandas as pd
import numpy as np


class NormalDistribution:
    def __init__(self, data_csv):
        self.means = {c: np.mean(data_csv[c]) for c in data_csv.columns}
        self.stds = {c: np.std(data_csv[c]) for c in data_csv.columns}
        self.distibutions = {c: norm(np.mean(data_csv[c]), np.std(data_csv[c])) for c in data_csv.columns}

    def predict(self, column, values):
        result = []
        for v in values:
            result.append(self.distibutions[column].pdf(v) / self.distibutions[column].pdf(self.means[c]))
        return np.array(result, dtype=float)
