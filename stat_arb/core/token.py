import numpy as np

class Token:
    def __init__(self, address: str, data: list):
        self.address = address
        self.data = data

    def calc_log_returns(self):
        return np.log(np.array(self.data[1:]) / np.array(self.data[:-1]))

    def normalize(self):
        # scale values
        pass

    def pca(self):
        # apply PCA
        pass

    def visualize(self):
        # plot
        pass

    def info(self):
        return {
            'address': self.address,
            'length': len(self.data)
        }
