import numpy as np
import pdb

class ReadAnalysis():

    def __init__(self,
                 filename):
        super(ReadAnalysis, self).__init__()
        self.data = np.loadtxt(filename)

    def get_data(self):
        return self.data

def main():
    fname = 'testing.dat'
    analysisObj = ReadAnalysis(fname)
    pdb.set_trace()

if __name__ == '__main__':
    main()
