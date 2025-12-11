import matplotlib.pyplot as plt
import GraphPlanner.sailboat_polar as sp

if __name__ == '__main__':
    polar_file = 'Data/polar.csv'
    polar = sp.SailboatPolar(polar_file)
    sp.PolarVisualizer(polar).visualize()
    plt.show()