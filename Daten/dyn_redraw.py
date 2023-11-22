import matplotlib.pyplot as plt
import numpy


def update_line(hl, new_data):
    hl.set_xdata(numpy.append(hl.get_xdata(), new_data))
    hl.set_ydata(numpy.append(hl.get_ydata(), new_data))
    ax.relim()
    ax.autoscale_view()
    plt.draw()


if __name__ == '__main__':
    hl, ax = plt.plot([], [])
    update_line(hl, [[1, 2, 3], [4, 5, 6]])
