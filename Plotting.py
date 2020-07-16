from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf

def plotData(dataset1, dataset2 = None, title = ""):
    # Input title as dataset1 then dataset2
    pyplot.plot(dataset1, color = "red")
    if title is not "":
        pyplot.title(title)
    if dataset2 is not None:
        pyplot.plot(dataset2, color = "blue")
        if title is not "":
            pyplot.title(title + " (red & blue)")
    pyplot.show()
    return

def plotACF(dataset):
    plot_acf(dataset)
    pyplot.show()
    return

def plotPACF(dataset):
    plot_pacf(dataset)
    pyplot.show()
    return