import matplotlib.pyplot as plt
class Logger():
    def __init__(self,log_path = None):
        self.log_path = log_path
        self.data = {}

    @staticmethod
    def write(loggerType, message):
        print(loggerType + ": " + message + "\n")
    
    def add_datapoint (self,key, x, y):
        if key in self.data:
            self.data[key]['X'].append(x)
            self.data[key]['Y'].append(y)
        else:
            Logger.write("ERROR", "This key doesn't exist in logger.")
    
    def create_dataholder(self,key):
        self.data[key] = { 'X': [], 'Y': []}
    
    def init_plot(self):
        _, self.ax = plt.subplots()
    
    def plot(self, key, plot_style = None, linestyle = ' '):
        if key in self.data:
            self.ax.plot(self.data[key]['X'], self.data[key]['Y'], marker =plot_style, 
            linestyle = linestyle, label = key)
        else:
            Logger.write("ERROR", "This key doesn't exist in logger.")
    
    def show(self):
        legend = self.ax.legend(loc='upper center', shadow=True, fontsize='x-large')
        plt.show()

        