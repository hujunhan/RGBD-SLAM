import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import yaml

yaml_file = open('../config.yaml', 'r', encoding='utf-8')
cfg = yaml.safe_load(yaml_file)
path = cfg["test"]["dataset_path"]
datasize = int(cfg["test"]["data_size"])
print("This is the trajectory for data in " + path)

f = open(path + "traj.txt")
try:
    truex = []
    truey = []
    truth = open(path + "groundtruth.txt")
    count = 0
    line1=truth.readline()
    data=line1.split()
    x0=float(data[2])
    y0=float(data[3])
    print("x0={},y0={}:".format(x0,y0))
    for line in truth:
        data = line.split()
        truex.append(-(float(data[2]) - x0) )
        truey.append((float(data[3]) - y0) )

    plt.plot(truey, truex, color='red', label='groundtruth')
except:
    pass
x = []
y = []
z = []
for line in f:
    if line[0] == '#':
        continue
    data = line.split()
    x.append(float(data[0])*100)
    y.append(float(data[1]))
    z.append(-float(data[2])*100)
plt.plot(x, z, color='black', label='estimation')
plt.legend(loc='upper left')
plt.show()
