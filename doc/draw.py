import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import yaml
yaml_file=open('../config.yaml','r',encoding='utf-8')
cfg=yaml.safe_load(yaml_file)
path=cfg["test"]["dataset_path"]

print("This is the trajectory for data in "+path)

f = open(path+"traj.txt")
try:
    truex=[]
    truey=[]
    truth=open(path+"groundtruth.txt")
    for line in truth:
        data=line.split()
        truex.append(float(data[1]))
        truey.append(float(data[2]))
        ex=plt.subplot(211)
    plt.plot(truey,truex)
except:
    pass
x = []
y = []
z = []
for line in f:
    if line[0] == '#':
        continue
    data = line.split()
    x.append( float(data[0] ) )
    y.append( float(data[1] ) )
    z.append( float(data[2] ) )
ax = plt.subplot(212)
plt.plot(x,z)
plt.show()