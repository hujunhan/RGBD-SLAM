import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d

f = open("../data/dormitory/traj.txt")
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
ax = plt.subplot( 111, projection='3d')
ax.plot(x,y,z)
plt.show()