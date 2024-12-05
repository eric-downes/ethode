import numpy as np

from examples import *

ESU_y0 = ComplexESU(
    ic = 
    tinfo: tuple[Num, Num, Num]
    params: Params
    


y = 1
fs = [10, 7.5, 5, 2.5, 2, 1.5, 1.25, 1, .9, .75, .5, .25, .1, .01]
rs = np.arange(0,1.1,.1)
bs = np.arange(0,1,.1)

data = {}
for f in fs:
    for r in rs:
        for b in bs:
            print(f'starting {(y,f,r,b)}')            
            escb = CompelxESU(
                ic = (1, .3, .7),
                tinfo = (0, 10**4, 1),
                params = (r,f,y,b))
            data[(r,f,y,b)] = escb.sim()

ends = {}
for k,v in data.items():
    e = v[-1]
    ends[k] = e[1] / (e[1] + e[2])

rf = np.ndarray(dtype=float, shape=(len(rs), len(fs)))
rb = np.ndarray(dtype=float, shape=(len(rs), len(bs)))
fb = np.ndarray(dtype=float, shape=(len(fs), len(bs)))
ri = {r:i for i,r in enumerate(rs)}
fi = {f:i for i,f in enumerate(fs)}
bi = {r:i for i,r in enumerate(bs)}

for (r, f, y, b), e in ends.items():
    rf[ ri[r], fi[f] ] = e
    rb[ ri[r], bi[b] ] = e
    fb[ fi[f], bi[b] ] = e
ax1 = plt.contour(rs, fs, rf.T)
ax1.set_title('S/E(r,f)')
ax2 = plt.contour(rs, bs, rb.T)
ax2.set_title('S/E(r,b))')
ax3 = plt.contour(fs, bs, fb.T)
ax3.set_title('S/E(f,b)')

    
