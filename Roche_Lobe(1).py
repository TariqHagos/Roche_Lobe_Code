# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 07:32:45 2021

@author: tariq
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import scipy.constants as con


# Sun and Earth Motion
G = con.G
m = 1.989*10**30
M1 = 1*m
M2 = 1*m
M = M1 + M2
# Distance in AU : semi-major axis
a = 1
# Eccentricity Of Orbit
e = 0.2# 0.0167
# Orbital Period
P = np.sqrt((4*(np.pi**2)*a**3)/(G*M))
# radius at apocentre
ra = a*(1+e)
# relative velocity at apocentre
va = np.sqrt((G*M*((2/ra)-(1/a))))
# stellar positions at apocentre
r1 = (M2/M) * ra
r2 = (-M1/M) * ra
# stellar velocities at apocentre
v1= M2/M * va
v2 = -M1/M * va
tspan = (0,P) # loop for 2 orbits
t_max = P
maxdt = P * 0.01 # this NEEDS ADJUSTMENT UNTIL CONVERGENCE
atol = 1e-7 # perhaps make this small
rtol = 1e-7 # make this small

# Rate Of Mass Transfer of M1
# starting mass transfer rate when if the donor star fills roche lobe
m0 = 1*10**-20*m
#Mass Ratio
q = (M1/M2) # place inside binary orbit

pi = np.linspace(0,2*np.pi,10000)
zero = np.linspace(0,0,10000)


# Initial Conditions
init1 = [r1,0,0,-v1,r2,0,0,-v2,0,0]



def binaryorbit(t,g):
    x1,y1,vx1,vy1,x2,y2,vx2,vy2,mass_1,mass_2 = g
    rx = x1 - x2
    ry = y1 - y2
    r = np.sqrt(rx**2 + ry**2)
    rL1 = (r*0.49*q**(2/3))/(0.6*q**(2/3)+np.log(1+q**(1/3)))
    Hp = 0.1*rL1
    dM2dt =  m0*np.exp((0.31-rL1)/Hp)
    dM1dt = -(m0*np.exp((0.31-rL1)/Hp))
    Px1 = mass_1*(vx1 - vx2)
    Px2 = mass_1*(vx1 - vx2)
    Py1 = mass_2*(vy1 - vy2)
    Py2 = mass_2*(vy1 - vy2)
    drxdt1 = vx1 
    drydt1 = vy1
    drxdt2 = vx2
    drydt2 = vy2  
    dvxdt1 = (-G*(M2+mass_2)*rx/r**3) + Px1
    dvydt1 = (-G*(M2+mass_2)*ry/r**3) + Py1
    dvxdt2 = (-G*(M1-mass_1)*(-rx)/r**3)  + Px2
    dvydt2 = (-G*(M1-mass_1)*(-ry)/r**3) + Py2
    return[drxdt1,drydt1,dvxdt1,dvydt1,drxdt2,drydt2,dvxdt2,dvydt2,dM1dt,dM2dt]

soln1 = solve_ivp(binaryorbit,
                  tspan,
                  init1,
                  first_step = maxdt,
                  atol = atol,
                  rtol = rtol,
                  method = 'Radau', # RK45, RK23, DOP853, Radau, LSODA, BDF
                  max_step = maxdt,
                  t_eval=np.linspace(0,P,10000))
x1,y1,vx1,vy1,x2,y2,vx2,vy2,mass_1,mass_2 = soln1.y


def rocheloberadius(q):
    rx = x1 - x2
    ry = y1 - y2
    r = np.sqrt(rx**2 + ry**2)
    rL = (r*0.49*q**(2/3))/(0.6*q**(2/3)+np.log(1+q**(1/3)))
    return[rL]



def masstransfer(q):
    rx = x1 - x2
    ry = y1 - y2
    r = np.sqrt(rx**2 + ry**2)
    rL1 = (r*0.49*q**(2/3))/(0.6*q**(2/3)+np.log(1+q**(1/3)))
    Hp = 0.1*rL1
    dM1dt = -m0*np.exp((0.1-rL1)/Hp)
    dM2dt =  m0*np.exp((0.1-rL1)/Hp)
    return[dM1dt,dM2dt]
    
X = []
def orbitcheckj():
    v1 = np.array([vx1,vy1,zero])
    r1 = np.array([x1,y1,zero])
    for i in zero:
        t = int(i)
        x = [v1[0][t],v1[1][t],v1[2][t]]
        y = [r1[0][t],r1[1][t],r1[2][t]]
        j1 = np.dot(M1,np.cross(y,x))
    v2 = np.array([vx2,vy2,zero])
    r2 = np.array([x2,y2,zero])
    for i in zero:
        t = int(i)
        x = [v2[0][t],v2[1][t],v2[2][t]]
        y = [r2[0][t],r2[1][t],r2[2][t]]
        j2 = np.dot(M2,np.cross(y,x))
        jorb = abs(np.add(j1,j2))
        X.append(jorb[2])



orbitcheckj()
rocheloberadius(q)
A = masstransfer(q)[0]
B = masstransfer(q)[1]


plt.plot(x1,y1,color = 'Red',Label = 'Mass 1')
plt.plot(x2,y2,'--',color = 'blue',Label= 'Mass 2')
plt.axes().set_aspect('equal')
plt.xlabel('Distance (au)')
plt.ylabel('Distance (au)')
plt.title('e = 0.8')
plt.legend(loc = 'upper right',frameon = 'false')

plt.show()

plt.plot(pi,rocheloberadius(q)[0],)
plt.axhline(y=0.20, color='b', linestyle='--')
plt.xlabel('Orbital Phase')
plt.ylabel('Roche Lobe Radius')
plt.show()

plt.plot(pi,A)
plt.xlabel('Orbital Phase')
plt.ylabel('Mass transfer')
plt.show()

plt.plot(pi,B)
plt.xlabel('Orbital Phase')
plt.ylabel('Mass transfer')
plt.show()

plt.plot(pi,X,color = 'blue',label = 'e = 0.2')
plt.legend(loc = 'upper right',frameon = 'false')
plt.xlabel('Orbital Phase')
plt.ylabel('Angular Momentum ($Kgm^{2}$/s)')
plt.savefig('Orbit0.2.pdf')
plt.show()













   



















