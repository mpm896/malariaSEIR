''' 
Malaria SEIR model for PHCP project
Parameters based primarily off Tanzania

@author: Matt Martinez
'''

from scipy.integrate import odeint
import numpy as np
import math
import matplotlib.pyplot as plt
from dataclasses import dataclass

### Global constants ###

# Start, stop times and interval in WEEKS
START = 0
STOP = 400
INTERVAL = 10_000

@dataclass
class Human:
    ''' HUMAN CONDITIONS '''

    # Dynamic parameters
    LAMBDA: float               # Force of infection

    S: int | np.ndarray
    E: int | np.ndarray
    I: int | np.ndarray
    R: int | np.ndarray

    H_total: int                # Total number of humans
    PREV: float                 # Prevalence of infection
        
    # Differential equations
    dS: float
    dE: float
    dI: float
    dR: float

    # Initial population parameters
    S_0: int       = 59_599_999   # Susceptible
    E_0: int       = 0            # Exposed
    I_0: int       = 1            # Infected
    R_0: int       = 0            # Recovered

    # Transmission parameters
    MU: float       = 0.0003    # Death/birth rate per host per week (avg lifespan = 64.48 year)
    CFR: float      = 0.0033    # Case fatality ratio
    PI: float       = 0.1       # Probability of infection in case of infectious bite
    EPSILON: float  = 7/12      # Rate of becoming infectious / host / week given an average incubation period of 12 days
    DELTA: float    = 0.024     # Rate of recovery given an average time spent infectious of ~40 weeks
    SIGMA: float    = 0.0038    # Rate of loss of immunity per host per week (~260 weeks)


@dataclass
class Mosquito:
    ''' MOSQUITO CONDITIONS '''
    
    # Dynamic parameters
    LAMBDA: float                   # Force of infection

    S: int | np.ndarray
    E: int | np.ndarray
    I: int | np.ndarray

    M_total: int                # Total number of mosquitos
    PREV: float                     # Prevalence of infection

    # Differential equations
    dS: float
    dE: float
    dI: float

    # Initial population parameters
    S_0: int          = 119_199_999   # Susceptible
    E_0: int          = 0             # Exposed
    I_0: int          = 1             # Infected

    # Transmission parameters
    a: int          = 70    # Bite rate (bites per mosquito per week) - from 10/day
    MU: float       = 0.35  # Death/birth rate per mosquito per week (avg lifespan ~3 weeks)
    PI: float       = 0.01  # Probability of infection in case of bite of infectious human
    EPSILON: float  = 7/9   # Rate of becoming infectious / host / week given an average incubation period of 9 days


def f(y: list, t: float) -> list:
    ''' 
    ODE function for SEIR model
    Args:
        y: list[Human.S, Human.E, Human.I, Human.R, 
                Mosquito.S, Mosquito.E, Mosquito.I]
        t: time
    Returns:
        dy/dt: list[dS/dt, dE/dt, dI/dt, dR/dt]
    '''

    # Set human and mosquito parameters
    Human.S, Human.E, Human.I, Human.R = y[0:4]
    Human.H_total = Human.S + Human.E + Human.I + Human.R
    Human.PREV = ((Human.I + Human.E) / Human.H_total) * 100

    Mosquito.S, Mosquito.E, Mosquito.I = y[4:]
    Mosquito.M_total = Mosquito.S + Mosquito.E + Mosquito.I
    Mosquito.PREV = ((Mosquito.I + Mosquito.E) / Mosquito.M_total) * 100

    # Define forces of infection below dataclass since they rely on each other
    Human.LAMBDA = ( (Mosquito.a * Mosquito.M_total / Human.H_total)  # Force of infection in Human
                        * (Mosquito.I / Mosquito.M_total)
                        * Human.PI)

    Mosquito.LAMBDA = (Mosquito.a * (Human.I / Human.H_total) * Mosquito.PI)      # Force of infection in Mosquito

    # Define the basic reproductive ratio, R0
    R0 = math.sqrt(((Mosquito.EPSILON / (Mosquito.EPSILON + Mosquito.MU)) 
                    * (Mosquito.a * Human.PI) * (1 / Mosquito.MU)) 
                    * ((Human.EPSILON / (Human.EPSILON + Human.MU)) 
                    * ((Mosquito.a * Mosquito.PI * Mosquito.M_total) / Human.H_total) 
                    * (1 / (Human.MU + Human.DELTA))))

    # Differential equations
    Human.dS = ((Human.MU * Human.H_total)
                + (Human.CFR * Human.DELTA * Human.I)
                + (Human.SIGMA * Human.R)
                - (Human.LAMBDA * Human.S)
                - (Human.MU * Human.S))
    
    Human.dE = ((Human.LAMBDA * Human.S)
                - (Human.MU * Human.E)
                - (Human.EPSILON * Human.E))
    
    Human.dI = ((Human.EPSILON * Human.E)
                - (Human.DELTA * Human.I)
                - (Human.MU * Human.I))
    
    Human.dR = (((1 - Human.CFR) * Human.DELTA * Human.I)
                - (Human.SIGMA * Human.R)
                - (Human.MU * Human.R))
    

    Mosquito.dS = ((Mosquito.MU * Mosquito.M_total)
                   - (Mosquito.LAMBDA * Mosquito.S)
                   - (Mosquito.MU * Mosquito.S))
    
    Mosquito.dE = ((Mosquito.LAMBDA * Mosquito.S)
                   - (Mosquito.MU * Mosquito.E)
                   - (Mosquito.EPSILON * Mosquito.E))
    
    Mosquito.dI = ((Mosquito.EPSILON * Mosquito.E)
                   - (Mosquito.MU * Mosquito.I))

    return [Human.dS, Human.dE, Human.dI, Human.dR, Mosquito.dS, Mosquito.dE, Mosquito.dI]


def main():
    '''
    Set the parameters, solve the ODE, and plot the results
    '''
    t = np.linspace(start=START, stop=STOP, num=INTERVAL)  # Domain / time range
    y0 = [Human.S_0, Human.E_0, Human.I_0, Human.R_0, Mosquito.S_0, Mosquito.E_0, Mosquito.I_0]  # Initial conditions
    y = odeint(f, y0, t)  # Solve ODEs

    # Set the human and mosquito parameters with the ODE results
    Human.S = y[:, 0]
    Human.E = y[:, 1]
    Human.I = y[:, 2]
    Human.R = y[:, 3]
    Mosquito.S = y[:, 4]
    Mosquito.E = y[:, 5]
    Mosquito.I = y[:, 6]

    # Plot the results
    plt.plot(t, Human.S, 'r', label='hS(T)')
    plt.plot(t, Human.I, 'b', label='hI(T)')
    plt.plot(t, Human.E, 'g', label='hE(T)')
    plt.plot(t, Human.R, 'k', label='hR(T)')
    plt.legend()
    plt.show()

    plt.plot(t, Mosquito.S, 'r', label='mS(T)')
    plt.plot(t, Mosquito.I, 'b', label='mI(T)')
    plt.plot(t, Mosquito.E, 'g', label='mE(T)')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()