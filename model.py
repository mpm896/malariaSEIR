''' 
Malaria SEIR model for PHCP project
Parameters based primarily off Tanzania

@author: Matt Martinez
'''

from scipy.integrate import solve_ivp  # Prefer this over odeint because can use Runge-Kutta 4th order of integrating
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

    PREV = []                   # Prevalence of infection
        
    # Differential equations
    dS: float
    dE: float
    dI: float
    dR: float

    # Initial population parameters
    Total = 1                                       # Default, since this will be based on fractions of the population
    TotalPop: int = 59_600_000.                     # Total number of humans
    S_0: int | float       = 59_599_999. / TotalPop # Susceptible
    E_0: int | float       = 0                      # Exposed
    I_0: int | float       = 1 - S_0                # Infected
    R_0: int | float       = 0                      # Recovered
    

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

    
    PREV = []                       # Prevalence of infection

    # Differential equations
    dS: float
    dE: float
    dI: float

    # Initial population parameters
    Total = 1                                           # Default, since this will be based on fractions of the population
    TotalPop: int = 119_200_000.                        # Total number of mosquitos
    S_0: int | float          = 119_199_999. / TotalPop # Susceptible
    E_0: int | float          = 0                       # Exposed
    I_0: int | float          = 1 - S_0                 # Infected

    # Transmission parameters
    a: int          = 70    # Bite rate (bites per mosquito per week) - from 10/day
    MU: float       = 0.35  # Death/birth rate per mosquito per week (avg lifespan ~3 weeks)
    PI: float       = 0.01  # Probability of infection in case of bite of infectious human
    EPSILON: float  = 7/9   # Rate of becoming infectious / host / week given an average incubation period of 9 days


def f(t: float, y: list) -> list:
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
    Mosquito.S, Mosquito.E, Mosquito.I = y[4:]

    # Define forces of infection below dataclass since they rely on each other
    Human.LAMBDA = ( (Mosquito.a * Mosquito.Total / Human.Total)  # Force of infection in Human
                        * (Mosquito.I / Mosquito.Total)
                        * Human.PI)

    Mosquito.LAMBDA = (Mosquito.a * (Human.I / Human.Total) * Mosquito.PI)  # Force of infection in Mosquito

    # Differential equations
    Human.dS = ((Human.MU * Human.Total)
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
    

    Mosquito.dS = ((Mosquito.MU * Mosquito.Total)
                   - (Mosquito.LAMBDA * Mosquito.S)
                   - (Mosquito.MU * Mosquito.S))
    
    Mosquito.dE = ((Mosquito.LAMBDA * Mosquito.S)
                   - (Mosquito.MU * Mosquito.E)
                   - (Mosquito.EPSILON * Mosquito.E))
    
    Mosquito.dI = ((Mosquito.EPSILON * Mosquito.E)
                   - (Mosquito.MU * Mosquito.I))

    return [Human.dS, Human.dE, Human.dI, Human.dR, 
            Mosquito.dS, Mosquito.dE, Mosquito.dI]

def calc_prevalence(t: np.array, e: np.array, i: np.array, Obj):
    ''' 
    Function for SEIR model to solve prevalence parameter
    Will do it for the time points that were solved by the ODE algorithm
    Args:
        t [np.array]: times from ODE solution
        e: [np.array]: Exposed, from ODE solution
        i: [np.array]: Infected, from ODE solution
        Obj: dataclass with prevalence: Dataclass object   
    Returns:
        None, appends Object.PREV with prev and set as np.array
    '''

    for k in range(t.size):
        Obj.PREV.append((i[k] + e[k]))

    Obj.PREV = np.asarray(Obj.PREV)

def calc_incidence(t: np.array, e: np.array, i: np.array) -> np.array:
    ''' 
    Function for SEIR model to solve incidence parameter
    Args:
        t [np.array]: times from ODE solution
        e: [np.array]: Exposed, from ODE solution
        i: [np.array]: Infected, from ODE solution
    Returns:
        inc: np.array
    '''
    pass

def makeplots(t: np.array, s: np.array, e: np.array = None, i: np.array = None, r: np.array = None) -> None:
    ''' Plot the data '''

    # Must contain at least SEI or SIR, in that order. Can also be SEIR.
    plt.plot(t, s, lw=2, label='Susceptible')
    if e is not None: plt.plot(t, e, lw=2, label='Exposed')
    if i is not None: plt.plot(t, i, lw=2, label='Infected')
    if r is not None: plt.plot(t, r, lw=2, label='Recovered')

    plt.xlabel('Time /weeks')
    plt.ylabel('Fraction of population')
    plt.ylim(0, 1.05)
    plt.title('Susceptible and Recovered Populations')
    plt.grid()
    plt.legend()
    plt.show()



def main():
    ''' Set the parameters, solve the ODE, and plot the results '''

    # Initial conditions
    y0 = [Human.S_0, Human.E_0, Human.I_0, Human.R_0, Mosquito.S_0, Mosquito.E_0, Mosquito.I_0]

    # Solve the ODE
    y = solve_ivp(f, (START, STOP), y0)

    # Define and solve for the basic reproductive ratio, R0
    R0 = math.sqrt(((Mosquito.EPSILON / (Mosquito.EPSILON + Mosquito.MU)) 
                    * (Mosquito.a * Human.PI) * (1 / Mosquito.MU)) 
                    * ((Human.EPSILON / (Human.EPSILON + Human.MU)) 
                    * ((Mosquito.a * Mosquito.PI * Mosquito.Total) / Human.Total) 
                    * (1 / (Human.MU + Human.DELTA))))
    
    print(f"R0: {R0}")

    # Set the human and mosquito parameters with the ODE results
    t = y.t
    Human.S = y.y[0, :]
    Human.E = y.y[1, :]
    Human.I = y.y[2, :]
    Human.R = y.y[3, :]
    Mosquito.S = y.y[4, :]
    Mosquito.E = y.y[5, :]
    Mosquito.I = y.y[6, :]

    # Calculate the prevalence based on the ODE solution times
    calc_prevalence(t, Human.E, Human.I, Human)
    calc_prevalence(t, Mosquito.E, Mosquito.I, Mosquito)
    
    # Plot the results
    makeplots(t, Human.S, Human.E, Human.I, Human.R)
    makeplots(t, Mosquito.S, Mosquito.E, Mosquito.I)

    plt.plot(t, Human.E + Human.I, label='Human Prev')
    plt.plot(t, Human.PREV, label='Human Prev')
    plt.plot(t, Mosquito.PREV, label='Mosquito Prev')
    plt.legend()
    plt.show()


    plt.plot(Human.S, Human.I, lw=2, label='s, i trajectory')
    plt.plot([1/R0, 1/R0], [0, 1], '--', lw=2, label='di/dt = 0')
    plt.plot(Human.S[0], Human.I[0], '.', ms=20, label='Initial Condition')
    plt.plot(Human.S[-1], Human.I[-1], '.', ms=20, label='Final Condition')
    plt.title('State Trajectory')
    plt.ylim(0, 1.05)
    plt.xlim(0, 1.05)
    plt.ylabel('Infectious')
    plt.xlabel('Susceptible')
    plt.legend()
    plt.show()



if __name__ == '__main__':
    main()