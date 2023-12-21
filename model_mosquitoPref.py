'''
Malaria SEIR mdoel for PHCP project
Parameters based primarily off of Tanzania.

Mosquito preference modeled in as OMEGA, 
the preference of infectious humans by mosquitos

@author: Matthew Martinez
'''
from __future__ import annotations

from dataclasses import dataclass, field
import math
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from scipy.integrate import solve_ivp  # type: ignore # uses Runge-Kutta 4th order of integrating
from typing import Optional

### Globals ###
START = 0
STOP = 10000

DO_SIMULATION = True
PARAM_TO_SIMULATE = "OMEGA"
SIM_START = 0.003
SIM_END = 1
SIM_INCREMENTS = 10

### PARAMETERS ###
PARAMETERS = {
    "Human": {
        "TotalPop": 59_600_000,
        "MU": 0.0003,
        "CFR": 0.0033,
        "PI": 0.1,
        "EPSILON": 7/12,
        "DELTA": 0.024,
        "SIGMA": 0.0038
    },

    "Mosquito": {
        "TotalPop": 119_200_000,
        "a": 70,
        "OMEGA": 1,
        "MU": 0.35,
        "PI": 0.01,
        "EPSILON": 7/9
    }
}


@dataclass
class Agent:
    ''' Default SEIR parameters '''

    # Parameters to be set during ODE solution
    S: Optional[np.ndarray] = None
    E: Optional[np.ndarray] = None
    I: Optional[np.ndarray] = None
    R: Optional[np.ndarray] = None

    dS: Optional[float] = None
    dE: Optional[float] = None
    dI: Optional[float] = None
    dR: Optional[float] = None
    LAMBDA: Optional[float] = None

    # Parameters to be set during initialization
    TotalPop: Optional[int]  = None
    PREV: np.ndarray = field(default_factory=lambda: np.zeros(0))
    Total = 1

    @classmethod
    def from_kwargs(cls, **kwargs) -> Agent:
        """ Set custom parameters from kwargs """
        ret = cls(0, 0, 0, 0)

        # Add kwargs to class
        for k, v in kwargs.items():
            setattr(ret, k, v)
    
        if "TotalPop" in kwargs and kwargs["TotalPop"] is not None:
            cls.TotalPop: int = kwargs["TotalPop"]
            cls.S_0        = (cls.TotalPop - 1) / cls.TotalPop
        else:
            cls.S_0          = 0
        cls.E_0: float       = 0
        cls.I_0: float       = 1 - cls.S_0
        cls.R_0: float       = 0

        return ret
    

def f(t: float, y: list, *args) -> list:
    ''' 
    ODE function for SEIR model
    Args:
        t: time [float]
        y: list[Human.S, Human.E, Human.I, Human.R, 
                Mosquito.S, Mosquito.E, Mosquito.I]
        args: tuple[Human, Mosquito]
    Returns:
        dy/dt: list[dS/dt, dE/dt, dI/dt, dR/dt]
    '''
    # Assign agents from args
    Human, Mosquito = args 

    # Set human and mosquito parameters
    Human.S, Human.E, Human.I, Human.R = y[0:4]
    Mosquito.S, Mosquito.E, Mosquito.I = y[4:]

    # Define forces of infection below dataclass since they rely on each other
    Human.LAMBDA = ((Mosquito.a * Mosquito.Total / Human.Total)  # Force of infection in Human
                        * (Mosquito.I / Mosquito.Total)
                        * Human.PI)

    Mosquito.LAMBDA = (Mosquito.a * (Mosquito.OMEGA * (Human.I / Human.Total)) * Mosquito.PI)  # Force of infection in Mosquito

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


def calc_prevalence(e: np.ndarray, 
                    i: np.ndarray, 
                    Obj: Agent) -> None:
    ''' 
    Function for SEIR model to solve prevalence parameter
    Will do it for the time points that were solved by the ODE algorithm
    Args:
        e: [np.array]: Exposed, from ODE solution
        i: [np.array]: Infected, from ODE solution
        Obj: dataclass with prevalence: Dataclass object
    '''
    Obj.PREV = e + i


def makeplots(t: np.ndarray, 
              s: np.ndarray, 
              e: Optional[np.ndarray] = None, 
              i: Optional[np.ndarray] = None, 
              r: Optional[np.ndarray] = None) -> None:
    ''' Plot the data '''

    # Plot disease dynamics
    plt.plot(t, s, lw=2, label='Susceptible')
    if e is not None: 
        plt.plot(t, e, lw=2, label='Exposed')
    if i is not None: 
        plt.plot(t, i, lw=2, label='Infected')
    if r is not None: 
        plt.plot(t, r, lw=2, label='Recovered')

    plt.xlabel('Time /weeks')
    plt.ylabel('Fraction of population')
    plt.ylim(0, 1.05)
    plt.title('Susceptible and Recovered Populations')
    plt.grid()
    plt.legend()
    plt.show()


def simulate(Obj: Agent,
             param: str, 
             begin: float, 
             end: float, 
             step: int, 
             *args) -> None:
    ''' Simulate changes to one parameter '''
    human, mosquito = args

    # Keep track of parameters
    simulations = {
        "y": list(),
        "R0": list(),
        "PREV": {
            "Human": list(),
            "Mosquito": list()
        }
    }

    # Initial conditions 
    y0 = [human.S_0, human.E_0, human.I_0, human.R_0, 
          mosquito.S_0, mosquito.E_0, mosquito.I_0]
    inc = (end - begin) / step

    for i in range(step+1):
        # Set the parameter
        val = begin + (i * inc)
        setattr(Obj, param, val)

        # Solve the ODE
        y = solve_ivp(f, (START, STOP), y0, args=(human, mosquito))
        simulations["y"].append(y)
        human.S = y.y[0, :]
        human.E = y.y[1, :]
        human.I = y.y[2, :]
        human.R = y.y[3, :]
        mosquito.S = y.y[4, :]
        mosquito.E = y.y[5, :]
        mosquito.I = y.y[6, :]

        simulations["R0"].append(
            math.sqrt(((mosquito.EPSILON / (mosquito.EPSILON + mosquito.MU)) 
                    * (mosquito.OMEGA * mosquito.a * human.PI) * (1 / mosquito.MU)) 
                    * ((human.EPSILON / (human.EPSILON + human.MU)) 
                    * ((mosquito.a * mosquito.PI * mosquito.TotalPop) / human.TotalPop) 
                    * (1 / (human.MU + human.DELTA))))
        )
        
        calc_prevalence(human.E, human.I, human)
        calc_prevalence(mosquito.E, mosquito.I, mosquito)
        simulations["PREV"]["Human"].append(human.PREV)
        simulations["PREV"]["Mosquito"].append(mosquito.PREV)

    print(f"R0: {simulations['R0']}")
        
    for i in range(step+1):
        plt.plot(simulations["y"][i].t, simulations["PREV"]["Human"][i], label=f'Human Prev {i}')
    plt.title(f"Prevalence of Infection over variable {param}")
    plt.xlabel('Time /weeks')
    plt.ylabel('Fraction of population')
    plt.ylim(0, 1.05)
    plt.grid()
    plt.legend()
    plt.show()
    

        
def main():
    ''' Set the parameters, solve the ODE, and plot the results '''
    human = Agent.from_kwargs(**PARAMETERS["Human"])
    mosquito = Agent.from_kwargs(**PARAMETERS["Mosquito"])

    # Initial conditions
    y0 = [human.S_0, human.E_0, human.I_0, human.R_0, mosquito.S_0, mosquito.E_0, mosquito.I_0]

    # Solve the ODE
    y = solve_ivp(f, (START, STOP), y0, args=(human, mosquito))

    # Define and solve for the basic reproductive ratio, R0
    R0 = math.sqrt(((mosquito.EPSILON / (mosquito.EPSILON + mosquito.MU)) 
                    * (mosquito.OMEGA * mosquito.a * human.PI) * (1 / mosquito.MU)) 
                    * ((human.EPSILON / (human.EPSILON + human.MU)) 
                    * ((mosquito.a * mosquito.PI * mosquito.TotalPop) / human.TotalPop) 
                    * (1 / (human.MU + human.DELTA))))
    
    print(f"R0: {R0}")

    # Set the human and mosquito parameters with the ODE results
    t = y.t
    human.S = y.y[0, :]
    human.E = y.y[1, :]
    human.I = y.y[2, :]
    human.R = y.y[3, :]
    mosquito.S = y.y[4, :]
    mosquito.E = y.y[5, :]
    mosquito.I = y.y[6, :]

    # Calculate the prevalence based on the ODE solution times
    calc_prevalence(human.E, human.I, human)
    calc_prevalence(mosquito.E, mosquito.I, mosquito)

    # Plot the results
    makeplots(t, human.S, human.E, human.I, human.R)
    makeplots(t, mosquito.S, mosquito.E, mosquito.I)

    plt.plot(t, human.PREV, label='Human Prev')
    plt.plot(t, mosquito.PREV, label='Mosquito Prev')
    plt.title("Prevalence of Infection")
    plt.grid()
    plt.legend()
    plt.show()
    

if __name__ == "__main__":
    if DO_SIMULATION:
        human = Agent.from_kwargs(**PARAMETERS["Human"])
        mosquito = Agent.from_kwargs(**PARAMETERS["Mosquito"])
        simulate(mosquito, PARAM_TO_SIMULATE, 
                 SIM_START, SIM_END, SIM_INCREMENTS, 
                 *(human, mosquito))
    else:
        main()

