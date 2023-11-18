import numpy as np
from Aufgabe4 import fitting_functions as ff
from scipy.optimize import root
import matplotlib.pyplot as plt

eta_Luft = 18.2e-6  # Viskosität [Pa]
rho_Luft = 1.204  # Dichte von Luft [kg/m^3]


def f_param(Re):
    if Re < 2200:  # laminar
        f = 16 / Re
    elif Re >= 2e4:  # turbulent Fall 1
        f = 0.046 * Re ** (-0.2)
    else:  # turbulent Fall 2
        f = 0.079 * Re ** (-0.25)

    return f


f_param = np.vectorize(f_param)


def rohr_dp(m_dot, D, L, eta=eta_Luft, rho=rho_Luft):
    A = D ** 2 / 4 * np.pi  # Fläche [m^2]
    P = D * np.pi  # Umfang [m]
    G = m_dot / A  # Massenfluss [kg/s*m^2]
    D_h = 4 * A / P  # hydrodynamischer Durchmesser [m]
    Re = G * D_h / eta  # Reynoldszahl [-]
    f = f_param(Re)  # Darcy-Reibungswert [-]
    zaehler = -f * 2 * m_dot ** 2 * P * L
    nenner = A ** 3 * rho * 4

    return zaehler / nenner


def druck_bilanz(m_dot, D, L):
    dp_rohr = rohr_dp(m_dot, D, L)

    global func
    dp_verdichter = func.func(m_dot, *func.get_params())

    return dp_rohr + dp_verdichter


if __name__ == "__main__":

    print(rohr_dp(0.05, 0.05, 200))


    functions = ff.fit_curves()

    plt.figure(1)
    global func
    func = functions["dpfrommdot"]
    L = np.linspace(0, 200, 100)
    for D in np.linspace(5e-2, 50e-2, 20):
        m_dot = []
        m_dot_guess = 1
        for L_element in L:
            #print(f"D={D} und L={L_element}")
            m_dot_solve = root(fun=druck_bilanz, x0=m_dot_guess, args=(D, L_element))

            if m_dot_solve.success:
                solution = m_dot_solve.x[0]
                m_dot.append(solution)
                m_dot_guess = solution
            else:
                m_dot.append(np.inf)
                print(f"Keine Lösung gefunden für D={D} und L={L}")

        plt.plot(L, m_dot, label=f"Durchmesser: {D*1e2} cm")

    plt.xlabel(r"Länge / [m]")
    plt.ylabel(r"Massenstrom / [kg/s]")
    #plt.legend()
    plt.grid()
    plt.show()

    plt.figure(2)
    D = [0.05, 0.06, 0.08, 0.1]
    for D_element in D:
        m_dot = np.linspace(0.001, 0.1, 20)
        dp_rohr = rohr_dp(m_dot, D_element, 200)
        plt.plot(m_dot, dp_rohr / 1e3, label=f"Durchmesser: {D_element*1e2} cm")

    plt.xlabel(r"Massentrom / [kg/s]")
    plt.ylabel(r"Druckverlust / [kPa]")
    plt.legend()
    plt.grid()
    plt.show()

