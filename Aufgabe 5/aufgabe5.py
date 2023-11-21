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

    global functions
    func = functions["dpfrommdot"]
    dp_verdichter = func.func(m_dot, *func.get_params())

    return dp_rohr + dp_verdichter


if __name__ == "__main__":

    # print(rohr_dp(0.05, 0.05, 200))

    global functions
    functions = ff.fit_curves()

    P_el_from_dp_func = functions["pelfromdp"]

    L = np.linspace(0, 200, 100)
    a_solution = []
    b_solution = []

    for D in [5e-2, 10e-2, 30e-2, 40e-2, 50e-2]:
        m_dot = []
        m_dot_guess = 1
        P_el = []
        for L_element in L:
            # print(f"D={D} und L={L_element}")
            m_dot_solve = root(fun=druck_bilanz, x0=m_dot_guess, args=(D, L_element))

            if m_dot_solve.success:
                solution = m_dot_solve.x[0]
                m_dot.append(solution)
                m_dot_guess = solution

                dp = -rohr_dp(solution, D, L_element)

                P_el_solution = P_el_from_dp_func.func(dp, *P_el_from_dp_func.get_params())
                P_el.append(P_el_solution)
            else:
                m_dot.append(np.inf)
                P_el.append(np.inf)
                print(f"Keine Lösung gefunden für D={D} und L={L}")

        a_solution.append([D, m_dot])
        b_solution.append([D, P_el])
    # Aufgabe 1a)

    plt.figure(1)
    for solution in a_solution:
        D, m_dot = solution
        plt.plot(L, m_dot, label=f"Durchmesser: {D * 1e2} cm")

    plt.xlabel(r"Länge / [m]")
    plt.ylabel(r"Massenstrom / [kg/s]")
    plt.title("Aufgabe 1a) Erreichbarer Massenstrom")
    plt.legend()
    plt.grid()
    plt.show()

    # Aufgabe 1b)
    plt.figure(2)

    for solution in b_solution:
        D, P_el = solution
        plt.plot(L, P_el, label=f"Durchmesser: {D * 1e2} cm")

    plt.xlabel(r"Länge / [m]")
    plt.ylabel(r"Elektrische Leistung / [W]")
    plt.title("Aufgabe 1b) Erforderliche Leistung")
    plt.legend()
    plt.grid()
    plt.show()
