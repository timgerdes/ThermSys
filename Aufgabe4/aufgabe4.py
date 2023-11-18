import numpy as np
import matplotlib.pyplot as plt
import fitting_functions as ff

figure_i = 1


def plot(func: ff.FitFuntion, x_label, y_label, x_max=None, plot=False):
    orig_data = func.get_data()

    if x_max is None:
        x_max = func.get_x_max()

    x = np.linspace(0, x_max, 200)
    y = func.func(x, *func.get_params())

    if not plot:
        return x, y

    global figure_i
    plt.figure(figure_i)
    figure_i += 1

    if orig_data is not None:
        plt.scatter(orig_data["x"], orig_data["y"], label="Messdaten")

    plt.plot(x, y, label="Fit")

    plt.title(f"{y_label} über {x_label}")
    plt.xlabel(x_label)
    plt.xlim(left=0)
    plt.ylabel(y_label)
    plt.ylim(bottom=0)
    plt.grid()
    plt.legend()
    plt.show()

    return x, y


if __name__ == "__main__":
    fit = True

    if fit:
        functions = ff.fit_curves()
    else:
        functions = ff.load_curves()

    # Aufgabe 1

    # Plotten: Volumenstrom als Funktion der Druckdifferenz
    plot(functions["vdotfromdp"], r"$\Delta$p [Pa]", r"$\dot{V}$ [$m^{3}$/s]")
    # Plotten: Elektrische Leistung als Funktion der Druckdifferenz
    plot(functions["pelfromdp"], r"$\Delta$p [Pa]", r"$P_{el}$ [W]")
    # Plotten: Temperaturdifferenz als Funktion der Druckdifferenz
    dp, dt = plot(functions["dtfromdp"], r"$\Delta$p [Pa]", r"$\Delta$T [°C]")


    # Plotten: Enthalpieerhöhung als Funktion der Druckdifferenz
    plot(functions["dhfromdp"], r"$\Delta$p [Pa]", r"$\Delta$H [J/kg]")

    # Aufgabe 2
    # Plotten: Druckdifferenz als Funktion des Volumenstroms
    plot(functions["dpfromvdot"], r"$\dot{V}$ [$m^{3}$/s]", r"$\Delta$p [Pa]")

    # Leistung durch Volumenstrom
    # Berechnen: Volumenstrom für vorhandes dp
    v_dot = functions["vdotfromdp"].func(dp, *functions["vdotfromdp"].get_params())  # [m^3/s]
    # Berechnen: Elektrische Leistung für vorhandes dp
    p_el = functions["pelfromdp"].func(dp, *functions["pelfromdp"].get_params())

    func = ff.LinearFunction(v_dot, p_el)
    plot(func, r"$\dot{V}$ [$m^{3}$/s]", r"$P_{el}$ [W]", plot=True)

    # Berechnen: m_dot als Funktion des Druckes
    m_dot = functions["mdotfromdp"].func(dp, *functions["mdotfromdp"].get_params())

    # Druckdifferenz durch Massenstrom
    plot(functions["dpfrommdot"], r"$\dot{m}$ [kg/s]", r"$\Delta$p [Pa]")

    # Leistung durch Massenstrom
    func = ff.LinearFunction(m_dot, p_el)
    plot(func, r"$\dot{m}$ [kg/s]", r"$P_{el}$ [W]")

    # Aufgabe 3
    p_0 = 1.013e5  # Druck [Pa]
    t_0 = 293.15  # Temperatur [K]
    r_L = 287  # Gaskonstante für Luft [J/kg*K]
    c_p = 1007

    # isentrope temperatur aus isentropengleichung
    kappa_l = 1.4  # Isentropenexponent [-]
    p1 = dp + p_0
    t_is = ((p_0 / (p1)) ** ((1 - kappa_l) / kappa_l)) * t_0
    # zum Ermitteln der Isentropen Leistung gilt: P_is = m_dot * c_p (t_is - t_0)
    p_is = m_dot * c_p * (t_is - t_0)
    # Reale Leistung aus Temperaturerhöhung: P_real = m_dot * c_p * dt
    p_real = m_dot * c_p * dt

    # Wirkungsgrad
    eta_is = p_is / p_real
    eta_mech = p_real / p_el

    plt.figure(figure_i)
    figure_i += 1

    plt.plot(m_dot, p_el, label=r"$P_{is}$")
    plt.plot(m_dot, p_is, label=r"$P_{is}$")
    plt.plot(m_dot, p_real, label=r"$P_{real}$")
    plt.xlabel("Druckdifferenz")
    plt.ylabel("Leistung")
    plt.grid()
    plt.legend()
    plt.legend()
    plt.show()

    plt.figure(figure_i)
    figure_i += 1
    plt.plot(m_dot, eta_is * 100, label=r"$\eta_{is}$")
    plt.plot(m_dot, eta_mech * 100, label=r"$\eta_{mech}$")
    plt.xlabel("Druckdifferenz [Pa]")
    plt.ylabel("Wirkungsgrad [%]")
    plt.grid()
    plt.legend()
    plt.legend()
    plt.show()