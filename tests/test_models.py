import numpy as np
from modules.kinetics import analyze_kinetics
from modules.parameters import calculate_characteristic_times
from modules.arrhenius import analyze_arrhenius

def test_zero_order():
    t = np.array([0, 1, 2, 3, 4], float)
    c = 100 - 2*t
    r = analyze_kinetics(t,c)
    assert abs(r["Orden cero"]["k"]-2) < 1e-9
    assert abs(r["Orden cero"]["r2"]-1) < 1e-9

def test_first_order():
    t = np.array([0, 1, 2, 3, 4], float)
    c = 100*np.exp(-0.1*t)
    r = analyze_kinetics(t,c)
    assert abs(r["Orden uno"]["k"]-0.1) < 1e-9
    assert abs(r["Orden uno"]["r2"]-1) < 1e-9

def test_second_order():
    t = np.array([0, 1, 2, 3, 4], float)
    c = 100/(1+0.5*t)
    r = analyze_kinetics(t,c)
    assert abs(r["Orden dos"]["k"]-0.005) < 1e-9
    assert abs(r["Orden dos"]["r2"]-1) < 1e-9

def test_half_life_first_order():
    vals = calculate_characteristic_times(1, 0.1, 100)
    assert abs(vals["t50"]-np.log(2)/0.1) < 1e-9

def test_arrhenius_recovery():
    temps = np.array([30,40,50,60], float)
    ea = 75000.0
    A = 1e10
    R = 8.314462618
    tk = temps + 273.15
    kvals = A*np.exp(-ea/(R*tk))
    result = analyze_arrhenius(temps,kvals)
    assert abs(result["Ea_J_mol"]-ea) < 1e-5
    assert abs(result["A"]/A-1) < 1e-8
