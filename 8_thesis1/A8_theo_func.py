import numpy as np

def func_C0(d):
    C0 = (d / (d + 4))**(d/2)
    return C0


def func_C1(d):
    C1 = 2 * (d / (d+8))**(d/2) * ((d**4 + 32 * d**3 + 116 * d**2 + 228 * d + 16) / (d * (d + 8)**2))
    return C1


def func_C2(d):
    C2 = 2 * (d / (d+8))**(d/2) * (2 + 8 * d / (d + 8))
    return C2


def func_C3(d):
    C3 = 2 * (d / (d+8))**(d/2) * (1 - ((d * (d + 8)) / (4 * (d + 4)**2))**(d/2))
    return C3


def func_C4(d):
    C4 = (d**2 / ((d + 2) * (d + 6)))**(d/2) * 8 * (d + 4)**2 / ((d + 2) * (d + 6))
    return C4


def func_C5(d):
    C5 = (d**2 / ((d + 2) * (d + 6)))**(d/2) * (1 - (((d + 2) * (d + 6)) / (d + 4)**2)**(d/2))
    return C5


def func_C6(d):
    C2 = func_C2(d)
    C4 = func_C4(d)
    C6 = np.sqrt(C2 + 4 * C4)
    return C6


def func_C7(d):
    C3 = func_C3(d)
    C5 = func_C5(d)
    C7 = np.sqrt(C3 + 4 * C5)
    return C7


def func_U1(u):
    Delta2 = np.dot(u, u)
    C2 = func_C2(2)
    C3 = func_C3(2)
    U1 = C2
    return U1





