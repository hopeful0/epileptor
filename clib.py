from ctypes import *
import numpy as np
import time
import random
from utils import array2cp, loadConnection, loadDistance

epileptor = cdll.LoadLibrary('./epileptor.so')

class Parameter_Epileptor_5(Structure):
    _fields_ = [('x_0', c_double),
                ('y_0', c_double),
                 ('tau_0', c_double),
                ('tau_1', c_double),
                ('tau_2', c_double),
                ('I_rest1', c_double),
                ('I_reset2', c_double),
                ('gamma', c_double),
                ('D_1', c_double),
                ('D_2', c_double)]

class Epileptor_5(Structure):
    _fields_ = [('x_1', c_double),
                ('y_1', c_double),
                ('z', c_double),
                ('x_2', c_double),
                ('y_2', c_double),
                ('u', c_double),
                ('paras', POINTER(Parameter_Epileptor_5))]

class Epileptor_Network(Structure):
    _fields_ = [('vars', POINTER(POINTER(Epileptor_5))),
                ('K', POINTER(POINTER(c_double))),
                ('n', c_int),
                ('Tau', POINTER(POINTER(c_int)))]

class Epileptor_Result(Structure):
    _fields_ = [('x_1s', POINTER(POINTER(c_double))),
                ('zs', POINTER(POINTER(c_double))),
                ('x_2s', POINTER(POINTER(c_double))),
                ('Izs', POINTER(POINTER(c_double)))]

class Epileptor_Result_Full(Structure):
    _fields_ = [('x_1s', POINTER(POINTER(c_double))),
                ('y_1s', POINTER(POINTER(c_double))),
                ('zs', POINTER(POINTER(c_double))),
                ('x_2s', POINTER(POINTER(c_double))),
                ('y_2s', POINTER(POINTER(c_double))),
                ('Izs', POINTER(POINTER(c_double)))]


epileptor.gen_parameter_epileptor_5.restype = POINTER(Parameter_Epileptor_5)
epileptor.init_epileptor_5.argtypes = (POINTER(Epileptor_5), POINTER(c_uint))
epileptor.init_stable_epileptor_5.argtypes = (POINTER(Epileptor_5),)
epileptor.gen_model_vars.restype = POINTER(POINTER(Epileptor_5))
epileptor.gen_model_vars.argtype = (c_int, POINTER(c_uint))
epileptor.gen_stable_model_vars.restype = POINTER(POINTER(Epileptor_5))
epileptor.gen_stable_model_vars.argtype = (c_int, POINTER(c_uint))
epileptor.simulation.restype = Epileptor_Result
epileptor.simulation_extrenal_input.restype = Epileptor_Result
epileptor.simulation_full.restype = Epileptor_Result_Full
epileptor.simulation_delay.restype = Epileptor_Result
epileptor.gen_model_vars.argtype = (POINTER(Epileptor_Network), c_double, POINTER(c_uint), c_int)
epileptor.free_data.argtype = (Epileptor_Result, Epileptor_Network)
epileptor.extract_seizure_event.argtype = (POINTER(c_double), c_int, POINTER(c_int))
epileptor.free_full_data.argtype = (Epileptor_Result_Full, Epileptor_Network)
epileptor.extract_seizure_event.restype = POINTER(c_int)
epileptor.simulation_sin.restype = Epileptor_Result
epileptor.simulation_sin_x1.restype = Epileptor_Result

# callback function, intput node index and evalution time, then get a double value as return for next evalution.
getDoubleAtT4I = CFUNCTYPE(c_double, c_int, c_double)

def extract_seizure_event(result, n, ic):
    ses_all = np.empty((n), object)
    for i in range(n):
        sec = c_int(0)
        ses = epileptor.extract_seizure_event(result.x_1s[i], ic, byref(sec))
        ses_all[i] = ses[:sec.value]
    return ses_all

def check_is_divergent(result, n, ic):
    for i in range(n):
        if (np.any(np.array(result.x_1s[i][0:ic]) > 10)):
            raise Exception("epileptor is divergent")

def initNetwork(seed, FIP, stable=False):
    conn=loadConnection()

    # network size
    N = len(conn)
    c_conn = array2cp(conn, N, N)

    # init models
    if stable:
        vs = epileptor.gen_stable_model_vars(N, seed)
    else:
        vs = epileptor.gen_model_vars(N, seed)
    # network
    network = Epileptor_Network()
    network.vars = vs
    network.n = N
    network.K = c_conn

    # setting parameters of network
    FIP(network)

    return network

def run(dt, ic, FIP, stable=False):
    # t1=time.time()

    # random seed
    seed = POINTER(c_uint)(c_uint(random.randint(0, 4294967295)))

    network = initNetwork(seed, FIP, stable=stable)

    #simulation
    result = epileptor.simulation(byref(network), c_double(dt), seed, ic)

    check_is_divergent(result, network.n, ic)

    # t2=time.time()
    # print('time cost ', t2-t1, ' s')
    return result, network

def run_full(dt, ic, FIP, stable=False):
    t1=time.time()

    # random seed
    seed = POINTER(c_uint)(c_uint(random.randint(0, 4294967295)))

    network = initNetwork(seed, FIP, stable=stable)

    #simulation
    result = epileptor.simulation_full(byref(network), c_double(dt), seed, ic)

    check_is_divergent(result, network.n, ic)

    t2=time.time()
    print('time cost ', t2-t1, ' s')
    return result, network


def initNetwork_delay(seed, FIP, distance, stable=False):
    conn=loadConnection()

    # network size
    N = len(conn)
    c_conn = array2cp(conn, N, N)

    # init models
    if stable:
        vs = epileptor.gen_stable_model_vars(N, seed)
    else:
        vs = epileptor.gen_model_vars(N, seed)
    # network
    network = Epileptor_Network()
    network.vars = vs
    network.n = N
    network.K = c_conn
    network.Tau = array2cp(distance, N, N, dtype=c_int)

    # setting parameters of network
    FIP(network)

    return network

def run_delay(dt, ic, FIP, stable=False):
    t1=time.time()

    # random seed
    seed = POINTER(c_uint)(c_uint(random.randint(0, 4294967295)))

    distance = loadDistance()
    network = initNetwork_delay(seed, FIP, distance, stable=stable)

    #simulation
    result = epileptor.simulation_delay(byref(network), c_int(np.max(distance)), c_double(dt), seed, ic)

    check_is_divergent(result, network.n, ic)

    t2=time.time()
    print('time cost ', t2-t1, ' s')
    return result, network