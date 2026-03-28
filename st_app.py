import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- PAGE SETUP ---
st.set_page_config(page_title="HH Neuron Simulator", layout="wide")
st.title("Hodgkin-Huxley Action Potential Simulation")

# --- RESET LOGIC ---
if st.sidebar.button("Reset Simulation"):
    st.rerun()

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Simulation Parameters")

# Pulse inputs with limits -79 to 80
I1 = st.sidebar.number_input("Pulse 1 Amp (uA)", min_value=-79.0, max_value=80.0, value=0.0)
p1 = st.sidebar.number_input("Pulse 1 Width (ms)", min_value=0.0, max_value=80.0, value=0.0)
delay = st.sidebar.number_input("Delay (ms)", min_value=0.0, max_value=80.0, value=0.0)
I2 = st.sidebar.number_input("Pulse 2 Amp (uA)", min_value=-79.0, max_value=80.0, value=0.0)
p2 = st.sidebar.number_input("Pulse 2 Width (ms)", min_value=0.0, max_value=80.0, value=0.0)
t_max = st.sidebar.number_input("Time Span (ms)", min_value=1.0, max_value=500.0, value=20.0)

# SMART VOLTAGE LOGIC
# If any pulse/delay value is changed (not 0), default becomes -60. Otherwise -53.
if I1 != 0 or p1 != 0 or delay != 0 or I2 != 0 or p2 != 0:
    v_default = -60.0
else:
    v_default = -53.0

v_start = st.sidebar.number_input("Initial V (mV)", min_value=-79.0, max_value=80.0, value=v_default)

# --- THE MATH ---
def rate_constants(v):
    am = 0.1 * (-35 - v) / (np.exp((-35 - v) / 10) - 1)
    bm = 4 * np.exp((-60 - v) / 18)
    ah = 0.07 * np.exp((-60 - v) / 20)
    bh = 1.0 / (np.exp((-30 - v) / 10) + 1)
    an = 0.01 * (-50 - v) / (np.exp((-50.0000001 - v) / 10) - 1)
    bn = 0.125 * np.exp((-60 - v) / 80)
    return an, bn, am, bm, ah, bh

def equations(t, y, p1, p2, delay, I1, I2):
    v, n, m, h = y
    ggK, ggNa, ggL = 36.0, 120.0, 0.3
    vK, vNa, vL = -72.14, 55.17, -49.24
    Cm = 1.0
    
    if t < p1:
        Is = I1
    elif t < (p1 + delay + p2):
        Is = I2
    else:
        Is = 0.0
            
    an, bn, am, bm, ah, bh = rate_constants(v)
    dv = (-ggK*(n**4)*(v-vK) - ggNa*(m**3)*h*(v-vNa) - ggL*(v-vL) + Is) / Cm
    dn = an * (1 - n) - bn * n
    dm = am * (1 - m) - bm * m
    dh = ah * (1 - h) - bh * h
    return [dv, dn, dm, dh]

# Solve
an0, bn0, am0, bm0, ah0, bh0 = rate_constants(-60.0)
y0 = [v_start, an0/(an0+bn0), am0/(am0+bm0), ah0/(ah0+bh0)]

sol = solve_ivp(equations, [0, t_max], y0, args=(p1, p2, delay, I1, I2),
                method='BDF', t_eval=np.linspace(0, t_max, 1000))

# --- PLOTTING ---
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
plt.subplots_adjust(hspace=0.4, wspace=0.3)

# 1. Membrane Potential (SCALING FIXED TO -80 to 80)
axs[0,0].plot(sol.t, sol.y[0], 'k', lw=2)
axs[0,0].set_title("Membrane Potential (mV)")
axs[0,0].set_ylim(-80, 80)
axs[0,0].grid(True)

# 2. Gating Variables
axs[0,1].plot(sol.t, sol.y[1], label='n(t)')
axs[0,1].plot(sol.t, sol.y[2], label='m(t)')
axs[0,1].plot(sol.t, sol.y[3], label='h(t)')
axs[0,1].set_title("Gating Variables")
axs[0,1].set_ylim(0, 1)
axs[0,1].legend()

# 3. Conductances
gK = 36 * (sol.y[1]**4)
gNa = 120 * (sol.y[2]**3) * sol.y[3]
axs[0,2].plot(sol.t, gK, label='gK')
axs[0,2].plot(sol.t, gNa, label='gNa')
axs[0,2].set_title("Conductances")
axs[0,2].legend()

# 4. Time Constants
v_range = np.linspace(-100, 100, 400)
an, bn, am, bm, ah, bh = rate_constants(v_range)
axs[1,0].plot(v_range, 1/(an+bn), label='tn')
axs[1,0].plot(v_range, 1/(am+bm), label='tm')
axs[1,0].plot(v_range, 1/(ah+bh), label='th')
axs[1,0].set_title("Time Constants (ms)")
axs[1,0].set_ylim(0, 10)
axs[1,0].legend()

# 5. Steady State Gating
axs[1,1].plot(v_range, an/(an+bn), label='n_inf')
axs[1,1].plot(v_range, am/(am+bm), label='m_inf')
axs[1,1].plot(v_range, ah/(ah+bh), label='h_inf')
axs[1,1].set_title("Steady State Gating")
axs[1,1].legend()

# 6. Phase Plot
axs[1,2].plot(sol.y[0], sol.y[1], 'r')
axs[1,2].set_title("Phase Plane (V vs n)")
axs[1,2].set_xlabel("V (mV)")
axs[1,2].set_xlim(-80, 80)

st.pyplot(fig)
