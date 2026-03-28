import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- PAGE CONFIG ---
st.set_page_config(page_title="Neuron Action Potential Simulator", layout="wide")
st.title("Hodgkin-Huxley Neuron Simulation")

# --- SIDEBAR CONTROLS (Restoring your original boxes) ---
st.sidebar.header("Simulation Parameters")
v_start = st.sidebar.number_input("Initial V (mV)", value=-53.0)
I1 = st.sidebar.number_input("Pulse 1 Amp (uA)", value=0.0)
p1 = st.sidebar.number_input("Pulse 1 Width (ms)", value=0.0)
delay = st.sidebar.number_input("Delay (ms)", value=0.0)
I2 = st.sidebar.number_input("Pulse 2 Amp (uA)", value=0.0)
p2 = st.sidebar.number_input("Pulse 2 Width (ms)", value=0.0)
t_max = st.sidebar.number_input("Time Span (ms)", value=20.0)

# --- THE MATH (Exact copy from your original) ---
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
    
    # Stimulus current logic
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

# --- RUN SIMULATION ---
an0, bn0, am0, bm0, ah0, bh0 = rate_constants(-60.0)
n_ss, m_ss, h_ss = an0/(an0+bn0), am0/(am0+bm0), ah0/(ah0+bh0)
y0 = [v_start, n_ss, m_ss, h_ss]

sol = solve_ivp(
    equations, [0, t_max], y0, 
    args=(p1, p2, delay, I1, I2),
    method='BDF', t_eval=np.linspace(0, t_max, 1000)
)

# --- PLOTTING (The 6-Panel Layout) ---
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
plt.subplots_adjust(hspace=0.4, wspace=0.3)

# Plot 1: Membrane Potential
axs[0,0].plot(sol.t, sol.y[0], 'k', lw=2)
axs[0,0].set_title("Membrane Potential (mV)")
axs[0,0].grid(True)

# Plot 2: Gating Variables
axs[0,1].plot(sol.t, sol.y[1], label='n(t)')
axs[0,1].plot(sol.t, sol.y[2], label='m(t)')
axs[0,1].plot(sol.t, sol.y[3], label='h(t)')
axs[0,1].set_title("Gating Variables")
axs[0,1].legend(loc='upper right', fontsize='small')

# Plot 3: Conductances
gK = 36 * (sol.y[1]**4)
gNa = 120 * (sol.y[2]**3) * sol.y[3]
axs[0,2].plot(sol.t, gK, label='gK')
axs[0,2].plot(sol.t, gNa, label='gNa')
axs[0,2].set_title("Conductances")
axs[0,2].legend(loc='upper right', fontsize='small')

# Steady State Data
v_range = np.linspace(-100, 100, 400)
an, bn, am, bm, ah, bh = rate_constants(v_range)

# Plot 4: Time Constants
axs[1,0].plot(v_range, 1/(an+bn), label='tn')
axs[1,0].plot(v_range, 1/(am+bm), label='tm')
axs[1,0].plot(v_range, 1/(ah+bh), label='th')
axs[1,0].set_title("Time Constants (ms)")
axs[1,0].set_ylim(0, 10)
axs[1,0].legend(fontsize='small')

# Plot 5: Steady State Gating
axs[1,1].plot(v_range, an/(an+bn), label='n_inf')
axs[1,1].plot(v_range, am/(am+bm), label='m_inf')
axs[1,1].plot(v_range, ah/(ah+bh), label='h_inf')
axs[1,1].set_title("Steady State Gating")
axs[1,1].legend(fontsize='small')

# Plot 6: Phase Plot
axs[1,2].plot(sol.y[0], sol.y[1], 'r')
axs[1,2].set_title("Phase Plane (V vs n)")
axs[1,2].set_xlabel("V (mV)")

st.pyplot(fig)
