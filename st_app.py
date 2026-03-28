import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- PAGE SETUP ---
st.set_page_config(page_title="HH Neuron Simulator", layout="wide")
st.title("Hodgkin-Huxley Action Potential Simulation")

# --- COLORS ---
C_RED = 'darkred'
C_GOLD = 'darkgoldenrod'
C_BLUE = 'darkblue'

# --- 1. SESSION STATE INITIALIZATION ---
if 'v_start' not in st.session_state: st.session_state.v_start = -53.0
if 'I1' not in st.session_state: st.session_state.I1 = 0.0
if 'p1' not in st.session_state: st.session_state.p1 = 0.0
if 'delay' not in st.session_state: st.session_state.delay = 0.0
if 'I2' not in st.session_state: st.session_state.I2 = 0.0
if 'p2' not in st.session_state: st.session_state.p2 = 0.0
if 't_max' not in st.session_state: st.session_state.t_max = 20.0
if 'needs_run' not in st.session_state: st.session_state.needs_run = False

# --- 2. SMART VOLTAGE LOGIC ---
pulses_active = any([st.session_state.I1 != 0, st.session_state.p1 != 0, 
                     st.session_state.delay != 0, st.session_state.I2 != 0, 
                     st.session_state.p2 != 0])

if pulses_active and st.session_state.v_start == -53.0:
    st.session_state.v_start = -60.0
elif not pulses_active and st.session_state.v_start == -60.0:
    st.session_state.v_start = -53.0

# --- 3. SIDEBAR INPUTS ---
st.sidebar.header("Simulation Parameters")
v_start = st.sidebar.number_input("Initial V (mV)", min_value=-79.0, max_value=80.0, key="v_start")
I1 = st.sidebar.number_input("Pulse 1 Amp (uA)", min_value=-79.0, max_value=80.0, key="I1")
p1 = st.sidebar.number_input("Pulse 1 Width (ms)", min_value=0.0, max_value=80.0, key="p1")
delay = st.sidebar.number_input("Delay (ms)", min_value=0.0, max_value=80.0, key="delay")
I2 = st.sidebar.number_input("Pulse 2 Amp (uA)", min_value=-79.0, max_value=80.0, key="I2")
p2 = st.sidebar.number_input("Pulse 2 Width (ms)", min_value=0.0, max_value=80.0, key="p2")
t_max = st.sidebar.number_input("Time Span (ms)", min_value=1.0, max_value=500.0, key="t_max")

# --- 4. BUTTONS ---
st.sidebar.markdown("---")
if st.sidebar.button("RUN", type="primary", use_container_width=True):
    st.session_state.needs_run = True

if st.sidebar.button("RESET", use_container_width=True):
    st.session_state.needs_run = False

if st.sidebar.button("DEFAULT VALUES", use_container_width=True):
    st.session_state.v_start = -53.0
    st.session_state.I1 = 0.0; st.session_state.p1 = 0.0; st.session_state.delay = 0.0
    st.session_state.I2 = 0.0; st.session_state.p2 = 0.0; st.session_state.t_max = 20.0
    st.session_state.needs_run = False
    st.rerun()

# --- 5. MATH ---
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
    Is = I1 if t < p1 else (I2 if t < (p1 + delay + p2) else 0.0)
    an, bn, am, bm, ah, bh = rate_constants(v)
    dv = (-ggK*(n**4)*(v-vK) - ggNa*(m**3)*h*(v-vNa) - ggL*(v-vL) + Is) / Cm
    dn = an * (1 - n) - bn * n
    dm = am * (1 - m) - bm * m
    dh = ah * (1 - h) - bh * h
    return [dv, dn, dm, dh]

# --- 6. PLOTTING (CRISP RENDER) ---
fig, axs = plt.subplots(2, 3, figsize=(14, 9))
plt.subplots_adjust(hspace=0.4, wspace=0.3)

# ALWAYS set up axes first so they are visible but empty
for ax in axs.flat: ax.grid(True, linestyle=':', alpha=0.6)
axs[0,0].set_ylim(-80, 80); axs[0,0].set_title("Membrane Potential (mV)")
axs[0,1].set_ylim(0, 1.1); axs[0,1].set_title("Gating Variables")
axs[0,2].set_ylim(0, 40); axs[0,2].set_title("Conductances")
axs[1,0].set_ylim(0, 10); axs[1,0].set_title("Time Constants (ms)")
axs[1,1].set_ylim(0, 1.1); axs[1,1].set_title("Steady State Gating")
axs[1,2].set_xlim(-80, 80); axs[1,2].set_ylim(0, 1.1); axs[1,2].set_title("Phase Plane (V vs n)")

# LOGIC GATE: Only plot if RUN was the last button clicked
if st.session_state.needs_run:
    an0, bn0, am0, bm0, ah0, bh0 = rate_constants(-60.0)
    y0 = [st.session_state.v_start, an0/(an0+bn0), am0/(am0+bm0), ah0/(ah0+bh0)]
    sol = solve_ivp(equations, [0, st.session_state.t_max], y0, 
                    args=(st.session_state.p1, st.session_state.p2, st.session_state.delay, st.session_state.I1, st.session_state.I2), 
                    method='BDF', t_eval=np.linspace(0, st.session_state.t_max, 1000))

    axs[0,0].plot(sol.t, sol.y[0], color='black', lw=2, label='V')
    axs[0,0].legend(loc='upper right')

    axs[0,1].plot(sol.t, sol.y[1], color=C_RED, label='n')
    axs[0,1].plot(sol.t, sol.y[2], color=C_GOLD, label='m')
    axs[0,1].plot(sol.t, sol.y[3], color=C_BLUE, label='h')
    axs[0,1].legend(loc='upper right', fontsize='x-small')

    gK = 36 * (sol.y[1]**4); gNa = 120 * (sol.y[2]**3) * sol.y[3]
    axs[0,2].plot(sol.t, gK, color=C_RED, label='gK')
    axs[0,2].plot(sol.t, gNa, color=C_GOLD, label='gNa')
    axs[0,2].legend(loc='upper right')

    v_range = np.linspace(-100, 100, 400); an, bn, am, bm, ah, bh = rate_constants(v_range)
    axs[1,0].plot(v_range, 1/(an+bn), color=C_RED, label='tn'); axs[1,0].plot(v_range, 1/(am+bm), color=C_GOLD, label='tm'); axs[1,0].plot(v_range, 1/(ah+bh), color=C_BLUE, label='th')
    axs[1,0].legend(loc='upper right', fontsize='x-small')

    axs[1,1].plot(v_range, an/(an+bn), color=C_RED, label='n_inf'); axs[1,1].plot(v_range, am/(am+bm), color=C_GOLD, label='m_inf'); axs[1,1].plot(v_range, ah/(ah+bh), color=C_BLUE, label='h_inf')
    axs[1,1].legend(loc='upper right', fontsize='x-small')

    axs[1,2].plot(sol.y[0], sol.y[1], color=C_BLUE, label='V vs n')
    axs[1,2].legend(loc='upper right')

# The final crisp render
st.pyplot(fig, clear_figure=True)
