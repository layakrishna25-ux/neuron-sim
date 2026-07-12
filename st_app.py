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

# --- 1. INITIAL SESSION STATE ---
if 'v_start' not in st.session_state: st.session_state.v_start = -53.0
if 'I1' not in st.session_state: st.session_state.I1 = 0.0
if 'p1' not in st.session_state: st.session_state.p1 = 0.0
if 'delay' not in st.session_state: st.session_state.delay = 0.0
if 'I2' not in st.session_state: st.session_state.I2 = 0.0
if 'p2' not in st.session_state: st.session_state.p2 = 0.0
if 't_max' not in st.session_state: st.session_state.t_max = 20.0
if 'needs_run' not in st.session_state: st.session_state.needs_run = False

if 'old_v' not in st.session_state: st.session_state.old_v = st.session_state.v_start
if 'old_pulses' not in st.session_state:
    st.session_state.old_pulses = [st.session_state.I1, st.session_state.p1, st.session_state.delay, st.session_state.I2, st.session_state.p2]

# --- 2. SIDEBAR PARAMETER INTERDEPENDENCIES ---
st.sidebar.header("Simulation Parameters")

# Always read and write directly to/from the session state so inputs never lock up
v_start_input = st.sidebar.number_input("Initial V (mV)", value=st.session_state.v_start, step=1.0)
I1_input = st.sidebar.number_input("Pulse 1 Amp (uA)", min_value=-79.0, max_value=80.0, value=st.session_state.I1, step=1.0)
p1_input = st.sidebar.number_input("Pulse 1 Width (ms)", min_value=0.0, max_value=80.0, value=st.session_state.p1, step=1.0)
delay_input = st.sidebar.number_input("Delay (ms)", min_value=0.0, max_value=80.0, value=st.session_state.delay, step=1.0)
I2_input = st.sidebar.number_input("Pulse 2 Amp (uA)", min_value=-79.0, max_value=80.0, value=st.session_state.I2, step=1.0)
p2_input = st.sidebar.number_input("Pulse 2 Width (ms)", min_value=0.0, max_value=80.0, value=st.session_state.p2, step=1.0)
t_max_input = st.sidebar.number_input("Time Span (ms)", min_value=1.0, max_value=500.0, value=st.session_state.t_max, step=1.0)

# Check what changed relative to our historical baseline tracking
v_changed = (v_start_input != st.session_state.old_v)
current_pulses = [I1_input, p1_input, delay_input, I2_input, p2_input]
pulses_changed = (current_pulses != st.session_state.old_pulses)

# Process cascading business logic rules seamlessly
if v_changed:
    st.session_state.v_start = v_start_input
    st.session_state.I1 = 0.0
    st.session_state.p1 = 0.0
    st.session_state.delay = 0.0
    st.session_state.I2 = 0.0
    st.session_state.p2 = 0.0
    st.session_state.old_v = v_start_input
    st.session_state.old_pulses = [0.0, 0.0, 0.0, 0.0, 0.0]
    st.rerun()
elif pulses_changed:
    st.session_state.v_start = -60.0
    st.session_state.I1 = I1_input
    st.session_state.p1 = p1_input
    st.session_state.delay = delay_input
    st.session_state.I2 = I2_input
    st.session_state.p2 = p2_input
    st.session_state.old_v = -60.0
    st.session_state.old_pulses = current_pulses
    st.rerun()
else:
    st.session_state.v_start = v_start_input
    st.session_state.t_max = t_max_input

# --- 3. BUTTONS ---
st.sidebar.markdown("---")
if st.sidebar.button("RUN", type="primary", use_container_width=True):
    st.session_state.needs_run = True

if st.sidebar.button("RESET", use_container_width=True):
    st.session_state.v_start = -53.0
    st.session_state.I1 = 0.0
    st.session_state.p1 = 0.0
    st.session_state.delay = 0.0
    st.session_state.I2 = 0.0
    st.session_state.p2 = 0.0
    st.session_state.t_max = 20.0
    st.session_state.old_v = -53.0
    st.session_state.old_pulses = [0.0, 0.0, 0.0, 0.0, 0.0]
    st.session_state.needs_run = False
    st.rerun()

# --- 4. MATH ---
def rate_constants(v):
    am = 0.1 * (-35 - v) / (np.exp((-35 - v) / 10) - 1 + 1e-6)
    bm = 4 * np.exp((-60 - v) / 18)
    ah = 0.07 * np.exp((-60 - v) / 20)
    bh = 1.0 / (np.exp((-30 - v) / 10) + 1)
    an = 0.01 * (-50 - v) / (np.exp((-50.0000001 - v) / 10) - 1 + 1e-6)
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

# --- 5. INDEPENDENT VALIDATION CHECK ---
v_in_bounds = (-79.0 <= st.session_state.v_start <= 80.0)

if not v_in_bounds:
    st.error("🚨 **Input Range Error:** Initial Voltage must be restricted strictly between -79.0 mV and 80.0 mV to run a simulation.")
    st.session_state.needs_run = False

# --- 6. PLOTTING ---
fig, axs = plt.subplots(2, 3, figsize=(14, 9))
plt.subplots_adjust(hspace=0.4, wspace=0.3)

for ax in axs.flat: 
    ax.grid(True, linestyle=':', alpha=0.6)

# Force data lines to pull cleanly out of the defined origins
axs[0,0].set_ylim(-80, 80); axs[0,0].set_xlim(left=0); axs[0,0].set_title("Membrane Potential (mV)")
axs[0,1].set_ylim(0, 1.1); axs[0,1].set_xlim(left=0); axs[0,1].set_title("Gating Variables")
axs[0,2].set_ylim(0, 40); axs[0,2].set_xlim(left=0); axs[0,2].set_title("Conductances")

axs[1,0].set_ylim(0, 10); axs[1,0].set_xlim(left=-100, right=100); axs[1,0].set_title("Time Constants (ms)")
axs[1,1].set_ylim(0, 1.1); axs[1,1].set_xlim(left=-100, right=100); axs[1,1].set_title("Steady State Gating")
axs[1,2].set_ylim(0, 1.1); axs[1,2].set_xlim(left=-75, right=80); axs[1,2].set_title("Phase Plane (V vs n)")

if st.session_state.needs_run and v_in_bounds:
    an0, bn0, am0, bm0, ah0, bh0 = rate_constants(-60.0)
    y0 = [st.session_state.v_start, an0/(an0+bn0), am0/(am0+bm0), ah0/(ah0+bh0)]
    
    sol = solve_ivp(equations, [0, st.session_state.t_max], y0, 
                    args=(st.session_state.p1, st.session_state.p2, st.session_state.delay, st.session_state.I1, st.session_state.I2), 
                    method='BDF', t_eval=np.linspace(0, st.session_state.t_max, 1000))

    axs[0,0].set_xlim(0, st.session_state.t_max)
    axs[0,1].set_xlim(0, st.session_state.t_max)
    axs[0,2].set_xlim(0, st.session_state.t_max)

    # 1. Potential
    axs[0,0].plot(sol.t, sol.y[0], color='black', lw=2, label='V')
    axs[0,0].legend(loc='upper right')

    # 2. Gating
    axs[0,1].plot(sol.t, sol.y[1], color=C_RED, label='n')
    axs[0,1].plot(sol.t, sol.y[2], color=C_GOLD, label='m')
    axs[0,1].plot(sol.t, sol.y[3], color=C_BLUE, label='h')
    axs[0,1].legend(loc='upper right', fontsize='x-small')

    # 3. Conductance
    gK = 36 * (sol.y[1]**4); gNa = 120 * (sol.y[2]**3) * sol.y[3]
    axs[0,2].plot(sol.t, gK, color=C_RED, label='gK')
    axs[0,2].plot(sol.t, gNa, color=C_GOLD, label='gNa')
    axs[0,2].legend(loc='upper right')

    # 4. Tau
    v_range = np.linspace(-100, 100, 400); an, bn, am, bm, ah, bh = rate_constants(v_range)
    axs[1,0].plot(v_range, 1/(an+bn), color=C_RED, label='tn'); axs[1,0].plot(v_range, 1/(am+bm), color=C_GOLD, label='tm'); axs[1,0].plot(v_range, 1/(ah+bh), color=C_BLUE, label='th')
    axs[1,0].legend(loc='upper right', fontsize='x-small')

    # 5. Infinity
    axs[1,1].plot(v_range, an/(an+bn), color=C_RED, label='n_inf'); axs[1,1].plot(v_range, am/(am+bm), color=C_GOLD, label='m_inf'); axs[1,1].plot(v_range, ah/(ah+bh), color=C_BLUE, label='h_inf')
    axs[1,1].legend(loc='upper right', fontsize='x-small')

    # 6. Phase
    axs[1,2].plot(sol.y[0], sol.y[1], color=C_BLUE, label='V vs n')
    axs[1,2].legend(loc='upper right')

st.pyplot(fig, clear_figure=True)
