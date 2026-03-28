import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- PART 1: THE BRAIN (The Hodgkin-Huxley Math) ---
# These functions handle the actual neuron science

def alpha_n(v):
    denom = 1 - np.exp(-(v + 55) / 10)
    return 0.01 * (v + 55) / denom if not np.isclose(denom, 0) else 0.1

def beta_n(v):
    return 0.125 * np.exp(-(v + 65) / 80)

def alpha_m(v):
    denom = 1 - np.exp(-(v + 40) / 10)
    return 0.1 * (v + 40) / denom if not np.isclose(denom, 0) else 1.0

def beta_m(v):
    return 4.0 * np.exp(-(v + 65) / 18)

def alpha_h(v):
    return 0.07 * np.exp(-(v + 65) / 20)

def beta_h(v):
    return 1 / (1 + np.exp(-(v + 35) / 10))

def model(t, y, I_inj_func):
    V, m, h, n = y
    C_m = 1.0
    g_Na, g_K, g_L = 120.0, 36.0, 0.3
    E_Na, E_K, E_L = 50.0, -77.0, -54.387
    
    # Calculate currents
    I_Na = g_Na * (m**3) * h * (V - E_Na)
    I_K = g_K * (n**4) * (V - E_K)
    I_L = g_L * (V - E_L)
    
    dVdt = (I_inj_func(t) - I_Na - I_K - I_L) / C_m
    dmdt = alpha_m(V) * (1 - m) - beta_m(V) * m
    dhdt = alpha_h(V) * (1 - h) - beta_h(V) * h
    dndt = alpha_n(V) * (1 - n) - beta_n(V) * n
    return [dVdt, dmdt, dhdt, dndt]

# --- PART 2: THE FACE (Streamlit Web Layout) ---
st.set_page_config(layout="wide", page_title="HH Neuron Simulator")
st.title("Hodgkin-Huxley Neuron Simulator")

# Sidebar inputs to match your original UI
with st.sidebar:
    st.header("Simulation Parameters")
    v_init = st.number_input("Initial V (mV)", value=-53.0)
    t_max = st.number_input("Time Span (ms)", value=20.0)
    i_amp = st.number_input("Stimulus Amp (uA)", value=10.0)
    run_btn = st.button("RUN SIMULATION", type="primary")

if run_btn:
    # Set up the simulation
    t_eval = np.linspace(0, t_max, 1000)
    # Simple pulse: on from 2ms to 7ms
    I_inj = lambda t: i_amp if 2 <= t <= 7 else 0.0
    
    # Initial steady state values
    y0 = [v_init, 0.05, 0.6, 0.32]
    
    sol = solve_ivp(model, [0, t_max], y0, t_eval=t_eval, args=(I_inj,))

    # PART 3: THE PLOTS
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    
    # Voltage Plot
    axs[0, 0].plot(sol.t, sol.y[0], color='black', lw=2)
    axs[0, 0].set_title("Membrane Potential (mV)")
    axs[0, 0].grid(True)

    # Gating Variables Plot
    axs[0, 1].plot(sol.t, sol.y[1], label='m (Na activation)')
    axs[0, 1].plot(sol.t, sol.y[2], label='h (Na inactivation)')
    axs[0, 1].plot(sol.t, sol.y[3], label='n (K activation)')
    axs[0, 1].set_title("Gating Variables")
    axs[0, 1].legend()

    # Phase Plane (V vs n)
    axs[1, 0].plot(sol.y[0], sol.y[3], color='red')
    axs[1, 0].set_xlabel("V (mV)")
    axs[1, 1].set_ylabel("n variable")
    axs[1, 0].set_title("Phase Plane (V vs n)")

    # Stimulus Current
    current_vals = [I_inj(t) for t in sol.t]
    axs[1, 1].plot(sol.t, current_vals, color='blue')
    axs[1, 1].set_title("Injected Current (uA)")

    plt.tight_layout()
    st.pyplot(fig) # This displays the graphs on the website