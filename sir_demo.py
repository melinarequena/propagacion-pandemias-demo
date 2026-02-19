import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.integrate import solve_ivp


st.title("Simulador de graficas SIR")
st.header("Metodos y Modelos - Examen Final")
st.subheader("Melina Requena")
st.divider()

presets = {
    "Custom": None,
    "Influenza (R0≈2.1, Trec≈10d)": {"beta": 0.210, "gamma": 0.100, "days": 150},
    "COVID-19 (R0≈5.7, Trec≈15d)": {"beta": 0.380, "gamma": 0.067, "days": 150},
    "Sarampión (R0≈15, Trec≈12d)": {"beta": 1.245, "gamma": 0.083, "days": 150},
}

preset_name = st.selectbox("Epidemias", list(presets.keys()), index=0)
preset = presets[preset_name]

N = st.number_input("Población total N", min_value=100, value=10000, step=100)
I0 = st.number_input("Infectados iniciales", min_value=0, value=1, step=1)
R_init = st.number_input("Recuperados iniciales", min_value=0, value=0, step=1)

S0 = N - I0 - R_init # Susceptibles iniciales: N - Inf(0) - Rec(0)
if S0 < 0:
    st.error("N - I0 - R(0) debe ser ≥ 0.")
    st.stop()

beta_default = preset["beta"] if preset else 0.25
gamma_default = preset["gamma"] if preset else 0.14
days_default = preset["days"] if preset else 150

beta = st.slider("β (tasa de transmisión)", 0.0, 3.0, float(beta_default), 0.001)
gamma = st.slider("γ (tasa de recuperación)", 0.01, 1.0, float(gamma_default), 0.001)
days = st.slider("Días a simular", 30, 365, int(days_default), 5)

def sir_ode(t, y, beta, gamma, N):
    S, I, R = y
    dS = -beta * S * I / N
    dI = beta * S * I / N - gamma * I
    dR = gamma * I
    return [dS, dI, dR]

t = np.linspace(0, days, 2000)

sol = solve_ivp(
    sir_ode,
    (0, days),
    [S0, I0, R_init],
    t_eval=t,
    args=(beta, gamma, N),
    rtol=1e-8,
    atol=1e-10
)

S, I, R = sol.y

st.metric("R₀ = β/γ", f"{(beta/gamma):.2f}")

mass_error = np.max(np.abs((S + I + R) - N))
st.caption(f"Error máximo en S+I+R-N: {mass_error:.4f}")

fig = plt.figure()
plt.plot(t, S, label="S")
plt.plot(t, I, label="I")
plt.plot(t, R, label="R")
plt.xlabel("Días")
plt.ylabel("Personas")
plt.legend()
st.pyplot(fig)

peak_day = float(t[np.argmax(I)])
peak_value = float(np.max(I))
st.write(f"Pico de infectados: **{peak_value:.0f}** personas en el día **{peak_day:.1f}**.")

st.divider()
