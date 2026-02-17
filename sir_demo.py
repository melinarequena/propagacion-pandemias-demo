import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.integrate import solve_ivp


st.title("Demo interactiva del modelo SIR")

presets = {
    "Custom": None,
    "Influenza (R0≈1.28, Trec≈5d)": {"beta": 0.256, "gamma": 0.200, "days": 150},
    "COVID-19 (R0≈2.5, Trec≈7d)": {"beta": 0.357, "gamma": 0.143, "days": 150},
    "Sarampión (R0≈15, Trec≈8d)": {"beta": 1.875, "gamma": 0.125, "days": 150},
}

preset_name = st.selectbox("Epidemias", list(presets.keys()), index=0)
preset = presets[preset_name]

N = st.number_input("Población total N", min_value=100, value=10000, step=100)

default_I0 = 1 if preset_name.startswith("Sarampión") else 10  # opcional pero recomendable
I0 = st.number_input("Infectados iniciales", min_value=0, value=default_I0, step=1)

R_init = st.number_input("Removidos iniciales (R(0))", min_value=0, value=0, step=1)

S0 = N - I0 - R_init
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
st.subheader("Tests automáticos ✅/❌")

EPS_MASS = 1e-2      # tolerancia en conservación (personas)
EPS_NEG  = -1e-6     # tolerancia para no negatividad (por redondeo)
EPS_I0   = 1e-6

def run_sir(N, S0, I0, R0, beta, gamma, days, npoints=2000):
    def sir_ode(t, y):
        S, I, R = y
        dS = -beta * S * I / N
        dI = beta * S * I / N - gamma * I
        dR = gamma * I
        return [dS, dI, dR]

    t = np.linspace(0, days, npoints)
    sol = solve_ivp(
        lambda tt, yy: sir_ode(tt, yy),
        (0, days),
        [S0, I0, R0],
        t_eval=t,
        rtol=1e-8,
        atol=1e-10
    )
    S, I, R = sol.y
    return t, S, I, R

def check_common(N, S, I, R):
    mass_error = float(np.max(np.abs((S + I + R) - N)))
    min_val = float(min(np.min(S), np.min(I), np.min(R)))
    return mass_error, min_val

def ok(cond):  # helper visual
    return "✅" if cond else "❌"









tests = []

# Test A: conservación y no negatividad (con tus parámetros actuales)
mass_error, min_val = check_common(N, S, I, R)
tests.append(("Sanidad (config actual): conservación", mass_error <= EPS_MASS, f"error={mass_error:.6f}"))
tests.append(("Sanidad (config actual): no-negatividad", min_val >= EPS_NEG, f"min={min_val:.6f}"))

# Test 1: I0=0 ⇒ I(t)=0 siempre
N1 = 10000
I0_1, R0_1 = 0, 0
S0_1 = N1 - I0_1 - R0_1
beta1, gamma1, days1 = 0.25, 1/7, 150
t1, S1, I1, R1 = run_sir(N1, S0_1, I0_1, R0_1, beta1, gamma1, days1)
tests.append(("Caso límite: I0=0 ⇒ I(t)=0", float(np.max(I1)) <= EPS_I0, f"maxI={float(np.max(I1)):.6f}"))

# Test 2: β=0 ⇒ no nuevos contagios, S constante
N2 = 10000
I0_2, R0_2 = 10, 0
S0_2 = N2 - I0_2 - R0_2
beta2, gamma2, days2 = 0.0, 1/7, 150
t2, S2, I2, R2 = run_sir(N2, S0_2, I0_2, R0_2, beta2, gamma2, days2)
tests.append(("Caso límite: β=0 ⇒ S constante", float(np.max(np.abs(S2 - S2[0]))) <= 1e-3, f"ΔSmax={float(np.max(np.abs(S2 - S2[0]))):.6f}"))

# Test 3: R0<1 ⇒ I decrece desde el inicio (sin crecimiento inicial)
N3 = 10000
I0_3, R0_3 = 10, 0
S0_3 = N3 - I0_3 - R0_3
beta3, gamma3, days3 = 0.08, 1/7, 150  # R0≈0.56
t3, S3, I3, R3 = run_sir(N3, S0_3, I0_3, R0_3, beta3, gamma3, days3)
tests.append(("Epidemia subcrítica: R0<1 ⇒ I no crece al inicio", I3[1] <= I3[0] + 1e-6, f"I0={I3[0]:.3f}, I(dt)={I3[1]:.3f}"))

# Test 4: R0>1 ⇒ I crece al inicio
N4 = 10000
I0_4, R0_4 = 10, 0
S0_4 = N4 - I0_4 - R0_4
beta4, gamma4, days4 = 0.25, 1/7, 150  # R0≈1.75
t4, S4, I4, R4 = run_sir(N4, S0_4, I0_4, R0_4, beta4, gamma4, days4)
tests.append(("Epidemia supercrítica: R0>1 ⇒ I crece al inicio", I4[1] > I4[0], f"I0={I4[0]:.3f}, I(dt)={I4[1]:.3f}"))

# Test 5: Intervención bajar β ⇒ pico menor
# A: sin medidas
betaA, gammaA = 0.25, 1/7
tA, SA, IA, RA = run_sir(N4, S0_4, I0_4, 0, betaA, gammaA, 150)
# B: con distanciamiento
betaB, gammaB = 0.12, 1/7
tB, SB, IB, RB = run_sir(N4, S0_4, I0_4, 0, betaB, gammaB, 150)
tests.append(("Intervención: bajar β ⇒ pico I menor", float(np.max(IB)) < float(np.max(IA)), f"picoA={float(np.max(IA)):.1f}, picoB={float(np.max(IB)):.1f}"))

# Test 6: Intervención subir γ ⇒ pico menor (misma β)
betaC = 0.25
gammaC1, gammaC2 = 0.10, 0.20
tC1, SC1, IC1, RC1 = run_sir(N4, S0_4, I0_4, 0, betaC, gammaC1, 150)
tC2, SC2, IC2, RC2 = run_sir(N4, S0_4, I0_4, 0, betaC, gammaC2, 150)
tests.append(("Intervención: subir γ ⇒ pico I menor", float(np.max(IC2)) < float(np.max(IC1)), f"picoγ0.10={float(np.max(IC1)):.1f}, picoγ0.20={float(np.max(IC2)):.1f}"))

# Mostrar resultados
for name, passed, info in tests:
    st.write(f"{ok(passed)} {name} — {info}")
