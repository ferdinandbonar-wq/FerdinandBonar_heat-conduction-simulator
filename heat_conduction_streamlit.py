import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="1D Heat Conduction",
    layout="wide"
)

st.title("🔥 1D Transient Heat Conduction")
st.write("With Fixed-Temperature Point Sources")

# =====================================================
# MATERIAL DATABASE
# =====================================================

materials = {
    "Aluminum": (237, 2700, 897),
    "Copper": (401, 8960, 385),
    "Iron": (80.2, 7870, 449),
    "Steel": (50.2, 7850, 470),
    "Brass": (109, 8500, 380),
    "Titanium": (21.9, 4500, 522),
    "Concrete": (1.7, 2400, 880),
    "Glass": (1.05, 2500, 840),
    "Water": (0.60, 1000, 4186),
    "Custom": (None, None, None)
}

# =====================================================
# SIDEBAR INPUTS
# =====================================================

st.sidebar.header("Simulation Inputs")

material = st.sidebar.selectbox(
    "Material",
    list(materials.keys())
)

k, rho, cp = materials[material]

if material == "Custom":

    k = st.sidebar.number_input(
        "k [W/m-K]",
        value=10.0
    )

    rho = st.sidebar.number_input(
        "rho [kg/m³]",
        value=1000.0
    )

    cp = st.sidebar.number_input(
        "cp [J/kg-K]",
        value=1000.0
    )

alpha = k/(rho*cp)

# =====================================================
# DOMAIN
# =====================================================

L = st.sidebar.number_input(
    "Domain Length [m]",
    value=1.0
)

t_final = st.sidebar.number_input(
    "Simulation Time [s]",
    value=100.0
)

# Fixed numerical parameters
dx = 0.1
Fo = 0.1

# =====================================================
# INITIAL CONDITION
# =====================================================

IC_type = st.sidebar.selectbox(
    "Initial Condition",
    [
        "Uniform Temperature",
        "Linear Gradient",
        "Gaussian Hot Spot"
    ]
)

if IC_type == "Uniform Temperature":

    T0 = st.sidebar.number_input(
        "Initial Temperature [K]",
        value=300.0
    )

elif IC_type == "Linear Gradient":

    TL = st.sidebar.number_input(
        "Left Temperature [K]",
        value=300.0
    )

    TR = st.sidebar.number_input(
        "Right Temperature [K]",
        value=500.0
    )

else:

    T_base = st.sidebar.number_input(
        "Base Temperature [K]",
        value=300.0
    )

    T_peak = st.sidebar.number_input(
        "Peak Increase [K]",
        value=500.0
    )

    x_center = st.sidebar.number_input(
        "Center Position [m]",
        value=0.5
    )

    sigma = st.sidebar.number_input(
        "Sigma [m]",
        value=0.1
    )

# =====================================================
# BOUNDARY CONDITION
# =====================================================

BC_type = st.sidebar.radio(
    "Boundary Condition",
    [
        "Insulated",
        "Fixed Temperature"
    ]
)

if BC_type == "Fixed Temperature":

    BC_left = st.sidebar.number_input(
        "Left BC [K]",
        value=300.0
    )

    BC_right = st.sidebar.number_input(
        "Right BC [K]",
        value=300.0
    )

# =====================================================
# SOURCES
# =====================================================

n_sources = st.sidebar.number_input(
    "Number of Sources",
    min_value=0,
    value=1,
    step=1
)

source_positions = []
source_temperatures = []

for i in range(n_sources):

    st.sidebar.markdown(
        f"### Source {i+1}"
    )

    xpos = st.sidebar.number_input(
        f"Position {i+1} [m]",
        value=0.5,
        key=f"x{i}"
    )

    Ts = st.sidebar.number_input(
        f"Temperature {i+1} [K]",
        value=500.0,
        key=f"T{i}"
    )

    source_positions.append(xpos)
    source_temperatures.append(Ts)

# =====================================================
# RUN BUTTON
# =====================================================

if st.button("Run Simulation"):

    Nx = int(L/dx) + 1

    x = np.linspace(
        0,
        L,
        Nx
    )

    dt = Fo * dx**2 / alpha

    Nt = int(t_final/dt)

    # -----------------------------------------
    # Initial Condition
    # -----------------------------------------

    if IC_type == "Uniform Temperature":

        T = np.ones(Nx) * T0

    elif IC_type == "Linear Gradient":

        T = np.linspace(
            TL,
            TR,
            Nx
        )

    else:

        T = (
            T_base
            +
            T_peak
            *
            np.exp(
                -(x-x_center)**2 /
                (2*sigma**2)
            )
        )

    # -----------------------------------------
    # Sources
    # -----------------------------------------

    source_indices = []

    for xpos in source_positions:

        idx = np.argmin(
            np.abs(x - xpos)
        )

        source_indices.append(idx)

    # -----------------------------------------
    # Storage
    # -----------------------------------------

    T_history = np.zeros(
        (Nt+1, Nx)
    )

    T_history[0] = T

    # -----------------------------------------
    # Solver
    # -----------------------------------------

    for n in range(Nt):

        T_new = T.copy()

        for i in range(1, Nx-1):

            T_new[i] = (
                T[i]
                +
                Fo
                *
                (
                    T[i+1]
                    -2*T[i]
                    +T[i-1]
                )
            )

        if BC_type == "Insulated":

            T_new[0] = T_new[1]
            T_new[-1] = T_new[-2]

        else:

            T_new[0] = BC_left
            T_new[-1] = BC_right

        for idx, Ts in zip(
            source_indices,
            source_temperatures
        ):

            T_new[idx] = Ts

        T = T_new

        T_history[n+1] = T

    # =================================================
    # RESULTS
    # =================================================

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Simulation Summary")

        st.write(f"Material: {material}")
        st.write(f"α = {alpha:.3e} m²/s")
        st.write(f"Nx = {Nx}")
        st.write(f"dt = {dt:.4f} s")
        st.write(f"Nt = {Nt}")

    with col2:

        st.metric(
            "Final Max Temperature [K]",
            f"{np.max(T_history[-1]):.2f}"
        )

    # =================================================
    # HEATMAP
    # =================================================

    st.subheader("Temperature Heatmap")

    fig1, ax1 = plt.subplots(
        figsize=(10,5)
    )

    im = ax1.imshow(
        T_history,
        aspect='auto',
        origin='lower',
        extent=[0, L, 0, Nt*dt]
    )

    plt.colorbar(
        im,
        ax=ax1,
        label="Temperature [K]"
    )

    ax1.set_xlabel("Position [m]")
    ax1.set_ylabel("Time [s]")

    st.pyplot(fig1)

    # =================================================
    # FINAL PROFILE
    # =================================================

    st.subheader(
        "Final Temperature Distribution"
    )

    fig2, ax2 = plt.subplots()

    ax2.plot(
        x,
        T_history[-1],
        linewidth=2
    )

    ax2.set_xlabel(
        "Position [m]"
    )

    ax2.set_ylabel(
        "Temperature [K]"
    )

    ax2.grid(True)

    st.pyplot(fig2)