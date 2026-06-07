import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# MATERIAL DATABASE
# =====================================================

materials = {
    "1": ("Aluminum", 237, 2700, 897),
    "2": ("Copper", 401, 8960, 385),
    "3": ("Iron", 80.2, 7870, 449),
    "4": ("Steel", 50.2, 7850, 470),
    "5": ("Brass", 109, 8500, 380),
    "6": ("Titanium", 21.9, 4500, 522),
    "7": ("Concrete", 1.7, 2400, 880),
    "8": ("Glass", 1.05, 2500, 840),
    "9": ("Water", 0.60, 1000, 4186),
    "10": ("Custom", None, None, None)
}

# =====================================================
# INPUT SECTION
# =====================================================

print("="*60)
print("1D TRANSIENT HEAT CONDUCTION")
print("WITH FIXED-TEMPERATURE POINT SOURCES")
print("="*60)

# ---------------- Material ----------------

print("\nAvailable Materials:")

for key, (name, k, rho, cp) in materials.items():
    print(f"{key}. {name}")

mat_choice = input("\nSelect Material: ")

name, k, rho, cp = materials[mat_choice]

if mat_choice == "10":

    k = float(input("k [W/m-K]: "))
    rho = float(input("rho [kg/m³]: "))
    cp = float(input("cp [J/kg-K]: "))

alpha = k/(rho*cp)

# ---------------- Domain ----------------

L = float(input("\nDomain Length [m]: "))

# Fixed grid spacing
dx = 0.1

# ---------------- Simulation Time ----------------

t_final = float(
    input("Final Simulation Time [s]: ")
)

# ---------------- Initial Condition ----------------

print("\nInitial Condition")

print("1. Uniform Temperature")
print("2. Linear Gradient")
print("3. Gaussian Hot Spot")

IC_type = input("Select: ")

if IC_type == "1":

    T0 = float(
        input("Initial Temperature [K]: ")
    )

elif IC_type == "2":

    TL = float(
        input("Left Temperature [K]: ")
    )

    TR = float(
        input("Right Temperature [K]: ")
    )

elif IC_type == "3":

    T_base = float(
        input("Base Temperature [K]: ")
    )

    T_peak = float(
        input("Peak Increase [K]: ")
    )

    x_center = float(
        input("Center Position [m]: ")
    )

    sigma = float(
        input("Sigma [m]: ")
    )

# ---------------- Boundary Condition ----------------

print("\nBoundary Condition")

print("1. Insulated")
print("2. Fixed Temperature")

BC_type = input("Select: ")

if BC_type == "2":

    BC_left = float(
        input("Left Boundary Temperature [K]: ")
    )

    BC_right = float(
        input("Right Boundary Temperature [K]: ")
    )

# ---------------- Heat Sources ----------------

n_sources = int(
    input("\nNumber of Point Heat Sources: ")
)

source_positions = []
source_temperatures = []

for n in range(n_sources):

    print(f"\nHeat Source #{n+1}")

    xpos = float(
        input("Position [m]: ")
    )

    Tsource = float(
        input("Source Temperature [K]: ")
    )

    source_positions.append(xpos)
    source_temperatures.append(Tsource)

# =====================================================
# GRID
# =====================================================

Nx = int(L/dx) + 1

x = np.linspace(0, L, Nx)

# =====================================================
# TIME STEP
# =====================================================

Fo = 0.1

dt = Fo * dx**2 / alpha

Nt = int(t_final/dt)

# =====================================================
# SUMMARY
# =====================================================

print("\n" + "="*60)
print("SIMULATION SUMMARY")
print("="*60)

print(f"Material : {name}")
print(f"k         = {k:.3f} W/m-K")
print(f"rho       = {rho:.3f} kg/m³")
print(f"cp        = {cp:.3f} J/kg-K")
print(f"alpha     = {alpha:.3e} m²/s")

print(f"\nLength    = {L:.2f} m")
print(f"dx        = {dx:.2f} m")
print(f"Nx        = {Nx}")

print(f"\nFo        = {Fo}")
print(f"dt        = {dt:.4f} s")
print(f"Nt        = {Nt}")

# =====================================================
# INITIAL CONDITION
# =====================================================

if IC_type == "1":

    T = np.ones(Nx) * T0

elif IC_type == "2":

    T = np.linspace(TL, TR, Nx)

elif IC_type == "3":

    T = (
        T_base
        +
        T_peak
        *
        np.exp(
            -(x - x_center)**2 /
            (2*sigma**2)
        )
    )

# =====================================================
# SOURCE INDICES
# =====================================================

source_indices = []

for xpos in source_positions:

    idx = np.argmin(
        np.abs(x - xpos)
    )

    source_indices.append(idx)

# =====================================================
# STORAGE
# =====================================================

T_history = np.zeros((Nt+1, Nx))

T_history[0] = T

# =====================================================
# SOLVER (EXPLICIT FTCS)
# =====================================================

for n in range(Nt):

    T_new = T.copy()

    # Interior nodes
    for i in range(1, Nx-1):

        T_new[i] = (
            T[i]
            +
            Fo *
            (
                T[i+1]
                -
                2*T[i]
                +
                T[i-1]
            )
        )

    # Boundary conditions

    if BC_type == "1":

        # Insulated

        T_new[0] = T_new[1]
        T_new[-1] = T_new[-2]

    elif BC_type == "2":

        # Fixed temperature

        T_new[0] = BC_left
        T_new[-1] = BC_right

    # Internal fixed-temperature sources

    for idx, Ts in zip(
        source_indices,
        source_temperatures
    ):

        T_new[idx] = Ts

    T = T_new

    T_history[n+1] = T

# =====================================================
# TIME ARRAY
# =====================================================

time = np.linspace(
    0,
    Nt*dt,
    Nt+1
)

# =====================================================
# HEATMAP
# =====================================================

plt.figure(figsize=(12,6))

im = plt.imshow(
    T_history,
    aspect='auto',
    origin='lower',
    extent=[0, L, 0, Nt*dt]
)

plt.colorbar(
    im,
    label='Temperature [K]'
)

plt.xlabel('Position [m]')
plt.ylabel('Time [s]')

plt.title(
    f'1D Heat Conduction - {name}'
)

plt.tight_layout()
plt.show()

# =====================================================
# FINAL TEMPERATURE PROFILE
# =====================================================

plt.figure(figsize=(10,5))

plt.plot(
    x,
    T_history[-1],
    linewidth=2
)

plt.xlabel('Position [m]')
plt.ylabel('Temperature [K]')

plt.title(
    'Final Temperature Distribution'
)

plt.grid(True)

plt.tight_layout()
plt.show()