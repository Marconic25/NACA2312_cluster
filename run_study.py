import numpy as np

# Parametri di test
t = np.linspace(0, 2, 50)
h = 0.05 * np.sin(np.pi * t)      # Oscillazione 5cm
alpha = 5.0 * np.sin(np.pi * t)   # Oscillazione 5°
delta = 10.0 * np.sin(2*np.pi * t) # Flap più veloce

# Scrittura (usa la funzione write_motion_data definita prima)
# write_motion_data('constant/motionData', t, h, alpha, delta)
print("File motionData generato per test singolo.")
