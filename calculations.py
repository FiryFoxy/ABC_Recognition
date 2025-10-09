import numpy as np

# --- 1. Definizione dei Pesi e Bias ---

# LAYER dense_14 (Input: 4, Output: 5)
W14 = np.array([
    [-0.12924245, -0.05241918, -0.23293193, -0.5036716,   0.4829337 ],
    [ 0.75072044,  0.47591314,  0.5632465,   0.56317776, -0.67133796],
    [ 0.24167576, -0.35225365,  0.07815648, -0.2000887,   0.6360845 ],
    [-0.7655489,   0.45864648, -0.09990204,  0.33306068, -0.3584951 ]
]).T  # La matrice è (4x5) nel testo, quindi la trasponiamo per avere (5x4)
b14 = np.array([ 0.27452618,  0.09939408, -0.09400602,  0.07851458,  0.0285877 ])

# LAYER dense_15 (Input: 5, Output: 3)
W15 = np.array([
    [ 0.6576484,  -0.23530254,  0.5175244 ],
    [-0.11396888,  0.08356182, -0.8046963 ],
    [-0.54549164, -0.3892803,  -1.2318124 ],
    [ 0.2931416,  -0.9409266,   0.3900506 ],
    [-1.1874342,   0.48640645,  0.50949097]
]).T # La matrice è (5x3) nel testo, quindi la trasponiamo per avere (3x5)
b15 = np.array([-0.02902211, -0.13917246,  0.18778509])

# --- 2. Input di Esempio ---
X = np.array([2.24, 8.00, 7.68, 11.08])
print(f"Input X: {X}\n")

# --- 3. Funzioni di Attivazione ---
def relu(Z):
    """Funzione di Attivazione ReLU: max(0, Z)"""
    return np.maximum(0, Z)

def softmax(Z):
    """Funzione di Attivazione Softmax per l'output di classificazione"""
    # Stabilità numerica: sottrae il max per evitare overflow
    exp_Z = np.exp(Z - np.max(Z))
    return exp_Z / np.sum(exp_Z)

# --- 4. Calcolo Layer dense_14 (ReLU) ---

# Calcolo del Net Input (Z14)
# Z = W * X + b
Z14 = np.dot(W14, X) + b14
A14 = relu(Z14)

print("--- Calcolo Layer dense_14 (ReLU) ---")
print(f"Dimensione Pesi (W14): {W14.shape}, Dimensione Bias (b14): {b14.shape}")
print(f"Net Input (Z14): W14 * X + b14")
print(f"Z14:\n{Z14}")
print(f"Attivazione (A14): ReLU(Z14)")
print(f"A14:\n{A14}\n")

# --- 5. Calcolo Layer dense_15 (Softmax) ---

# Calcolo del Net Input (Z15)
# Z = W * A_prev + b
Z15 = np.dot(W15, A14) + b15
Y_pred = softmax(Z15)

print("--- Calcolo Layer dense_15 (Softmax) ---")
print(f"Dimensione Pesi (W15): {W15.shape}, Dimensione Bias (b15): {b15.shape}")
print(f"Net Input (Z15): W15 * A14 + b15")
print(f"Z15:\n{Z15}")
print(f"Output Finale (Y_pred): Softmax(Z15)")
print(f"Y_pred (Probabilità):\n{Y_pred}")
print(f"Somma delle probabilità (dovrebbe essere 1): {np.sum(Y_pred)}")