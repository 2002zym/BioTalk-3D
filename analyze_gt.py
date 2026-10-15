import numpy as np
import matplotlib.pyplot as plt

# Load the ground truth files
file1 = '/Users/zym/EATalker/sentence68_liu1_scared_bs.npy'
file2 = '/Users/zym/EATalker/sentence66_liu1_scared_bs.npy'

try:
    data1 = np.load(file1)
    data2 = np.load(file2)
    print(f"Loaded {file1} with shape {data1.shape}")
    print(f"Loaded {file2} with shape {data2.shape}")
except Exception as e:
    print(f"Error loading files: {e}")
    exit()

def analyze_data(data, name):
    print(f"\n--- Analysis for {name} ---")
    
    # 1. Sparsity Analysis
    # Count how many values are close to 0
    zero_threshold = 1e-4
    sparsity = np.mean(np.abs(data) < zero_threshold)
    print(f"Sparsity (values < {zero_threshold}): {sparsity:.4f}")
    
    # Max and Mean values
    print(f"Max value: {np.max(data):.4f}")
    print(f"Mean value: {np.mean(data):.4f}")
    print(f"Median value: {np.median(data):.4f}")
    
    # 2. Velocity Analysis (1st derivative)
    velocity = data[1:] - data[:-1]
    print(f"Max Velocity: {np.max(np.abs(velocity)):.6f}")
    print(f"Mean Velocity (abs): {np.mean(np.abs(velocity)):.6f}")
    
    # 3. Acceleration Analysis (2nd derivative)
    acceleration = velocity[1:] - velocity[:-1]
    print(f"Max Acceleration: {np.max(np.abs(acceleration)):.6f}")
    print(f"Mean Acceleration (abs): {np.mean(np.abs(acceleration)):.6f}")

    return velocity, acceleration

v1, a1 = analyze_data(data1, "sentence68")
v2, a2 = analyze_data(data2, "sentence66")
