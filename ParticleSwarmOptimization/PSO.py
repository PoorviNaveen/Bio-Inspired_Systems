import numpy as np

# --- Objective function (traffic delay model, simplified) ---
def traffic_delay(position):
    # position = [green_NS, green_EW]
    green_NS, green_EW = position
    cycle = 120  # total cycle length in seconds

    # Penalty if sum of greens > cycle
    if green_NS + green_EW > cycle:
        return 1e6
    
    # Assume arrival rates (vehicles/minute)
    lambda_NS, lambda_EW = 40, 60
    
    # Simplified delay model: more green -> less delay
    delay = (lambda_NS / (green_NS + 1)) + (lambda_EW / (green_EW + 1))
    return delay

# --- PSO parameters ---
n_particles = 10
n_iterations = 30
w, c1, c2 = 0.7, 1.5, 1.5  # inertia, cognitive, social

# Bounds for green times
lb, ub = np.array([10, 10]), np.array([110, 110])

# --- Initialization ---
positions = np.random.uniform(lb, ub, (n_particles, 2))
velocities = np.random.uniform(-1, 1, (n_particles, 2))
pbest = positions.copy()
pbest_val = np.array([traffic_delay(p) for p in positions])

gbest = pbest[np.argmin(pbest_val)]
gbest_val = np.min(pbest_val)

# --- Main loop ---
for it in range(n_iterations):
    for i in range(n_particles):
        r1, r2 = np.random.rand(2)
        velocities[i] = (w * velocities[i] 
                        + c1 * r1 * (pbest[i] - positions[i]) 
                        + c2 * r2 * (gbest - positions[i]))
        
        positions[i] = np.clip(positions[i] + velocities[i], lb, ub)
        
        val = traffic_delay(positions[i])
        if val < pbest_val[i]:
            pbest[i], pbest_val[i] = positions[i].copy(), val
    
    if np.min(pbest_val) < gbest_val:
        gbest, gbest_val = pbest[np.argmin(pbest_val)], np.min(pbest_val)
    
    print(f"Iter {it+1}: Best delay = {gbest_val:.4f}, Best green times = {gbest}")

# --- Result ---
print("\nOptimal signal timings:")
print(f"North-South green: {gbest[0]:.2f} sec")
print(f"East-West green:   {gbest[1]:.2f} sec")
print(f"Minimum waiting score = {gbest_val:.4f}")

