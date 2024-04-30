import random
import math
from matplotlib import pyplot as plt
import numpy as np


def levy_distribution(beta):
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.normal(0, sigma, size=1)[0]
    v = np.random.normal(0, 1, size=1)[0]
    step = u / abs(v) ** (1 / beta)
    return step

def levy_flight(current_pos, beta):
    step = levy_distribution(beta)
    return [(x[0] + step * (x[0] - current_pos[i][0]), x[1] + step * (x[1] - current_pos[i][1])) for i, x in enumerate(current_pos)]

def average_latency(data_centers, user_locations):
    total_latency = 0
    for user in user_locations:
        nearest_data_center = min(data_centers, key=lambda dc: math.dist(list(dc), list(user)))
        total_latency += math.dist(list(nearest_data_center), list(user))
    return total_latency / len(user_locations)


def cuckoo_search(num_data_centers, num_cuckoos, max_iterations, beta, p_a,user_locations):
  """
  Cuckoo Search algorithm for data center placement.

  Args:
      num_data_centers: Number of data centers.
      num_cuckoos: Number of cuckoos (potential placements) in the population.
      max_iterations: Maximum number of iterations.
      beta: Levy flight parameter.
      p_a: Discovery rate of alien eggs.
      user_locations: A list of user location coordinates.

  Returns:
      The best data center placement configuration (list of coordinates).
  """

  # Generate random initial population of cuckoos (data center placements)
  population = [[(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(num_data_centers)] for _ in range(num_cuckoos)]

  # Calculate fitness (average latency) for each cuckoo
  fitness = [average_latency(cuckoo, user_locations) for cuckoo in population]

  # Main loop
  for _ in range(max_iterations):
    # Lévy Flight for each cuckoo
    new_population = []
    for i in range(num_cuckoos):
      current_pos = population[i]
      new_pos = levy_flight(current_pos, beta)
      new_population.append(new_pos)

    # Calculate fitness for new cuckoos
    new_fitness = [average_latency(cuckoo, user_locations) for cuckoo in new_population]

    # Egg laying and replacement
    combined = list(zip(population, fitness))
    combined_new = list(zip(new_population, new_fitness))
    for i, (cuckoo, old_fitness) in enumerate(combined):
      if random.random() < p_a:
        # Replace with new cuckoo if fitness is better
        if new_fitness[i] < old_fitness:
          combined[i] = combined_new[i]

    # Update population and fitness
    population, fitness = zip(*combined)

    # Abandonment (optional)
    # ... (implement logic to remove and replace abandoned cuckoos)

  # Select the best cuckoo (placement with lowest average latency)
  best_index = fitness.index(min(fitness))
  return population[best_index]

# Parameters (adjust as needed)
num_data_centers = 5
num_cuckoos = 20
max_iterations = 50
beta = 1.5
p_a = 0.1  # Discovery rate of alien eggs

user_locations = []
for _ in range(100):
  user_locations.append([random.uniform(0, 100), random.uniform(0, 100)])  # Random X,Y coordinates


# Lists to store the best latency and placement found in each run
best_latencies = []
best_placements = []

# Run the Cuckoo Search 30 times
for _ in range(50):
    best_placement = cuckoo_search(num_data_centers, num_cuckoos, max_iterations, beta, p_a, user_locations)
    best_latency = average_latency(best_placement, user_locations)
    best_latencies.append(best_latency)
    best_placements.append(best_placement)

# Calculate and print the average of the best latencies
average_best_latency = sum(best_latencies) / len(best_latencies)
print(f"Average Best Latency: {average_best_latency}")

# Find and print the best placement
index_of_best_latency = best_latencies.index(min(best_latencies))
best_placement = best_placements[index_of_best_latency]
print(f"Best Data Center Placement: {best_placement}")

# Plot the convergence curve
plt.plot(best_latencies)
plt.xlabel('Run')
plt.ylabel('Best Latency')
plt.title('Convergence Curve')
plt.show()