import random
import math
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

import math

def average_latency(data_centers, user_locations):
  """
  Calculate the average latency between user locations and the nearest data center.

  Args:
    data_centers (list): A list of coordinates representing the locations of data centers.
    user_locations (list): A list of coordinates representing the locations of users.

  Returns:
    float: The average latency between user locations and the nearest data center.
  """
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


# Run Cuckoo Search with sample user locations
best_placement = cuckoo_search(num_data_centers, num_cuckoos, max_iterations, beta, p_a, user_locations)

# Print the final optimized locations for all data centers
print("Final Data Center Locations:")
for i, location in enumerate(best_placement):
  print(f"Data Center {i+1}: ({location[0]}, {location[1]})")

# Calculate and print the final average latency
final_latency = average_latency(best_placement, user_locations)
print(f"\nFinal Average Latency: {final_latency}")


