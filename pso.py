import numpy as np

class PSO:
    def __init__(self, fitness_func, dim, bounds, num_particles=30, max_iter=100):
        self.fitness_func = fitness_func
        self.dim = dim
        self.bounds = bounds if isinstance(bounds[0], (list, tuple)) else [bounds] * dim
        self.num_particles = num_particles
        self.max_iter = max_iter

        self.swarm_position = np.random.uniform(
            low=[b[0] for b in self.bounds],
            high=[b[1] for b in self.bounds],
            size=(self.num_particles, self.dim)
        )
        self.swarm_velocity = np.zeros((self.num_particles, self.dim))
        self.pbest_position = np.copy(self.swarm_position)
        self.pbest_fitness = np.array([float('inf')] * self.num_particles)
        self.gbest_position = None
        self.gbest_fitness = float('inf')

    def optimize(self):
        w = 0.5  # inertia weight
        c1 = 1.5  # cognitive constant
        c2 = 1.5  # social constant

        fitness_history = []

        for iteration in range(self.max_iter):
            for i in range(self.num_particles):
                fitness = self.fitness_func(self.swarm_position[i])
                if fitness < self.pbest_fitness[i]:
                    self.pbest_fitness[i] = fitness
                    self.pbest_position[i] = self.swarm_position[i]

                if fitness < self.gbest_fitness:
                    self.gbest_fitness = fitness
                    self.gbest_position = self.swarm_position[i]

            for i in range(self.num_particles):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                cognitive_velocity = c1 * r1 * (self.pbest_position[i] - self.swarm_position[i])
                social_velocity = c2 * r2 * (self.gbest_position - self.swarm_position[i])
                self.swarm_velocity[i] = w * self.swarm_velocity[i] + cognitive_velocity + social_velocity
                self.swarm_position[i] += self.swarm_velocity[i]

                # Clamp position within bounds
                for d in range(self.dim):
                    self.swarm_position[i][d] = np.clip(self.swarm_position[i][d], self.bounds[d][0], self.bounds[d][1])

            fitness_history.append(self.gbest_fitness)

        return self.gbest_position, self.gbest_fitness, fitness_history
