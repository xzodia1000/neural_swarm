import copy
from neural_swarm.constants import operators
from neural_swarm.pso.particle import Particle


class PSO:
    def __init__(
        self,
        fun,
        swarm_size,
        alpha,
        beta,
        gamma,
        delta,
        epsilon,
        informants_type,
        informants_size,
        iterations,
        opt,
    ):
        """Initializes a particle swarm optimization algorithm.

        Args:
            fun: The fitness function to be optimized.
            swarm_size (int): Number of particles in the swarm.
            alpha, beta, gamma, delta, epsilon: Coefficients for the velocity and position update equations.
            informants_type: Type of informant selection strategy for each particle.
            informants_size (int): Number of informants for each particle.
            iterations (int): Number of iterations to run the optimization.
            opt (str): Optimization operator (minimize or maximize).
        """
        self.fun = fun  # Fitness function
        self.swarm_size = swarm_size  # Size of the particle swarm
        self.alpha = alpha  # Coefficient for velocity scaling
        self.beta = beta  # Coefficient for personal best influence
        self.gamma = gamma  # Coefficient for local best influence
        self.delta = delta  # Coefficient for global best influence
        self.epsilon = epsilon  # Coefficient for position update scaling
        self.informants_type = informants_type  # Type of informants selection strategy
        self.informants_size = informants_size  # Number of informants per particle
        self.iterations = iterations  # Number of iterations for optimization
        self.opt = operators.get(opt)  # Optimization function (minimize or maximize)

        self.particles = []  # List of particles in the swarm
        self.global_best = None  # Global best particle
        self.global_best_fitness = None  # Global best fitness
        self.global_best_acc = None  # Global best accuracy

        self.init_particles()  # Initialize the particles
        self.update_global_best()  # Update the global best particle
        self.init_informants()  # Initialize informants for each particle

    def init_particles(self):
        """Initializes particles in the swarm and computes their fitness."""
        for _ in range(self.swarm_size):
            particle = Particle(self.fun)  # Create a new particle
            particle.compute_fitness(self.opt)  # Compute its fitness
            self.particles.append(particle)  # Add it to the swarm

    def init_informants(self):
        """Initializes the informants for each particle in the swarm."""
        for particle in self.particles:
            particles = self.particles.copy()  # Copy list of particles
            particles.remove(particle)  # Remove current particle
            # Set informants for the particle
            particle.set_informants(
                self.informants_type, self.informants_size, particles, self.global_best
            )

    def get_global_best(self):
        """Returns the global best particle in the swarm.

        Returns:
            Particle: The global best particle.
        """
        return self.global_best

    def update_global_best(self):
        """Updates the global best particle based on the swarm's performance."""
        for particle in self.particles:
            # Update global best if a better particle is found
            if self.global_best is None or self.opt(
                particle.get_fitness(), self.global_best.fitness
            ):
                self.global_best = copy.deepcopy(particle)
                self.global_best_fitness = particle.get_fitness()
                self.global_best_acc = particle.compute_fitness(self.opt)[0]
                self.fun.set_variable(self.global_best.position)

    def evolve(self):
        """Runs the PSO algorithm for a specified number of iterations.

        Yields:
            Tuple: Global best fitness and accuracy after each iteration.
        """
        for _ in range(self.iterations):
            for particle in self.particles:
                # Update the position and velocity of each particle
                particle.move(
                    self.global_best,
                    self.alpha,
                    self.beta,
                    self.gamma,
                    self.delta,
                    self.epsilon,
                    self.opt,
                )
                # Compute the fitness of each particle
                particle.compute_fitness(self.opt)

            self.update_global_best()  # Update the global best particle

            # Yield the global best fitness and accuracy
            yield copy.deepcopy(self.global_best_fitness), copy.deepcopy(
                self.global_best_acc
            )
