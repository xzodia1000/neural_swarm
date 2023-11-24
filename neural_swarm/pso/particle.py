import random
import secrets
import numpy as np
from neural_swarm.constants import (
    RANDOM_INFORMANTS,
    LOCAL_NEIGHBOUR_INFORMANTS,
    LOCAL_GLOBAL_INFORMANTS,
)


class Particle:
    def __init__(self, fun):
        """Initializes a particle for use in particle swarm optimization.

        Args:
            fun: The fitness function to evaluate the particle.
        """
        self.fun = fun  # Fitness function
        self.velocity = 0  # Initial velocity
        self.position = None  # Position in the search space
        self.fitness = None  # Fitness value
        self.pbestposition = None  # Personal best position
        self.pbestfitness = None  # Personal best fitness
        self.informants = None  # Neighboring particles (informants)

        self.init_position()  # Initialize the particle's position

    def init_position(self):
        """Initializes the particle's position randomly within the search space."""
        dimension = (
            self.fun.get_dimension()
        )  # Get the dimension from the fitness function
        self.position = np.random.uniform(-1, 1, dimension)  # Random initial position

    def distance(self, particle):
        """Calculates the Euclidean distance to another particle.

        Args:
            particle: The other particle to measure distance to.

        Returns:
            float: Euclidean distance to the other particle.
        """
        return np.linalg.norm(self.position - particle.position)

    def find_neighbours(self, particles, informant_size):
        """Finds the closest neighbours to the particle.

        Args:
            particles: List of all particles in the swarm.
            informant_size: Number of neighbours to find.

        Returns:
            List: Closest neighbours to the particle.
        """
        particles.sort(key=lambda p: self.distance(p))  # Sort particles by distance
        return particles[:informant_size]  # Return the closest neighbours

    def set_informants(
        self, informants_type, informants_size, particles, global_best=None
    ):
        """Sets the informants for the particle based on the specified strategy.

        Args:
            informants_type: Type of informant selection strategy.
            informants_size: Number of informants to select.
            particles: List of all particles in the swarm.
            global_best (optional): The global best particle in the swarm.
        """
        # Select informants based on the chosen strategy
        if informants_type == RANDOM_INFORMANTS:
            self.informants = random.sample(particles, k=informants_size)

        elif informants_type == LOCAL_NEIGHBOUR_INFORMANTS:
            self.informants = self.find_neighbours(particles, informants_size)

        elif informants_type == LOCAL_GLOBAL_INFORMANTS:
            self.informants = self.find_neighbours(particles, informants_size - 1)
            self.informants.append(global_best)

    def get_fitness(self):
        """Returns the fitness of the particle.

        Returns:
            Fitness value of the particle.
        """
        return self.fitness

    def get_personal_best(self):
        """Returns the personal best position and fitness of the particle.

        Returns:
            Tuple: Personal best position and fitness.
        """
        return self.pbestposition, self.pbestfitness

    def compute_fitness(self, opt):
        """Evaluates the fitness of the particle and updates personal best if necessary.

        Args:
            opt: The optimization function (minimize or maximize).

        Returns:
            Tuple: Accuracy and loss for the current position.
        """
        acc, loss = self.fun.evaluate(self.position)  # Evaluate current position
        self.fitness = loss  # Update fitness

        # Update personal best if better than current best
        if self.pbestfitness is None or opt(loss, self.pbestfitness):
            self.pbestfitness = loss
            self.pbestposition = self.position.copy()

        return acc, loss

    def find_local_best(self, opt):
        """Finds the best among the informants based on fitness.

        Args:
            opt: The optimization function (minimize or maximize).

        Returns:
            Particle: The best particle among informants.
        """
        local_best = None
        local_best_fitness = None

        for informant in self.informants:
            _, informant_fitness = informant.get_personal_best()
            if local_best is None or opt(informant_fitness, local_best_fitness):
                local_best = informant
                local_best_fitness = informant_fitness

        return local_best

    def move(self, global_best, alpha, beta, gamma, delta, epsilon, opt):
        """Updates the particle's position and velocity.

        Args:
            global_best: The global best particle in the swarm.
            alpha, beta, gamma, delta, epsilon: Coefficients for the velocity and position update equations.
            opt: The optimization function (minimize or maximize).
        """
        local_best = self.find_local_best(opt)  # Find the local best among informants

        seed = secrets.randbits(128)  # Generate a random seed
        rng = np.random.default_rng(seed)  # Initialize a random number generator

        # Update the velocity based on personal, local, and global bests
        self.velocity = alpha * self.velocity
        self.velocity += (
            beta
            * rng.random(size=self.position.shape)
            * (self.pbestposition - self.position)
        )
        self.velocity += (
            gamma
            * rng.random(size=self.position.shape)
            * (local_best.position - self.position)
        )
        self.velocity += (
            delta
            * rng.random(size=self.position.shape)
            * (global_best.position - self.position)
        )

        self.position += epsilon * self.velocity  # Update the position
