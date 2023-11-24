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
        self.fun = fun
        self.velocity = 0
        self.position = None
        self.fitness = None
        self.pbestposition = None
        self.pbestfitness = None
        self.informants = None

        self.init_position()

    def init_position(self):
        dimension = self.fun.get_dimension()
        self.position = np.random.uniform(-1, 1, dimension)

    def distance(self, particle):
        return np.linalg.norm(self.position - particle.position)

    def find_neighbours(self, particles, informant_size):
        particles.sort(key=lambda p: self.distance(p))
        return particles[:informant_size]

    def set_informants(
        self, informants_type, informants_size, particles, global_best=None
    ):
        if informants_type == RANDOM_INFORMANTS:
            self.informants = random.sample(particles, k=informants_size)

        elif informants_type == LOCAL_NEIGHBOUR_INFORMANTS:
            self.informants = self.find_neighbours(particles, informants_size)

        elif informants_type == LOCAL_GLOBAL_INFORMANTS:
            self.informants = self.find_neighbours(particles, informants_size - 1)
            self.informants.append(global_best)

    def get_fitness(self):
        return self.fitness

    def get_personal_best(self):
        return self.pbestposition, self.pbestfitness

    def compute_fitness(self, opt):
        acc, loss = self.fun.evaluate(self.position)
        self.fitness = loss

        if self.pbestfitness is None or opt(loss, self.pbestfitness):
            self.pbestfitness = loss
            self.pbestposition = self.position.copy()

        return acc, loss

    def find_local_best(self, opt):
        local_best = None
        local_best_fitness = None

        for informant in self.informants:
            _, informant_fitness = informant.get_personal_best()
            if local_best is None or opt(informant_fitness, local_best_fitness):
                local_best = informant
                local_best_fitness = informant_fitness

        return local_best

    def move(self, global_best, alpha, beta, gamma, delta, epsilon, opt):
        local_best = self.find_local_best(opt)

        seed = secrets.randbits(128)
        rng = np.random.default_rng(seed)

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

        self.position += epsilon * self.velocity
