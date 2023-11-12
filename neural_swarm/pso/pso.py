import copy

import numpy as np
from neural_swarm.pso.constants import operators
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
        self.fun = fun
        self.swarm_size = swarm_size
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        self.informants_type = informants_type
        self.informants_size = informants_size
        self.iterations = iterations
        self.opt = operators.get(opt)

        self.particles = []
        self.global_best = None
        self.global_best_fitness = None
        self.global_best_position = None
        self.global_best_acc = None

        self.init_particles()
        self.update_global_best()
        self.init_informants()

    def init_particles(self):
        for _ in range(self.swarm_size):
            particle = Particle(self.fun)
            particle.compute_fitness(self.opt)
            self.particles.append(particle)

    def init_informants(self):
        for particle in self.particles:
            particles = self.particles.copy()
            particles.remove(particle)
            particle.set_informants(
                self.informants_type, self.informants_size, particles, self.global_best
            )

    def get_global_best(self):
        return self.global_best

    def update_global_best(self):
        for particle in self.particles:
            if self.global_best is None or self.opt(
                particle.get_fitness(), self.global_best.fitness
            ):
                self.global_best = particle
                self.global_best_fitness = particle.get_fitness()
                self.global_best_position = np.copy(particle.position)
                self.global_best_acc = particle.compute_fitness(self.opt)[0]

    def evolve(self):
        loss = []
        acc = []
        decrement = (self.epsilon - 0.4) / 1000
        for i in range(self.iterations):
            for particle in self.particles:
                particle.move(
                    self.global_best_position,
                    self.alpha,
                    self.beta,
                    self.gamma,
                    self.delta,
                    self.epsilon,
                    self.opt,
                )
                particle.compute_fitness(self.opt)
            self.update_global_best()
            self.init_informants()
            print("Epoch " + str(i + 1) + ": " + str(self.global_best_fitness))
            loss.append(self.global_best_fitness)
            acc.append(self.global_best_acc)
            self.epsilon -= decrement

        return acc, loss
