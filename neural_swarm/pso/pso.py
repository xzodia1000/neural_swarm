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
                self.global_best = copy.deepcopy(particle)
                self.global_best_fitness = particle.get_fitness()
                self.global_best_acc = particle.compute_fitness(self.opt)[0]
                self.fun.set_variable(self.global_best.position)

    def evolve(self):
        for _ in range(self.iterations):
            for particle in self.particles:
                particle.move(
                    self.global_best,
                    self.alpha,
                    self.beta,
                    self.gamma,
                    self.delta,
                    self.epsilon,
                    self.opt,
                )
                particle.compute_fitness(self.opt)

            self.update_global_best()

            yield copy.deepcopy(self.global_best_fitness), copy.deepcopy(
                self.global_best_acc
            )
