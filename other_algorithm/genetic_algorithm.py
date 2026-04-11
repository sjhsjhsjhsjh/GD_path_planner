import numpy as np


class GeneticAlgorithmPlanner:
    """遗传算法路径规划器。

    染色体是固定长度动作序列，动作空间与 Env.step 一致（默认 4 动作）。
    """

    def __init__(
        self,
        n_actions=4,
        population_size=64,
        elite_count=6,
        mutation_rate=0.02,
        max_steps=1000,
        seed=42,
    ):
        self.n_actions = int(n_actions)
        self.population_size = int(population_size)
        self.elite_count = int(elite_count)
        self.mutation_rate = float(mutation_rate)
        self.max_steps = int(max_steps)
        self.rng = np.random.default_rng(seed)

        if self.population_size <= 0:
            raise ValueError("population_size 必须 > 0")
        if self.max_steps <= 0:
            raise ValueError("max_steps 必须 > 0")
        if self.elite_count < 1:
            raise ValueError("elite_count 必须 >= 1")
        if self.elite_count > self.population_size:
            raise ValueError("elite_count 不能大于 population_size")

    def init_population(self):
        return self.rng.integers(
            low=0,
            high=self.n_actions,
            size=(self.population_size, self.max_steps),
            dtype=np.int32,
        )

    def tournament_select(self, population, fitness, tournament_size=3):
        idxs = self.rng.integers(0, len(population), size=tournament_size)
        best_idx = idxs[int(np.argmax(fitness[idxs]))]
        return population[best_idx]

    def crossover(self, parent_a, parent_b):
        if self.max_steps <= 1:
            return parent_a.copy(), parent_b.copy()
        point = int(self.rng.integers(1, self.max_steps))
        child_a = np.concatenate([parent_a[:point], parent_b[point:]])
        child_b = np.concatenate([parent_b[:point], parent_a[point:]])
        return child_a, child_b

    def mutate(self, chromosome):
        mutation_mask = self.rng.random(self.max_steps) < self.mutation_rate
        if not np.any(mutation_mask):
            return chromosome
        chromosome = chromosome.copy()
        chromosome[mutation_mask] = self.rng.integers(
            0, self.n_actions, size=int(np.sum(mutation_mask))
        )
        return chromosome

    def next_generation(self, population, fitness):
        order = np.argsort(fitness)[::-1]
        elites = population[order[: self.elite_count]].copy()

        new_population = [elites[i] for i in range(len(elites))]
        while len(new_population) < self.population_size:
            p1 = self.tournament_select(population, fitness)
            p2 = self.tournament_select(population, fitness)
            c1, c2 = self.crossover(p1, p2)
            c1 = self.mutate(c1)
            c2 = self.mutate(c2)
            new_population.append(c1)
            if len(new_population) < self.population_size:
                new_population.append(c2)

        return np.stack(new_population, axis=0)
