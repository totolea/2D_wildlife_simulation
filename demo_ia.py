import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass

# ========================
# Config projet
# ========================
SEED = 42
np.random.seed(SEED)

WIDTH, HEIGHT = 800, 600            # taille de la scène (pixels)
POP_SIZE = 150                       # taille de la population
LIFESPAN = 180                       # nombre de pas par génération
MAX_SPEED = 4.0                      # vitesse max d'un dot
MAX_FORCE = 0.4                      # amplitude max d'une accélération (gène)
MUTATION_RATE = 0.02                 # probabilité de muter un gène
ELITISM = 4                          # nombre d'élites copiées telles quelles
GENERATIONS = 80                     # nombre de générations à jouer

# cible & départ
START = np.array([60.0, HEIGHT - 60.0])
TARGET = np.array([WIDTH - 80.0, 80.0])
TARGET_RADIUS = 20.0

# obstacles (x, y, w, h)
OBSTACLES = [
    (140, 260, 520, 20),             # un grand mur horizontal
    (340, 140, 20, 120),             # un pilier vertical
]

# ========================
# Modèle génétique
# ========================
@dataclass
class DNA:
    genes: np.ndarray  # shape (LIFESPAN, 2) accélérations

    @staticmethod
    def random():
        # vecteurs aléatoires bornés par MAX_FORCE
        angles = np.random.uniform(0, 2*np.pi, size=LIFESPAN)
        mags = np.random.uniform(0, MAX_FORCE, size=LIFESPAN)
        genes = np.stack([np.cos(angles)*mags, np.sin(angles)*mags], axis=1)
        return DNA(genes=genes)

    def crossover(self, other: "DNA") -> "DNA":
        # 1-point crossover
        cut = np.random.randint(1, LIFESPAN-1)
        child_genes = np.vstack([self.genes[:cut], other.genes[cut:]])
        return DNA(genes=child_genes)

    def mutate(self):
        mask = np.random.rand(LIFESPAN) < MUTATION_RATE
        if np.any(mask):
            angles = np.random.uniform(0, 2*np.pi, size=mask.sum())
            mags = np.random.uniform(0, MAX_FORCE, size=mask.sum())
            new_vecs = np.stack([np.cos(angles)*mags, np.sin(angles)*mags], axis=1)
            self.genes[mask] = new_vecs

class Dot:
    def __init__(self, dna: DNA | None = None):
        self.pos = START.copy().astype(float)
        self.vel = np.zeros(2, dtype=float)
        self.acc = np.zeros(2, dtype=float)
        self.dna = dna if dna is not None else DNA.random()
        self.alive = True
        self.reached = False
        self.step_reached = None
        self.crashed = False

    def apply_force(self, f):
        self.acc += f

    def update(self, t):
        if not self.alive:
            return
        # appliquer le gène courant
        self.apply_force(self.dna.genes[t])

        # intégration
        self.vel += self.acc
        # clamp speed
        speed = np.linalg.norm(self.vel)
        if speed > MAX_SPEED:
            self.vel = self.vel / speed * MAX_SPEED
        self.pos += self.vel
        self.acc[:] = 0.0

        # contrôles de fin
        # limite écran
        if not (0 <= self.pos[0] <= WIDTH and 0 <= self.pos[1] <= HEIGHT):
            self.alive = False
            self.crashed = True
            return

        # obstacle collision
        for (ox, oy, ow, oh) in OBSTACLES:
            if ox <= self.pos[0] <= ox+ow and oy <= self.pos[1] <= oy+oh:
                self.alive = False
                self.crashed = True
                return

        # cible atteinte ?
        if np.linalg.norm(self.pos - TARGET) <= TARGET_RADIUS:
            self.alive = False
            self.reached = True
            self.step_reached = t

    def fitness(self):
        # score basé sur la distance finale à la cible + bonus si atteinte + pénalité si crash
        d = np.linalg.norm(self.pos - TARGET)
        # base (distance plus petite -> fitness plus grande)
        base = 1.0 / (d + 1e-6)

        bonus = 0.0
        if self.reached:
            # gros bonus, amélioré si atteint tôt
            bonus = 2.0 + (LIFESPAN - self.step_reached) / LIFESPAN

        penalty = 0.5 if self.crashed else 0.0

        return max(1e-6, base + bonus - penalty)

class Population:
    def __init__(self):
        self.dots = [Dot() for _ in range(POP_SIZE)]
        self.generation = 1
        self.t = 0
        self.history = []  # (gen, best_fit, mean_fit)

    def step(self):
        for d in self.dots:
            d.update(self.t)
        self.t += 1
        return self.t >= LIFESPAN or all(not d.alive for d in self.dots)

    def eval(self):
        fits = np.array([d.fitness() for d in self.dots])
        best_idx = int(np.argmax(fits))
        best_fit = float(fits[best_idx])
        mean_fit = float(np.mean(fits))
        self.history.append((self.generation, best_fit, mean_fit))
        return fits, best_idx, best_fit, mean_fit

    def select_parent(self, probs):
        idx = np.random.choice(np.arange(POP_SIZE), p=probs)
        return self.dots[idx].dna

    def reproduce(self, fits):
        # tri pour l'élitisme
        order = np.argsort(fits)[::-1]
        elites = [self.dots[i].dna for i in order[:ELITISM]]

        # roulette wheel
        probs = fits / fits.sum()

        new_dots = []
        # garder les élites telles quelles
        for e in elites:
            new_dots.append(Dot(dna=DNA(genes=e.genes.copy())))

        # remplir le reste avec crossover + mutation
        while len(new_dots) < POP_SIZE:
            p1 = self.select_parent(probs)
            p2 = self.select_parent(probs)
            child = p1.crossover(p2)
            child.mutate()
            new_dots.append(Dot(dna=child))

        self.dots = new_dots
        self.generation += 1
        self.t = 0

# ========================
# Animation matplotlib
# ========================
pop = Population()

fig, ax = plt.subplots(figsize=(WIDTH/100, HEIGHT/100), dpi=100)
ax.set_xlim(0, WIDTH)
ax.set_ylim(HEIGHT, 0)  # y inversé pour coord “écran”
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("Smart Dot (Genetic Rockets)")

# cible
target_circle = plt.Circle(TARGET, TARGET_RADIUS, ec='black', fc='lime', alpha=0.8)
ax.add_patch(target_circle)
# obstacles
ob_rects = []
for (ox, oy, ow, oh) in OBSTACLES:
    r = plt.Rectangle((ox, oy), ow, oh, ec='black', fc='gray', alpha=0.6)
    ob_rects.append(r)
    ax.add_patch(r)

# scatter des dots
scat = ax.scatter([d.pos[0] for d in pop.dots],
                  [d.pos[1] for d in pop.dots],
                  s=12)

# texte HUD
hud = ax.text(10, 20, "", fontsize=10, color='black', bbox=dict(fc='white', alpha=0.7, ec='none'))

def update(_frame):
    global pop
    done = pop.step()
    # maj positions
    xs = [d.pos[0] for d in pop.dots]
    ys = [d.pos[1] for d in pop.dots]
    scat.set_offsets(np.c_[xs, ys])

    # HUD courant
    reached = sum(1 for d in pop.dots if d.reached)
    crashed = sum(1 for d in pop.dots if d.crashed)
    hud.set_text(
        f"Gen {pop.generation}/{GENERATIONS} | Step {pop.t}/{LIFESPAN}\n"
        f"Reached: {reached}  |  Crashed: {crashed}"
    )

    if done:
        fits, best_idx, best_fit, mean_fit = pop.eval()
        # bandeau fitness
        hud.set_text(
            f"Gen {pop.generation}/{GENERATIONS} | Step {pop.t}/{LIFESPAN}\n"
            f"Reached: {sum(1 for d in pop.dots if d.reached)} | "
            f"Crashed: {sum(1 for d in pop.dots if d.crashed)}\n"
            f"Best fitness: {best_fit:.3f} | Mean: {mean_fit:.3f}"
        )
        # nouvelle génération (sauf si on a atteint la limite)
        if pop.generation < GENERATIONS:
            pop.reproduce(fits)
        else:
            anim.event_source.stop()
    return scat, hud

anim = FuncAnimation(fig, update, interval=16, blit=False)
plt.show()
