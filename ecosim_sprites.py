import math
import random
import sys
from dataclasses import dataclass, replace
from typing import Tuple, List, Optional

import numpy as np
import pygame

# --- Si tu veux forcer des chemins absolus, modifie ici ---
SPRITE_PRED_PATH = "predator.png"   # ex: "/mnt/data/predator.png"
SPRITE_PREY_PATH = "prey.png"       # ex: "/mnt/data/prey.png"
SPRITE_FOOD_PATH = "ressource.png"  # ex: "/mnt/data/ressource.png"

# =======================
# Param GUI (Tkinter)
# =======================
def ask_params():
    """Fenêtre Tkinter simplifiée pour saisir les paramètres essentiels."""
    base_cfg = dict(
        W=1000,
        H=700,
        sidebar=260,
        fps=60,
        init_prey=30,
        init_pred=6,
        init_food=120,
        food_spawn_rate=0.02,
        food_energy=22.0,
        move_cost=0.02,
        idle_cost=0.004,
        hunt_cost=0.04,
        repro_energy_prey=65.0,
        repro_energy_pred=90.0,
        mut_rate=0.12,
        mut_scale=0.15,
        seed=7,
    )

    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception:
        # Pas de Tkinter ? On renvoie des valeurs par défaut
        return base_cfg

    root = tk.Tk()
    root.title("EcoSim — Paramètres essentiels")

    main = ttk.Frame(root, padding=12)
    main.grid(row=0, column=0, sticky="nsew")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    intro = ttk.Label(
        main,
        text="Ajustez les paramètres principaux de la simulation." \
             " Chaque champ est accompagné d'une courte explication pour clarifier son rôle.",
        wraplength=360,
        justify="left"
    )
    intro.grid(row=0, column=0, columnspan=2, pady=(0, 10), sticky="w")

    fields = [
        ("Largeur (px)", "W", int,
         "Définit la largeur totale de la scène de simulation."),
        ("Hauteur (px)", "H", int,
         "Hauteur totale de la scène : augmentez-la pour plus d'espace vertical."),
        ("Proies initiales", "init_prey", int,
         "Nombre de proies générées au démarrage. Elles se nourrissent des ressources."),
        ("Prédateurs initiaux", "init_pred", int,
         "Nombre de prédateurs présents au lancement. Ils chassent les proies."),
        ("Ressources initiales", "init_food", int,
         "Quantité de nourriture semée au début dans l'environnement."),
        ("Taux d'apparition de nourriture", "food_spawn_rate", float,
         "Probabilité par tick qu'une ressource aléatoire apparaisse."),
        ("Énergie d'une ressource", "food_energy", float,
         "Quantité d'énergie gagnée par une proie lorsqu'elle consomme une ressource."),
        ("Intensité de mutation", "mut_rate", float,
         "Probabilité qu'un gène (vitesse, vision, agressivité) mute à chaque reproduction."),
        ("Graine aléatoire", "seed", int,
         "Fixe la graine de l'aléatoire pour rendre la simulation reproductible."),
    ]

    entries = {}
    row = 1
    for label, key, caster, tooltip in fields:
        ttk.Label(main, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=(4, 0))
        entry = ttk.Entry(main)
        entry.insert(0, str(base_cfg[key]))
        entry.grid(row=row, column=1, sticky="ew", pady=(4, 0))
        main.columnconfigure(1, weight=1)
        entries[key] = (entry, caster)

        ttk.Label(main, text=tooltip, wraplength=360, justify="left").grid(
            row=row + 1, column=0, columnspan=2, sticky="w", padx=(0, 8), pady=(0, 6)
        )
        row += 2

    info = ttk.Label(
        main,
        text="Les autres paramètres utilisent des valeurs équilibrées par défaut.",
        wraplength=360,
        justify="left"
    )
    info.grid(row=row, column=0, columnspan=2, pady=(6, 12), sticky="we")

    def parse_value(key, widget, caster):
        raw = widget.get().strip()
        if not raw:
            return base_cfg[key]
        try:
            return caster(float(raw)) if caster is int else caster(raw)
        except Exception:
            return base_cfg[key]

    def on_validate():
        cfg = base_cfg.copy()
        for key, (entry_widget, caster) in entries.items():
            cfg[key] = parse_value(key, entry_widget, caster)
        # mut_rate pilote également mut_scale pour rester cohérent
        cfg["mut_scale"] = max(0.01, cfg["mut_rate"] * 1.25)
        root.destroy()
        return cfg

    result = {}

    def on_ok():
        result.update(on_validate())

    ttk.Button(main, text="Lancer la simulation", command=on_ok).grid(
        row=row + 1, column=0, columnspan=2, pady=(0, 4)
    )

    root.mainloop()
    return result or base_cfg.copy()

# =======================
# Utils
# =======================
def clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)

def rand_range(a, b):
    return a + (b - a) * random.random()

def norm(vec: np.ndarray):
    n = np.linalg.norm(vec)
    return vec / n if n > 1e-8 else vec

def wrap_pos(x, y, W, H):
    return x % W, y % H

def jitter(scale=0.5):
    return np.array([rand_range(-scale, scale), rand_range(-scale, scale)], dtype=float)

# =======================
# Gènes
# =======================
@dataclass
class Genes:
    max_speed: float
    view_radius: float
    aggressiveness: float  # surtout pour prédateurs

    def mutated(self, mut_rate, mut_scale):
        if random.random() < mut_rate:
            max_speed = max(0.5, self.max_speed * (1.0 + np.random.normal(0, mut_scale)))
        else:
            max_speed = self.max_speed
        if random.random() < mut_rate:
            view_radius = max(15.0, self.view_radius * (1.0 + np.random.normal(0, mut_scale)))
        else:
            view_radius = self.view_radius
        if random.random() < mut_rate:
            aggressiveness = clamp(self.aggressiveness + np.random.normal(0, mut_scale), 0.0, 1.0)
        else:
            aggressiveness = self.aggressiveness
        return Genes(max_speed, view_radius, aggressiveness)

# =======================
# Sprites helpers
# =======================
def load_sprite(path: str, scale: float = 1.0) -> Optional[pygame.Surface]:
    try:
        img = pygame.image.load(path).convert_alpha()
        if scale != 1.0:
            img = pygame.transform.rotozoom(img, 0, scale)
        return img
    except Exception:
        return None

def blit_center(surf: pygame.Surface, img: pygame.Surface, pos):
    rect = img.get_rect(center=(int(pos[0]), int(pos[1])))
    surf.blit(img, rect)

# =======================
# Entités
# =======================
class Food:
    __slots__ = ("pos", "energy")
    def __init__(self, pos: np.ndarray, food_energy: float):
        self.pos = pos.astype(float)
        self.energy = food_energy

    def draw(self, surf: pygame.Surface, sprite: Optional[pygame.Surface]):
        if sprite is None:
            pygame.draw.circle(surf, (66, 176, 72), (int(self.pos[0]), int(self.pos[1])), 3)
        else:
            blit_center(surf, sprite, self.pos)

class Creature:
    def __init__(self, pos, genes: Genes, energy: float, color: Tuple[int,int,int], world_w, world_h):
        self.pos = pos.astype(float)
        self.vel = jitter(1.0)
        self.genes = genes
        self.energy = energy
        self.col = color
        self.alive = True
        self.age = 0
        self.W = world_w
        self.H = world_h

    def step_base(self):
        self.age += 1
        self.pos += self.vel
        self.pos[0], self.pos[1] = wrap_pos(self.pos[0], self.pos[1], self.W, self.H)
        self.vel *= 0.99

    def steer_to(self, target: np.ndarray, intensity=1.0):
        desired = target - self.pos
        if np.linalg.norm(desired) < 1e-6:
            return
        desired = norm(desired) * self.genes.max_speed
        steer = (desired - self.vel) * 0.4 * intensity
        self.vel += steer

    def avoid(self, target: np.ndarray, intensity=1.0):
        away = self.pos - target
        if np.linalg.norm(away) < 1e-6:
            away = jitter(1.0)
        away = norm(away) * self.genes.max_speed
        steer = (away - self.vel) * 0.6 * intensity
        self.vel += steer

class Prey(Creature):
    RADIUS = 8
    BASE_SPEED = (1.2, 3.1)
    VIEW = (45, 140)

    def __init__(self, pos, genes: Genes, energy: float, world_w, world_h):
        super().__init__(pos, genes, energy, (60, 120, 255), world_w, world_h)

    @staticmethod
    def rand(world_w, world_h):
        genes = Genes(
            max_speed=rand_range(*Prey.BASE_SPEED),
            view_radius=rand_range(*Prey.VIEW),
            aggressiveness=0.0
        )
        pos = np.array([rand_range(0, world_w), rand_range(0, world_h)], dtype=float)
        return Prey(pos, genes, energy=rand_range(35, 75), world_w=world_w, world_h=world_h)

    def think(self, foods: List[Food], predators: List["Predator"], move_cost, idle_cost, hunt_cost):
        nearest_pred = None
        dmin = 1e9
        for p in predators:
            d = np.linalg.norm(p.pos - self.pos)
            if d < dmin:
                dmin = d
                nearest_pred = p
        if nearest_pred and dmin < self.genes.view_radius * 0.9:
            self.avoid(nearest_pred.pos, intensity=1.0)
            self.energy -= hunt_cost * 0.5
        else:
            nearest_food = None
            dmin = 1e9
            for f in foods:
                d = np.linalg.norm(f.pos - self.pos)
                if d < dmin:
                    dmin = d
                    nearest_food = f
            if nearest_food and dmin < self.genes.view_radius:
                self.steer_to(nearest_food.pos, intensity=1.0)
            else:
                self.vel += jitter(0.2)

        speed = np.linalg.norm(self.vel)
        self.energy -= idle_cost + move_cost * (speed / max(1e-6, self.genes.max_speed))

    def eat(self, foods: List[Food]):
        for i, f in enumerate(foods):
            if np.linalg.norm(f.pos - self.pos) < self.RADIUS + 6:
                self.energy += f.energy
                del foods[i]
                break

    def reproduce(self, repro_energy_prey, mut_rate, mut_scale):
        if self.energy >= repro_energy_prey:
            child_energy = self.energy * 0.4
            self.energy *= 0.6
            g = self.genes.mutated(mut_rate, mut_scale)
            g.max_speed = clamp(g.max_speed, Prey.BASE_SPEED[0], Prey.BASE_SPEED[1])
            g.view_radius = clamp(g.view_radius, Prey.VIEW[0], Prey.VIEW[1])
            g.aggressiveness = 0.0
            child = Prey(self.pos.copy(), g, energy=child_energy, world_w=self.W, world_h=self.H)
            child.vel = norm(self.vel + jitter(1.0)) * g.max_speed
            return child
        return None

    def draw(self, surf: pygame.Surface, sprite: Optional[pygame.Surface]):
        if sprite is None:
            pygame.draw.circle(surf, (60,120,255), (int(self.pos[0]), int(self.pos[1])), self.RADIUS)
        else:
            # la proie n'est pas orientée : on la blit telle quelle
            blit_center(surf, sprite, self.pos)

class Predator(Creature):
    RADIUS = 10
    BASE_SPEED = (1.0, 3.6)
    VIEW = (70, 200)

    def __init__(self, pos, genes: Genes, energy: float, world_w, world_h):
        super().__init__(pos, genes, energy, (225,60,60), world_w, world_h)

    @staticmethod
    def rand(world_w, world_h):
        genes = Genes(
            max_speed=rand_range(*Predator.BASE_SPEED),
            view_radius=rand_range(*Predator.VIEW),
            aggressiveness=rand_range(0.3, 0.8)
        )
        pos = np.array([rand_range(0, world_w), rand_range(0, world_h)], dtype=float)
        return Predator(pos, genes, energy=rand_range(65, 110), world_w=world_w, world_h=world_h)

    def think(self, preys: List[Prey], move_cost, idle_cost, hunt_cost):
        nearest_prey = None
        dmin = 1e9
        for pr in preys:
            d = np.linalg.norm(pr.pos - self.pos)
            if d < dmin:
                dmin = d
                nearest_prey = pr
        if nearest_prey and dmin < self.genes.view_radius:
            self.steer_to(nearest_prey.pos, intensity=0.8 + 0.6*self.genes.aggressiveness)
            self.energy -= hunt_cost
        else:
            self.vel += jitter(0.15)
            self.energy -= idle_cost

        speed = np.linalg.norm(self.vel)
        self.energy -= move_cost * (speed / max(1e-6, self.genes.max_speed))

    def eat(self, preys: List[Prey]):
        for i, p in enumerate(preys):
            if np.linalg.norm(p.pos - self.pos) < self.RADIUS + p.RADIUS - 1:
                self.energy += 0.6 * max(35.0, p.energy)
                del preys[i]
                break

    def reproduce(self, repro_energy_pred, mut_rate, mut_scale):
        if self.energy >= repro_energy_pred:
            child_energy = self.energy * 0.45
            self.energy *= 0.55
            g = self.genes.mutated(mut_rate, mut_scale)
            g.max_speed = clamp(g.max_speed, Predator.BASE_SPEED[0], Predator.BASE_SPEED[1])
            g.view_radius = clamp(g.view_radius, Predator.VIEW[0], Predator.VIEW[1])
            g.aggressiveness = clamp(g.aggressiveness, 0.0, 1.0)
            child = Predator(self.pos.copy(), g, energy=child_energy, world_w=self.W, world_h=self.H)
            child.vel = norm(self.vel + jitter(1.0)) * g.max_speed
            return child
        return None

    def draw(self, surf: pygame.Surface, sprite: Optional[pygame.Surface]):
        if sprite is None:
            pygame.draw.circle(surf, (225,60,60), (int(self.pos[0]), int(self.pos[1])), self.RADIUS)
        else:
            # orienter le sprite selon la vitesse
            v = self.vel
            ang_deg = -math.degrees(math.atan2(v[1], v[0])) if np.linalg.norm(v) > 1e-6 else 0.0
            rotated = pygame.transform.rotozoom(sprite, ang_deg, 1.0)
            blit_center(surf, rotated, self.pos)

# =======================
# Monde + Stats
# =======================
class World:
    def __init__(self, cfg):
        self.cfg = cfg
        self.W = cfg["W"]
        self.H = cfg["H"]
        self.sidebar = cfg["sidebar"]
        self.scene_w = self.W - self.sidebar

        self.preys: List[Prey] = []
        self.predators: List[Predator] = []
        self.foods: List[Food] = []
        self.tick = 0
        self.paused = False

        # Sprites
        self.spr_prey = load_sprite(SPRITE_PREY_PATH, scale=0.2)
        self.spr_pred = load_sprite(SPRITE_PRED_PATH, scale=0.3)
        self.spr_food = load_sprite(SPRITE_FOOD_PATH, scale=0.1)

        # Stats historiques (pour mini-graphes)
        self.hist_len = 300
        self.hist_prey = []
        self.hist_pred = []
        self.hist_mean_prey_speed = []
        self.hist_mean_pred_speed = []

    def populate(self):
        rnd = np.random.RandomState(self.cfg["seed"])
        random.seed(self.cfg["seed"])
        np.random.seed(self.cfg["seed"])
        self.preys = [Prey.rand(self.scene_w, self.H) for _ in range(self.cfg["init_prey"])]
        self.predators = [Predator.rand(self.scene_w, self.H) for _ in range(self.cfg["init_pred"])]
        self.foods = [Food(np.array([rand_range(0, self.scene_w), rand_range(0, self.H)], dtype=float),
                           self.cfg["food_energy"]) for _ in range(self.cfg["init_food"])]

    def reset(self):
        self.__init__(self.cfg)
        self.populate()

    def spawn_food(self, n=1, pos=None):
        if pos is None:
            for _ in range(n):
                self.foods.append(Food(np.array([rand_range(0, self.scene_w), rand_range(0, self.H)], dtype=float),
                                       self.cfg["food_energy"]))
        else:
            for _ in range(n):
                self.foods.append(Food(np.array(pos, dtype=float) + jitter(20),
                                       self.cfg["food_energy"]))

    def step(self):
        if self.paused:
            return
        self.tick += 1

        if random.random() < self.cfg["food_spawn_rate"]:
            self.spawn_food(n=random.randint(1, 2))

        # comportements
        for pr in self.preys:
            pr.think(self.foods, self.predators,
                     self.cfg["move_cost"], self.cfg["idle_cost"], self.cfg["hunt_cost"])
        for pd in self.predators:
            pd.think(self.preys, self.cfg["move_cost"], self.cfg["idle_cost"], self.cfg["hunt_cost"])

        # interactions + vie/mort
        for pr in self.preys:
            pr.step_base()
            pr.pos[0] = pr.pos[0] % self.scene_w  # wrap limité à la scène
            pr.eat(self.foods)
            if pr.energy <= 0:
                pr.alive = False
        for pd in self.predators:
            pd.step_base()
            pd.pos[0] = pd.pos[0] % self.scene_w
            pd.eat(self.preys)
            if pd.energy <= 0:
                pd.alive = False

        self.preys = [p for p in self.preys if p.alive]
        self.predators = [p for p in self.predators if p.alive]

        # reproductions
        newborns_p = []
        for p in self.preys:
            b = p.reproduce(self.cfg["repro_energy_prey"], self.cfg["mut_rate"], self.cfg["mut_scale"])
            if b: newborns_p.append(b)
        self.preys.extend(newborns_p)

        newborns_d = []
        for d in self.predators:
            b = d.reproduce(self.cfg["repro_energy_pred"], self.cfg["mut_rate"], self.cfg["mut_scale"])
            if b: newborns_d.append(b)
        self.predators.extend(newborns_d)

        # met à jour historiques
        self.hist_prey.append(len(self.preys))
        self.hist_pred.append(len(self.predators))
        self.hist_mean_prey_speed.append(
            float(np.mean([p.genes.max_speed for p in self.preys])) if self.preys else 0.0
        )
        self.hist_mean_pred_speed.append(
            float(np.mean([p.genes.max_speed for p in self.predators])) if self.predators else 0.0
        )
        # garder taille raisonnable
        for arr in (self.hist_prey, self.hist_pred, self.hist_mean_prey_speed, self.hist_mean_pred_speed):
            if len(arr) > self.hist_len:
                del arr[0]

    def draw(self, surf: pygame.Surface, fonts):
        # zone scène
        scene_rect = pygame.Rect(0, 0, self.scene_w, self.H)
        sidebar_rect = pygame.Rect(self.scene_w, 0, self.sidebar, self.H)
        surf.fill((242, 246, 255), scene_rect)
        surf.fill((250, 250, 252), sidebar_rect)

        # nourriture
        for f in self.foods:
            f.draw(surf, self.spr_food)
        # proies
        for p in self.preys:
            p.draw(surf, self.spr_prey)
        # prédateurs
        for p in self.predators:
            p.draw(surf, self.spr_pred)

        # séparateur
        pygame.draw.line(surf, (210, 215, 230), (self.scene_w, 0), (self.scene_w, self.H), 2)

        # HUD dans la sidebar
        self.draw_sidebar(surf, fonts, sidebar_rect)

    def draw_sidebar(self, surf: pygame.Surface, fonts, rect: pygame.Rect):
        pad = 12
        x = rect.x + pad
        y = rect.y + pad
        title = fonts["bold"].render("Évolution (live)", True, (25, 30, 45))
        surf.blit(title, (x, y))
        y += 26

        # Ligne 1: compte proies / preds
        txt = fonts["mono"].render(
            f"Tick: {self.tick}", True, (60, 65, 80)
        )
        surf.blit(txt, (x, y)); y += 20

        counts = fonts["mono"].render(
            f"Preys: {len(self.preys)}   Preds: {len(self.predators)}", True, (60, 65, 80)
        )
        surf.blit(counts, (x, y)); y += 24

        # mini-graph popula
        y = self.draw_mini_plot(surf, x, y, rect.width - 2*pad, 70,
                                self.hist_prey, self.hist_pred,
                                label_left="pop", colors=((76, 129, 235), (230, 91, 91)))

        # Moyennes de vitesse
        y += 4
        means = fonts["mono"].render("Vitesse moyenne", True, (60,65,80))
        surf.blit(means, (x, y)); y += 20

        y = self.draw_mini_plot(surf, x, y, rect.width - 2*pad, 70,
                                self.hist_mean_prey_speed, self.hist_mean_pred_speed,
                                label_left="speed", colors=((76,129,235), (230,91,91)))

        y += 8
        help1 = fonts["tiny"].render("SPACE: Pause  |  R: Reset", True, (100, 105, 120))
        surf.blit(help1, (x, y)); y += 16
        help2 = fonts["tiny"].render("L-Clic: Food  |  R-Clic: Predator  |  Molette: +/- Prey", True, (100, 105, 120))
        surf.blit(help2, (x, y))

    def draw_mini_plot(self, surf, x, y, w, h, series_a, series_b=None, label_left="", colors=((50,50,50),(150,150,150))):
        # fond
        rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(surf, (255,255,255), rect, border_radius=8)
        pygame.draw.rect(surf, (225,230,240), rect, width=1, border_radius=8)

        def draw_series(vals, color):
            if len(vals) < 2:
                return
            v = np.array(vals, dtype=float)
            if np.all(v==0):
                v = v + 1.0
            v = (v - v.min()) / (v.max() - v.min() + 1e-8)
            pts = []
            for i in range(len(v)):
                px = x + 6 + (i / (len(v) - 1)) * (w - 12)
                py = y + h - 6 - v[i] * (h - 12)
                pts.append((px, py))
            pygame.draw.lines(surf, color, False, pts, 2)

        draw_series(series_a, colors[0])
        if series_b is not None:
            draw_series(series_b, colors[1])

        # étiquette
        if label_left:
            lbl = pygame.font.SysFont("consolas", 14).render(label_left, True, (120,125,140))
            surf.blit(lbl, (x+8, y+6))
        return y + h + 6

# =======================
# Main
# =======================
def main():
    cfg = ask_params()

    # seed
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    pygame.init()
    screen = pygame.display.set_mode((cfg["W"], cfg["H"]))
    pygame.display.set_caption("EcoSim — Prédateurs/Proies évolutifs (sprites + GUI)")
    clock = pygame.time.Clock()

    fonts = {
        "bold": pygame.font.SysFont("segoeui,arial", 20, bold=True),
        "mono": pygame.font.SysFont("consolas", 16),
        "tiny": pygame.font.SysFont("consolas", 12),
    }

    world = World(cfg)
    world.populate()

    running = True
    while running:
        dt = clock.tick(cfg["fps"])
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                # uniquement dans la zone scène
                if mx < world.scene_w:
                    if event.button == 1:  # left -> paquet de food
                        n = random.randint(3, 7)
                        world.spawn_food(n=n, pos=(mx, my))
                    elif event.button == 3:  # right -> +1 predator
                        pos = np.array([mx, my], dtype=float)
                        world.predators.append(Predator(pos, Predator.rand(world.scene_w, world.H).genes,
                                                        energy=80.0, world_w=world.scene_w, world_h=world.H))
                    elif event.button == 4:  # wheel up -> +1 prey
                        pos = np.array([mx, my], dtype=float)
                        world.preys.append(Prey(pos, Prey.rand(world.scene_w, world.H).genes,
                                                energy=50.0, world_w=world.scene_w, world_h=world.H))
                    elif event.button == 5:  # wheel down -> -1 prey
                        if world.preys:
                            world.preys.pop()

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    world.paused = not world.paused
                elif event.key == pygame.K_r:
                    world.reset()

        world.step()
        world.draw(screen, fonts)
        pygame.display.flip()

    pygame.quit()
    sys.exit(0)

if __name__ == "__main__":
    main()
