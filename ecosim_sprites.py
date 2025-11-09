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
SPRITE_BG_PATH = "background.png"    # image de fond pour la zone de simulation

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
        init_prey=6,
        init_pred=2,
        init_food=120,
        food_spawn_rate=0.02,
        food_energy=22.0,
        move_cost=0.024,
        idle_cost=0.005,
        hunt_cost=0.05,
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
        # Ne pas préremplir largeur/hauteur pour éviter des valeurs par défaut visibles
        if key not in ("W", "H"):
            entry.insert(0, str(base_cfg[key]))
        entry.grid(row=row, column=1, sticky="ew", pady=(4, 0))
        main.columnconfigure(1, weight=1)
        entries[key] = (entry, caster)

        if tooltip:
            ttk.Label(main, text=tooltip, wraplength=360, justify="left").grid(
                row=row + 1, column=0, columnspan=2, sticky="w", padx=(0, 8), pady=(0, 6)
            )
            row += 2
        else:
            row += 1

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
        self.max_energy = max(energy, 1.0)
        self.col = color
        self.alive = True
        self.age = 0
        self.W = world_w
        self.H = world_h
        self.is_chasing = False

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
                self.max_energy = max(self.max_energy, self.energy)
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
        self.is_chasing = False
        nearest_prey = None
        dmin = 1e9
        for pr in preys:
            d = np.linalg.norm(pr.pos - self.pos)
            if d < dmin:
                dmin = d
                nearest_prey = pr
        if nearest_prey and dmin < self.genes.view_radius:
            self.is_chasing = True
            self.steer_to(nearest_prey.pos, intensity=0.8 + 0.6*self.genes.aggressiveness)
            self.energy -= hunt_cost
        else:
            self.vel += jitter(0.15)
            self.energy -= idle_cost

        speed = np.linalg.norm(self.vel)
        self.energy -= move_cost * (speed / max(1e-6, self.genes.max_speed))

    def eat(self, preys: List[Prey]):
        eaten: List[Prey] = []
        for idx in range(len(preys) - 1, -1, -1):
            p = preys[idx]
            if np.linalg.norm(p.pos - self.pos) < self.RADIUS + p.RADIUS - 1:
                eaten.append(preys.pop(idx))
                # Un seul repas par tick suffit généralement
                break
        if eaten:
            # Remet la barre de faim (énergie) à 100 plutôt que d'accumuler
            self.energy = 100.0
            self.max_energy = max(self.max_energy, self.energy)
        return eaten

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
        self.preys_eaten = 0

        # Sprites
        self.spr_prey = load_sprite(SPRITE_PREY_PATH, scale=0.2)
        self.spr_pred = load_sprite(SPRITE_PRED_PATH, scale=0.3)
        self.spr_food = load_sprite(SPRITE_FOOD_PATH, scale=0.1)
        # Background de la scène
        self.bg_img = None
        _bg_raw = load_sprite(SPRITE_BG_PATH, scale=1.0)
        if _bg_raw is not None:
            try:
                self.bg_img = pygame.transform.smoothscale(_bg_raw, (self.scene_w, self.H))
            except Exception:
                self.bg_img = None

        # Stats historiques (pour mini-graphes)
        self.hist_len = 300
        self.hist_prey = []
        self.hist_pred = []
        self.hist_mean_prey_speed = []
        self.hist_mean_pred_speed = []
        self.hist_mean_prey_view = []
        self.hist_mean_pred_view = []
        self.hist_mean_prey_agg = []
        self.hist_mean_pred_agg = []
        self.prey_life_samples: List[int] = []
        self.hist_life_expectancy = []
        self.hist_kill_rate = []
        self.hist_birth_rate = []
        self.hist_dispersion = []
        self.hist_pursuit = []
        self.hist_energy_ratio = []
        self.hist_stress = []
        self.hist_dom_prey = []
        self.hist_dom_pred = []
        self.hist_equilibrium = []

        self.last_life_expectancy = 0.0
        self.last_kills = 0
        self.last_births = 0
        self.last_dispersion = (0.0, 0.0)
        self.last_pursuit_ratio = 0.0
        self.last_energy_ratio = 0.0
        self.last_stress = 0.0
        self.avg_pred_hunger = 0.0
        self.avg_prey_hunger = 0.0

    def populate(self):
        random.seed(self.cfg["seed"])
        np.random.seed(self.cfg["seed"])
        self.preys = [Prey.rand(self.scene_w, self.H) for _ in range(self.cfg["init_prey"])]
        self.predators = [Predator.rand(self.scene_w, self.H) for _ in range(self.cfg["init_pred"])]
        self.foods = []
        for _ in range(self.cfg["init_food"]):
            pos = self._random_food_position()
            if pos is not None:
                self.foods.append(Food(pos, self.cfg["food_energy"]))

    def reset(self):
        self.__init__(self.cfg)
        self.populate()

    def spawn_food(self, n=1, pos=None):
        if pos is None:
            for _ in range(n):
                new_pos = self._random_food_position()
                if new_pos is not None:
                    self.foods.append(Food(new_pos, self.cfg["food_energy"]))
        else:
            for _ in range(n):
                offset = np.array(pos, dtype=float) + jitter(20)
                offset[0] = clamp(offset[0], 0, self.scene_w - 1)
                offset[1] = clamp(offset[1], 0, self.H - 1)
                self.foods.append(Food(offset, self.cfg["food_energy"]))

    def _random_food_position(self):
        return np.array([rand_range(0, self.scene_w), rand_range(0, self.H)], dtype=float)

    def step(self):
        if self.paused:
            return
        self.tick += 1

        kills_this_tick = 0
        births_this_tick = 0
        chasing_count = 0
        energy_spent = 0.0
        energy_gained = 0.0

        if random.random() < self.cfg["food_spawn_rate"]:
            self.spawn_food(n=random.randint(1, 2))

        # comportements
        prey_start_energy = {id(pr): pr.energy for pr in self.preys}
        for pr in self.preys:
            pr.think(self.foods, self.predators,
                     self.cfg["move_cost"], self.cfg["idle_cost"], self.cfg["hunt_cost"])
        pred_start_energy = {id(pd): pd.energy for pd in self.predators}
        for pd in self.predators:
            pd.think(self.preys,
                     self.cfg["move_cost"], self.cfg["idle_cost"], self.cfg["hunt_cost"])
            if pd.is_chasing:
                chasing_count += 1

        # interactions + vie/mort
        for pr in self.preys:
            pr.step_base()
            pr.pos[0] = pr.pos[0] % self.scene_w  # wrap limité à la scène
            pr.eat(self.foods)
            if pr.energy <= 0:
                pr.energy = 0.0
                pr.alive = False
        for pd in self.predators:
            pd.step_base()
            pd.pos[0] = pd.pos[0] % self.scene_w
            eaten = pd.eat(self.preys)
            if eaten:
                kills_this_tick += len(eaten)
                self.preys_eaten += len(eaten)
                for prey in eaten:
                    self.prey_life_samples.append(prey.age)
                    energy_spent += prey_start_energy.get(id(prey), prey.energy)
            if pd.energy <= 0:
                pd.energy = 0.0
                pd.alive = False

        self.preys = [p for p in self.preys if p.alive]
        self.predators = [p for p in self.predators if p.alive]

        # énergie gagnée/perdue suite aux comportements (hors reproduction)
        for pr in self.preys:
            start = prey_start_energy.get(id(pr), pr.energy)
            delta = pr.energy - start
            if delta > 0:
                energy_gained += delta
            else:
                energy_spent += -delta
        for pd in self.predators:
            start = pred_start_energy.get(id(pd), pd.energy)
            delta = pd.energy - start
            if delta > 0:
                energy_gained += delta
            else:
                energy_spent += -delta

        # reproductions
        newborns_p = []
        repro_energy_transfer = 0.0
        for p in self.preys:
            b = p.reproduce(self.cfg["repro_energy_prey"], self.cfg["mut_rate"], self.cfg["mut_scale"])
            if b:
                newborns_p.append(b)
                births_this_tick += 1
                repro_energy_transfer += b.energy
        if repro_energy_transfer > 0:
            energy_spent += repro_energy_transfer
            energy_gained += repro_energy_transfer
        self.preys.extend(newborns_p)

        # Pas de reproduction de prédateurs pour éviter le dédoublement après repas

        # met à jour historiques
        self.hist_prey.append(len(self.preys))
        self.hist_pred.append(len(self.predators))
        self.hist_mean_prey_speed.append(
            float(np.mean([p.genes.max_speed for p in self.preys])) if self.preys else 0.0
        )
        self.hist_mean_pred_speed.append(
            float(np.mean([p.genes.max_speed for p in self.predators])) if self.predators else 0.0
        )
        self.hist_mean_prey_view.append(
            float(np.mean([p.genes.view_radius for p in self.preys])) if self.preys else 0.0
        )
        self.hist_mean_pred_view.append(
            float(np.mean([p.genes.view_radius for p in self.predators])) if self.predators else 0.0
        )
        self.hist_mean_prey_agg.append(
            float(np.mean([p.genes.aggressiveness for p in self.preys])) if self.preys else 0.0
        )
        self.hist_mean_pred_agg.append(
            float(np.mean([p.genes.aggressiveness for p in self.predators])) if self.predators else 0.0
        )

        avg_life = float(np.mean(self.prey_life_samples[-40:])) if self.prey_life_samples else 0.0
        self.last_life_expectancy = avg_life
        self.hist_life_expectancy.append(avg_life)
        self.last_kills = kills_this_tick
        self.hist_kill_rate.append(kills_this_tick)
        self.last_births = births_this_tick
        self.hist_birth_rate.append(births_this_tick)

        disp_prey = self.compute_dispersion(self.preys)
        disp_pred = self.compute_dispersion(self.predators)
        self.last_dispersion = (disp_prey, disp_pred)
        self.hist_dispersion.append((disp_prey, disp_pred))

        pursuit_ratio = chasing_count / len(self.predators) if self.predators else 0.0
        self.last_pursuit_ratio = pursuit_ratio
        self.hist_pursuit.append(pursuit_ratio)

        energy_ratio = energy_gained / (energy_spent + 1e-6)
        self.last_energy_ratio = energy_ratio
        self.hist_energy_ratio.append(energy_ratio)

        avg_aggr = float(np.mean([p.genes.aggressiveness for p in self.predators])) if self.predators else 0.0
        stress = (len(self.predators) * avg_aggr) / max(1, len(self.preys))
        self.last_stress = stress
        self.hist_stress.append(stress)

        dom_pred = len(self.predators) / max(1, len(self.preys))
        dom_prey = len(self.preys) / max(1, len(self.predators)) if self.predators else float(len(self.preys))
        self.hist_dom_pred.append(dom_pred)
        self.hist_dom_prey.append(dom_prey)

        ratio = len(self.predators) / max(1, len(self.preys))
        self.hist_equilibrium.append(ratio)

        self.avg_pred_hunger = self.compute_hunger(self.predators)
        self.avg_prey_hunger = self.compute_hunger(self.preys)

        # garder taille raisonnable
        tracked_arrays = [
            self.hist_prey, self.hist_pred,
            self.hist_mean_prey_speed, self.hist_mean_pred_speed,
            self.hist_mean_prey_view, self.hist_mean_pred_view,
            self.hist_mean_prey_agg, self.hist_mean_pred_agg,
            self.hist_life_expectancy, self.hist_kill_rate,
            self.hist_birth_rate, self.hist_pursuit,
            self.hist_energy_ratio, self.hist_stress,
            self.hist_dom_prey, self.hist_dom_pred,
            self.hist_equilibrium
        ]
        for arr in tracked_arrays:
            if len(arr) > self.hist_len:
                del arr[0]
        if len(self.hist_dispersion) > self.hist_len:
            del self.hist_dispersion[0]
        if len(self.prey_life_samples) > self.hist_len:
            del self.prey_life_samples[0:len(self.prey_life_samples) - self.hist_len]

    def compute_dispersion(self, entities):
        if len(entities) < 2:
            return 0.0
        positions = np.array([ent.pos for ent in entities], dtype=float)
        diffs = positions[:, None, :] - positions[None, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        n = len(entities)
        # exclure diagonale
        total = np.sum(dists) - np.trace(dists)
        count = n * (n - 1)
        mean_dist = total / count
        max_dist = math.sqrt(self.scene_w ** 2 + self.H ** 2)
        return float(clamp(mean_dist / max_dist if max_dist else 0.0, 0.0, 1.0))

    def compute_hunger(self, entities):
        if not entities:
            return 0.0
        hunger_vals = []
        for ent in entities:
            if ent.max_energy <= 0:
                hunger_vals.append(1.0)
            else:
                ratio = clamp(ent.energy / ent.max_energy, 0.0, 1.0)
                hunger_vals.append(1.0 - ratio)
        return float(np.mean(hunger_vals)) if hunger_vals else 0.0

    def draw(self, surf: pygame.Surface, fonts):
        # zone scène
        scene_rect = pygame.Rect(0, 0, self.scene_w, self.H)
        sidebar_rect = pygame.Rect(self.scene_w, 0, self.sidebar, self.H)
        if self.bg_img is not None:
            # dessine le fond image sur la zone scène
            surf.blit(self.bg_img, (0, 0))
        else:
            surf.fill((242, 246, 255), scene_rect)
        surf.fill((250, 250, 252), sidebar_rect)

        # nourriture
        for f in self.foods:
            f.draw(surf, self.spr_food)
        # proies
        for p in self.preys:
            p.draw(surf, self.spr_prey)
            self.draw_hunger_bar(surf, p, (76, 129, 235), p.RADIUS)
        # prédateurs
        for p in self.predators:
            p.draw(surf, self.spr_pred)
            self.draw_hunger_bar(surf, p, (230, 91, 91), p.RADIUS)

        # séparateur
        pygame.draw.line(surf, (210, 215, 230), (self.scene_w, 0), (self.scene_w, self.H), 2)

        # HUD dans la sidebar
        self.draw_sidebar(surf, fonts, sidebar_rect)

        # info hover
        self.draw_hover_info(surf, fonts)

    def draw_hunger_bar(self, surf: pygame.Surface, creature: Creature, color: Tuple[int, int, int], radius: float):
        if creature.max_energy <= 0:
            return
        ratio = clamp(creature.energy / creature.max_energy, 0.0, 1.0)
        bar_w = 30
        bar_h = 5
        x = creature.pos[0] - bar_w / 2
        y = creature.pos[1] - radius - 10
        rect_bg = pygame.Rect(int(x), int(y), bar_w, bar_h)
        pygame.draw.rect(surf, (35, 40, 55), rect_bg)
        inner_w = int((bar_w - 2) * ratio)
        if inner_w > 0:
            rect_fg = pygame.Rect(int(x) + 1, int(y) + 1, inner_w, bar_h - 2)
            pygame.draw.rect(surf, color, rect_fg)

    def draw_hover_info(self, surf: pygame.Surface, fonts):
        mx, my = pygame.mouse.get_pos()
        if mx >= self.scene_w:
            return

        mouse_vec = np.array([mx, my], dtype=float)
        hovered = None
        min_dist = 1e9
        for pred in self.predators:
            d = np.linalg.norm(pred.pos - mouse_vec)
            if d <= pred.RADIUS + 6 and d < min_dist:
                hovered = ("Prédateur", pred, (230, 91, 91))
                min_dist = d
        for prey in self.preys:
            d = np.linalg.norm(prey.pos - mouse_vec)
            if d <= prey.RADIUS + 6 and d < min_dist:
                hovered = ("Proie", prey, (76, 129, 235))
                min_dist = d

        if not hovered:
            return

        label, entity, color = hovered
        hunger_ratio = clamp(entity.energy / entity.max_energy if entity.max_energy else 0.0, 0.0, 1.0)
        percent = int(hunger_ratio * 100)
        lines = [
            f"{label}",
            f"Âge : {entity.age}",
            f"Faim : {percent}%"
        ]
        font = fonts["tiny"]
        rendered = [font.render(text, True, (240, 240, 245)) for text in lines]
        width = max(r.get_width() for r in rendered) + 12
        height = sum(r.get_height() for r in rendered) + 10
        tooltip = pygame.Surface((width, height), pygame.SRCALPHA)
        tooltip.fill((20, 24, 35, 220))
        y = 6
        for surf_text in rendered:
            tooltip.blit(surf_text, (6, y))
            y += surf_text.get_height()

        px = int(entity.pos[0] + entity.RADIUS + 12)
        py = int(entity.pos[1] - height / 2)
        px = clamp(px, 4, self.scene_w - width - 4)
        py = clamp(py, 4, self.H - height - 4)
        pygame.draw.rect(tooltip, color + (255,), tooltip.get_rect(), width=1, border_radius=6)
        surf.blit(tooltip, (px, py))

    def draw_sidebar(self, surf: pygame.Surface, fonts, rect: pygame.Rect):
        pad = 12
        x = rect.x + pad
        y = rect.y + pad
        w = rect.width - 2 * pad

        title = fonts["bold"].render("Tableau écologique", True, (25, 30, 45))
        surf.blit(title, (x, y))
        y += 26

        def line(text, color=(60, 65, 80)):
            nonlocal y
            surf.blit(fonts["mono"].render(text, True, color), (x, y))
            y += 18

        line(f"Tick: {self.tick}")
        line(f"Proies: {len(self.preys)}  |  Mangées: {self.preys_eaten}")
        line(f"Prédateurs: {len(self.predators)}")
        y += 4

        # 1. Espérance de vie moyenne
        y = self.draw_mini_plot(surf, x, y, w, 60,
                                [self.hist_life_expectancy],
                                [(120, 165, 255)],
                                label_left="espérance vie")

        # 2. Histogramme des âges
        prey_ages = [p.age for p in self.preys]
        pred_ages = [p.age for p in self.predators]
        y = self.draw_histogram(surf, x, y, w, 70,
                                [prey_ages, pred_ages],
                                [(76, 129, 235), (230, 91, 91)],
                                bins=10,
                                label="âges")

        # 3. Fitness génétique
        y = self.draw_mini_plot(surf, x, y, w, 60,
                                [self.hist_mean_prey_speed, self.hist_mean_prey_view, self.hist_mean_prey_agg],
                                [(90, 150, 255), (140, 190, 255), (190, 220, 255)],
                                label_left="gènes proies")
        y = self.draw_mini_plot(surf, x, y, w, 60,
                                [self.hist_mean_pred_speed, self.hist_mean_pred_view, self.hist_mean_pred_agg],
                                [(255, 110, 110), (255, 160, 120), (240, 200, 140)],
                                label_left="gènes prédateurs")

        # 4. Taux de prédation
        y = self.draw_predation_bar(surf, x, y, w, 16)

        # 5. Densité spatiale
        y = self.draw_density_heatmap(surf, x, y, min(120, w))

        # 6. Stress écologique
        y = self.draw_stress_gauge(surf, x, y, w, 14)

        # 7. Indice de famine
        y = self.draw_famine_gauges(surf, x, y, w, 12)

        # 8. Taux de reproduction
        y = self.draw_mini_plot(surf, x, y, w, 60,
                                [self.hist_birth_rate],
                                [(120, 200, 150)],
                                label_left="naissances")

        # 9. Dispersion
        y = self.draw_dispersion_bar(surf, x, y, w, 14)

        # 10. Indice de poursuite
        y = self.draw_pursuit_gauge(surf, x, y, w, 40)

        # 11. Bilan énergétique
        y = self.draw_energy_balance(surf, x, y, w, 18)

        # 12. Carte génétique (radar)
        y = self.draw_genetic_radar(surf, x, y, min(110, w))

        # 13-14. Dominance et équilibre
        y = self.draw_dominance_section(surf, x, y, w, 16)

        y += 4
        help1 = fonts["tiny"].render("SPACE: Pause  |  R: Reset", True, (100, 105, 120))
        surf.blit(help1, (x, y)); y += 16
        help2 = fonts["tiny"].render("L-Clic: Food  |  R-Clic: Predator", True, (100, 105, 120))
        surf.blit(help2, (x, y)); y += 16
        help3 = fonts["tiny"].render("Molette: +/- Proie (jeu)", True, (100, 105, 120))
        surf.blit(help3, (x, y)); y += 16

    def draw_mini_plot(self, surf, x, y, w, h, series_list, colors, label_left=""):
        rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(surf, (255, 255, 255), rect, border_radius=8)
        pygame.draw.rect(surf, (225, 230, 240), rect, width=1, border_radius=8)

        for vals, color in zip(series_list, colors):
            if len(vals) < 2:
                continue
            v = np.array(vals, dtype=float)
            if np.allclose(v, v[0]):
                v = v + np.linspace(0, 0.01, len(v))
            v = (v - v.min()) / (v.max() - v.min() + 1e-8)
            pts = []
            for i in range(len(v)):
                px = x + 6 + (i / max(1, len(v) - 1)) * (w - 12)
                py = y + h - 6 - v[i] * (h - 12)
                pts.append((px, py))
            if len(pts) >= 2:
                pygame.draw.lines(surf, color, False, pts, 2)

        if label_left:
            lbl = pygame.font.SysFont("consolas", 14).render(label_left, True, (120, 125, 140))
            surf.blit(lbl, (x + 8, y + 6))
        return y + h + 8

    def draw_histogram(self, surf, x, y, w, h, data_sets, colors, bins=10, label=""):
        rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(surf, (255, 255, 255), rect, border_radius=8)
        pygame.draw.rect(surf, (225, 230, 240), rect, width=1, border_radius=8)

        max_age = max((max(data) if data else 0) for data in data_sets)
        max_age = max(max_age, 1)
        hist_list = []
        for data in data_sets:
            if data:
                hist, _ = np.histogram(data, bins=bins, range=(0, max_age))
            else:
                hist = np.zeros(bins, dtype=int)
            hist_list.append(hist)
        max_count = max((hist.max() for hist in hist_list), default=1)
        max_count = max(max_count, 1)
        group_w = (w - 12) / bins
        bar_w = group_w / max(1, len(hist_list))
        for i in range(bins):
            base_x = x + 6 + i * group_w
            for j, hist in enumerate(hist_list):
                value = hist[i] / max_count
                bar_h = value * (h - 20)
                bx = base_x + j * bar_w + 2
                by = y + h - 8 - bar_h
                pygame.draw.rect(surf, colors[j], pygame.Rect(bx, by, bar_w - 4, bar_h), border_radius=3)

        if label:
            lbl = pygame.font.SysFont("consolas", 14).render(label, True, (120, 125, 140))
            surf.blit(lbl, (x + 8, y + 6))
        return y + h + 8

    def draw_predation_bar(self, surf, x, y, w, h):
        recent = float(np.mean(self.hist_kill_rate[-20:])) if self.hist_kill_rate else float(self.last_kills)
        norm = clamp(recent / max(1, len(self.preys) + len(self.predators)), 0.0, 1.0)
        bg = pygame.Rect(x, y, w, h)
        pygame.draw.rect(surf, (225, 230, 240), bg, border_radius=6)
        color = (
            int(60 + 160 * norm),
            int(200 - 120 * norm),
            int(80 - 30 * norm)
        )
        fg = pygame.Rect(x + 2, y + 2, int((w - 4) * norm), h - 4)
        pygame.draw.rect(surf, color, fg, border_radius=6)
        label = pygame.font.SysFont("consolas", 14).render("taux prédation", True, (90, 95, 110))
        surf.blit(label, (x + 6, y - 18))
        return y + h + 12

    def draw_density_heatmap(self, surf, x, y, size, grid=40):
        cell_w = max(1.0, self.scene_w / grid)
        cell_h = max(1.0, self.H / grid)
        counts = np.zeros((grid, grid, 2), dtype=float)
        for pr in self.preys:
            gx = int(pr.pos[0] / cell_w) % grid
            gy = int(pr.pos[1] / cell_h) % grid
            counts[gy, gx, 0] += 1
        for pd in self.predators:
            gx = int(pd.pos[0] / cell_w) % grid
            gy = int(pd.pos[1] / cell_h) % grid
            counts[gy, gx, 1] += 1
        prey_max = counts[:, :, 0].max() or 1.0
        pred_max = counts[:, :, 1].max() or 1.0
        heat = pygame.Surface((grid, grid))
        for gy in range(grid):
            for gx in range(grid):
                pb = counts[gy, gx, 0] / prey_max
                pr = counts[gy, gx, 1] / pred_max
                r = int(clamp(60 + 180 * pr, 0, 255))
                g = int(clamp(70 + 140 * pb, 0, 255))
                b = int(clamp(90 + 50 * (1 - (pb + pr) / 2), 0, 255))
                heat.set_at((gx, gy), (r, g, b))
        scaled = pygame.transform.smoothscale(heat, (size, size))
        surf.blit(scaled, (x, y))
        pygame.draw.rect(surf, (40, 45, 60), pygame.Rect(x, y, size, size), width=2, border_radius=6)
        label = pygame.font.SysFont("consolas", 14).render("densité", True, (90, 95, 110))
        surf.blit(label, (x, y - 20))
        return y + size + 10

    def draw_stress_gauge(self, surf, x, y, w, h):
        value = clamp(self.last_stress / 3.0, 0.0, 1.0)
        bg = pygame.Rect(x, y, w, h)
        pygame.draw.rect(surf, (225, 230, 240), bg, border_radius=6)
        color = (
            int(80 + 140 * value),
            int(220 - 160 * value),
            int(90)
        )
        pygame.draw.rect(surf, color, pygame.Rect(x + 2, y + 2, int((w - 4) * value), h - 4), border_radius=6)
        lbl = pygame.font.SysFont("consolas", 14).render("stress eco", True, (90, 95, 110))
        surf.blit(lbl, (x + 6, y - 18))
        return y + h + 12

    def draw_famine_gauges(self, surf, x, y, w, h):
        labels = [("faim prédateurs", self.avg_pred_hunger, (230, 91, 91)),
                  ("faim proies", self.avg_prey_hunger, (76, 129, 235))]
        for name, value, color in labels:
            bg = pygame.Rect(x, y, w, h)
            pygame.draw.rect(surf, (225, 230, 240), bg, border_radius=6)
            pygame.draw.rect(surf, color, pygame.Rect(x + 2, y + 2, int((w - 4) * clamp(value, 0.0, 1.0)), h - 4), border_radius=6)
            lbl = pygame.font.SysFont("consolas", 13).render(name, True, (90, 95, 110))
            surf.blit(lbl, (x + 6, y - 16))
            y += h + 16
        return y

    def draw_dispersion_bar(self, surf, x, y, w, h):
        for label, value, color in (("dispersion proies", self.last_dispersion[0], (76, 129, 235)),
                                    ("dispersion prédateurs", self.last_dispersion[1], (230, 91, 91))):
            bg = pygame.Rect(x, y, w, h)
            pygame.draw.rect(surf, (225, 230, 240), bg, border_radius=6)
            pygame.draw.rect(surf, color, pygame.Rect(x + 2, y + 2, int((w - 4) * clamp(value, 0.0, 1.0)), h - 4), border_radius=6)
            lbl = pygame.font.SysFont("consolas", 13).render(label, True, (90, 95, 110))
            surf.blit(lbl, (x + 6, y - 16))
            y += h + 16
        return y

    def draw_pursuit_gauge(self, surf, x, y, w, size):
        radius = size // 2
        center = (x + radius, y + radius)
        pygame.draw.circle(surf, (230, 234, 244), center, radius)
        pygame.draw.circle(surf, (250, 250, 252), center, radius - 8)
        ratio = clamp(self.last_pursuit_ratio, 0.0, 1.0)
        rect = pygame.Rect(center[0] - radius + 4, center[1] - radius + 4, 2 * (radius - 4), 2 * (radius - 4))
        pygame.draw.arc(surf, (230, 91, 91), rect, -math.pi / 2, -math.pi / 2 + ratio * 2 * math.pi, width=6)
        pygame.draw.circle(surf, (40, 45, 60), center, radius, width=2)
        lbl = pygame.font.SysFont("consolas", 14).render("poursuite", True, (90, 95, 110))
        surf.blit(lbl, (x + size + 8, y + radius - 10))
        value_lbl = pygame.font.SysFont("consolas", 14).render(f"{int(ratio * 100)}%", True, (60, 65, 80))
        surf.blit(value_lbl, (x + radius - value_lbl.get_width() // 2, y + radius - value_lbl.get_height() // 2))
        return y + size + 8

    def draw_energy_balance(self, surf, x, y, w, h):
        ratio = self.last_energy_ratio
        bg = pygame.Rect(x, y, w, h)
        pygame.draw.rect(surf, (225, 230, 240), bg, border_radius=6)
        mid = x + w // 2
        pygame.draw.line(surf, (180, 185, 200), (mid, y), (mid, y + h), width=2)
        if ratio >= 1.0:
            value = clamp((ratio - 1.0) / 2.0, 0.0, 1.0)
            width = int((w // 2 - 2) * value)
            pygame.draw.rect(surf, (90, 190, 120), pygame.Rect(mid + 2, y + 2, width, h - 4))
        else:
            value = clamp(1.0 - ratio, 0.0, 1.0)
            width = int((w // 2 - 2) * value)
            pygame.draw.rect(surf, (225, 120, 90), pygame.Rect(mid - width, y + 2, width, h - 4))
        lbl = pygame.font.SysFont("consolas", 14).render("bilan énergie", True, (90, 95, 110))
        surf.blit(lbl, (x + 6, y - 18))
        return y + h + 12

    def draw_genetic_radar(self, surf, x, y, size):
        radius = size // 2 - 6
        center = (x + size // 2, y + size // 2)
        background = pygame.Surface((size, size), pygame.SRCALPHA)
        pygame.draw.circle(background, (240, 244, 252, 220), (size // 2, size // 2), radius + 4)
        pygame.draw.circle(background, (210, 215, 230, 255), (size // 2, size // 2), radius + 4, width=2)

        def polygon(values, color):
            pts = []
            for i, val in enumerate(values):
                angle = -math.pi / 2 + i * (2 * math.pi / len(values))
                r = radius * clamp(val, 0.0, 1.0)
                px = size // 2 + r * math.cos(angle)
                py = size // 2 + r * math.sin(angle)
                pts.append((px, py))
            if len(pts) >= 3:
                pygame.draw.polygon(background, color, pts)
                pygame.draw.polygon(background, (30, 35, 45), pts, width=1)

        for i in range(3):
            angle = -math.pi / 2 + i * (2 * math.pi / 3)
            px = size // 2 + radius * math.cos(angle)
            py = size // 2 + radius * math.sin(angle)
            pygame.draw.line(background, (180, 185, 200), (size // 2, size // 2), (px, py), width=1)

        prey_vals = [
            self._normalize_gene(np.mean([p.genes.max_speed for p in self.preys]) if self.preys else 0.0, Prey.BASE_SPEED),
            self._normalize_gene(np.mean([p.genes.view_radius for p in self.preys]) if self.preys else 0.0, Prey.VIEW),
            self._normalize_gene(np.mean([p.genes.aggressiveness for p in self.preys]) if self.preys else 0.0, (0.0, 1.0))
        ]
        pred_vals = [
            self._normalize_gene(np.mean([p.genes.max_speed for p in self.predators]) if self.predators else 0.0, Predator.BASE_SPEED),
            self._normalize_gene(np.mean([p.genes.view_radius for p in self.predators]) if self.predators else 0.0, Predator.VIEW),
            self._normalize_gene(np.mean([p.genes.aggressiveness for p in self.predators]) if self.predators else 0.0, (0.0, 1.0))
        ]

        polygon(prey_vals, (80, 140, 235, 90))
        polygon(pred_vals, (230, 91, 91, 90))
        surf.blit(background, (x, y))
        label = pygame.font.SysFont("consolas", 14).render("radar génétique", True, (90, 95, 110))
        surf.blit(label, (x, y - 20))
        return y + size + 12

    def draw_dominance_section(self, surf, x, y, w, h):
        dom_pred = self.hist_dom_pred[-1] if self.hist_dom_pred else 0.0
        dom_prey = self.hist_dom_prey[-1] if self.hist_dom_prey else 0.0
        ratio = self.hist_equilibrium[-1] if self.hist_equilibrium else 0.0
        ideal = 0.25
        deviation = abs(ratio - ideal) / (ideal + 1e-6)
        deviation = clamp(deviation, 0.0, 1.0)

        dom_text = pygame.font.SysFont("consolas", 14).render(
            f"dominance P:{dom_pred:.2f} / p:{dom_prey:.2f}", True, (60, 65, 80))
        surf.blit(dom_text, (x, y))
        y += h

        bg = pygame.Rect(x, y, w, h)
        pygame.draw.rect(surf, (225, 230, 240), bg, border_radius=6)
        balance = clamp(1.0 - deviation, 0.0, 1.0)
        color = (
            int(90 + 120 * (1 - balance)),
            int(180 + 40 * balance),
            int(90)
        )
        pygame.draw.rect(surf, color, pygame.Rect(x + 2, y + 2, int((w - 4) * balance), h - 4), border_radius=6)
        lbl = pygame.font.SysFont("consolas", 14).render("équilibre", True, (90, 95, 110))
        surf.blit(lbl, (x + 6, y - 18))
        return y + h + 16

    def _normalize_gene(self, value, bounds):
        lo, hi = bounds
        return 0.0 if hi - lo == 0 else clamp((value - lo) / (hi - lo), 0.0, 1.0)

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
                    if event.button == 1:  # left
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
