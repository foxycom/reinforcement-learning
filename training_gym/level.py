from enum import Enum


class Level(Enum):

    GRASS = 0
    PURE = 1

    def vae(self):
        if self is Level.GRASS:
            return "models\\lines-vae.pkl"
        elif self is Level.PURE:
            return "models\\smallgrid-lines-vae.pkl"

    def model(self):
        if self is Level.GRASS:
            return "models\\lines-agent.pkl"
        elif self is Level.PURE:
            return "models\\smallgrid-lines-agent.pkl"