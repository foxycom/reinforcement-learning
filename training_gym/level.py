from enum import Enum
from os.path import join, dirname


class Level(Enum):

    GRASS = 0
    PURE = 1

    def vae(self):
        if self is Level.GRASS:
            return join(dirname(__file__), "..", "models", "lines-vae.pkl")
        elif self is Level.PURE:
            return join(dirname(__file__), "..", "models", "smallgrid-lines-vae.pkl")

    def model(self):
        from os.path import join, dirname
        if self is Level.GRASS:
            return join(dirname(__file__), "..", "models", "lines-agent.pkl")
        elif self is Level.PURE:
            return join(dirname(__file__), "..", "models", "smallgrid-lines-agent.pkl")