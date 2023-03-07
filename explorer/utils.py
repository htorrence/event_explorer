import time
from dataclasses import dataclass, field
import plotly.express as px


class Timer:
    def __init__(self):
        self.start = time.time()

    def timecheck(self):
        timediff = time.time() - self.start
        print(f'{int(timediff)} seconds elapsed')

@dataclass
class VizOptions:
    color_scale: list[str] = field(default_factory=lambda: px.colors.qualitative.Plotly)
    default_marker_size: int = 5