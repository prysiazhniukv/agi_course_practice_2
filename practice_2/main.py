import numpy as np
from pathlib import Path
from simulation import Simulation

np.random.seed(42)

if __name__ == "__main__":
    pattern_file = Path(__file__).parent / 'patterns' / 'arrow.json'
    Simulation(warmup=1000, pattern_file=str(pattern_file)).run()
