# SenseGraph


[Interactive coverage map HTML  (location: Londrina, Paraná - Brazil)](./sensor_comparison_map.html)


# SenseGraph

*A toolkit for selecting optimal sensor locations in smart cities using reinforcement learning and graph-based coverage models.*

---

## Overview

SenseGraph provides a modular framework to explore and evaluate **sensor placement strategies** in urban environments. It enables researchers and practitioners to:

- Model coverage as a **graph problem**
- Apply and compare different **selection algorithms** (Greedy, Reinforcement Learning, etc.)
- Visualize **coverage maps, rankings, and chosen sensor positions**
- Incorporate new algorithms and domain‐specific **ISAC metrics**

---

## Project Structure

SenseGraph/
│
├── main.py                     → Command-line interface for running experiments
├── analysis.ipynb             → Jupyter notebook for comparing/plotting results
├── requirements.txt
├── README.md
│
├── src/
│   ├── algos/
│   │   ├── greedy/            → Greedy max-coverage algorithm
│   │   └── reinforce/         → Reinforcement-learning method
│   ├── environments/          → Sensor-placement environments + abstractions
│   ├── dataloaders/           → Load candidates/universe from .geojson/.csv
│   └── vis/                   → HTML and map visualizations
│
└── data/
└── example\_city/
├── candidates.geojson
├── universe.geojson
└── ...

## Installation

```bash
git clone https://github.com/mauricionoris/SenseGraph.git
cd SenseGraph

# (recommended) create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
````

---

## Running Experiments

### **1. Greedy Sensor Selection**

```bash
python main.py --algo greedy \
               --candidates data/example_city/candidates.geojson \
               --universe   data/example_city/universe.geojson \
               --budget     10 \
               --output     results/greedy/
```

### **2. Reinforcement-Learning Sensor Selection**

```bash
python main.py --algo reinforce \
               --candidates data/example_city/candidates.geojson \
               --universe   data/example_city/universe.geojson \
               --budget     10 \
               --episodes   500 \
               --output     results/reinforce/
```

---

## Analyze & Visualize Results

After running the experiments, launch the Jupyter notebook to compare the ranking, coverage and selected sensors:

```bash
jupyter notebook analysis.ipynb
```

The notebook generates plots and opens interactive `*.html` maps (leaflet) stored in the output directory.

---

## Extending the Framework

You can easily add new algorithms in `src/algos/` by implementing the standard interface:

```python
class BaseSelector:
    def __init__(...):
        ...
    def select(self, candidates, universe, budget):
        """Returns a list of ids of chosen candidates."""
```

Additional ISAC performance metrics and datasets can also be plugged into:

* `src/environments/`
* `src/vis/`
* `src/dataloaders/`

---

## 🧾 License

This project is licensed under the **MIT License** — see `LICENSE` for details.

