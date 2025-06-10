# Global Datacenter Capacity Planning and Geopolitical Risk Simulation

This project is a comprehensive, multi-agent simulation designed to model the complex and interconnected challenges of global datacenter capacity planning. It addresses the explosive growth in demand driven by AI, constraints in the global semiconductor supply chain, and escalating geopolitical tensions that impact trade, technology, and investment.

The simulation provides a framework for analyzing strategic decisions by key actors (hyperscale cloud providers and nation-states) and understanding the systemic effects of various economic, political, and technological scenarios.

## Key Features

- **Sophisticated Multi-Agent System**:
    - **Hyperscaler Agents**: Models major cloud providers (Google, Amazon, Microsoft, Meta) with realistic multi-billion dollar capex budgets, capacity expansion strategies, and supply chain diversification logic.
    - **Nation Agents**: Models national governments with complex decision-making across 8 policy domains (Trade, Industrial, Energy, Technology, etc.), geopolitical stances, and institutional constraints.

- **Advanced Economic Modeling**:
    - **Welfare Economics Analysis**: End-to-end analysis of consumer/producer surplus, deadweight loss, and distributional impacts of infrastructure investments and policies.
    - **International Trade Analysis**: A gravity-based model to simulate trade flows, comparative advantages, supply chain vulnerabilities (using HHI), and the impact of tariffs and restrictions.

- **Realistic Data Generation**:
    - Procedurally generates 10,000+ potential datacenter sites with realistic attributes (land cost, power availability, political stability, etc.).
    - Models the semiconductor supply chain with major suppliers (TSMC, Samsung, Intel), technology nodes, and geopolitical constraints.
    - Generates correlated time-series data for key economic indicators.

- **Comprehensive Geopolitical & Scenario Engine**:
    - Simulates various scenarios like `trade_war`, `high_growth`, `green_transition`, and `supply_crisis`.
    - Models the impact of trade tariffs, technology export controls, and shifting geopolitical alliances.
    - Allows for detailed policy impact analysis and scenario comparisons.

- **Interactive Visualization**:
    - Includes a web-based dashboard (powered by Streamlit) for real-time parameter adjustment, scenario exploration, and visualization of results.

- **Extensible & Configurable**:
    - Utilizes a Pydantic-based configuration system (`src/core/config.py`) for easy tuning of all simulation parameters.
    - Designed with a modular architecture for extensibility.

## Project Structure

The project is organized into a Python package for clarity and scalability:

```
├── main.py                     # Main simulation runner & CLI
├── requirements.txt            # Project dependencies
├── test_imports.py             # Script to test all module imports
├── src/
│   ├── agents/                 # Agent definitions (Hyperscaler, Nation)
│   │   ├── __init__.py
│   │   ├── base_agent.py
│   │   ├── hyperscaler_agent.py
│   │   └── nation_agent.py
│   ├── core/                   # Core simulation logic
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── market_dynamics.py
│   │   └── simulation_engine.py
│   ├── data/                   # Data generation and management
│   │   ├── __init__.py
│   │   └── generators.py
│   ├── economics/              # Advanced economic models
│   │   ├── __init__.py
│   │   ├── international_trade.py
│   │   └── welfare_analysis.py
│   └── visualization/          # Interactive dashboard code
│       ├── __init__.py
│       └── dashboard.py
└── results/                    # Default output directory for simulation runs
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/deluair/Global-Datacenter-Capacity-Planning-and-Geopolitical-Risk-Simulation.git
    cd Global-Datacenter-Capacity-Planning-and-Geopolitical-Risk-Simulation
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

## Usage

The simulation is controlled via the `main.py` script, which provides a comprehensive command-line interface.

### Generate Simulation Data
Before the first run, you can pre-generate the required datasets.

```bash
python main.py --generate-data
```

### Run a Simulation Scenario
Run a single scenario (e.g., `baseline` for 24 months). Results will be saved to the `output/` directory by default.

```bash
python main.py --scenario baseline --duration 24
```
Available scenarios: `baseline`, `high_growth`, `trade_war`, `green_transition`, `supply_crisis`.

### Run Scenario Comparison
Analyze and compare the outcomes of multiple scenarios.

```bash
python main.py --scenario-analysis baseline trade_war green_transition
```

### Perform Economic Analysis
Run a post-simulation economic analysis on the results of a run.

```bash
python main.py --analysis economic
```

### Launch the Interactive Dashboard
Explore the simulation results and adjust parameters in real-time.

```bash
python main.py --dashboard
```

### Export a Research Package
Package the simulation results, economic analysis, and documentation into a shareable format for academic use.

```bash
python main.py --export-data
```

## Core Concepts

-   **Simulation Engine (`src/core/simulation_engine.py`):** Orchestrates the entire simulation loop, managing time steps, agent interactions, and market updates.
-   **Market Dynamics (`src/core/market_dynamics.py`):** Represents the global market, handling supply/demand calculations, price adjustments, and tracking key indices like semiconductor supply and geopolitical risk.
-   **Configuration (`src/core/config.py`):** A Pydantic-based system that defines all tunable parameters for the simulation, from agent budgets to geopolitical event probabilities. This is the central place to modify the simulation's behavior.

## Dependencies

The project relies on a number of powerful libraries for scientific computing, data analysis, and visualization:

-   `numpy`
-   `pandas`
-   `scipy`
-   `networkx`
-   `pydantic`
-   `streamlit`
-   `plotly`

...and others listed in `requirements.txt`. 