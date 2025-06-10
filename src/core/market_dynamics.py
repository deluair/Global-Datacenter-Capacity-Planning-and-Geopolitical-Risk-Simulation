"""
Market Dynamics for the Global Datacenter Simulation.

This module defines the market environment where agents interact, including:
- Supply and demand dynamics
- Pricing mechanisms
- Market impact of agent actions
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any

@dataclass
class MarketImpact:
    """Represents the impact of an action on the market."""
    price_change: float = 0.0
    demand_change: float = 0.0
    supply_change: float = 0.0
    reputation_change: float = 0.0

class Market:
    """Represents the global market for datacenter services."""

    def __init__(self, config):
        self.config = config
        self.current_price = 100.0  # $/MWh
        self.total_capacity_mw = 50000
        self.total_demand_mw = 45000
        self.semiconductor_supply_level = 1.0  # 1.0 = balanced
        self.energy_price_index = 1.0
        self.geopolitical_risk_index = 0.1
        self.sustainability_score = 0.5
        self.market_sentiment = 0.5  # 0-1 scale, neutral

    def update_market(self, agent_actions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Update market conditions based on agent actions and dynamics."""
        # Aggregate supply/demand changes
        for action in agent_actions:
            if action['type'] == 'capacity_expansion':
                self.total_capacity_mw += action['amount_mw']
            elif action['type'] == 'sustainability_investment':
                self.sustainability_score += action['impact']

        # Update demand based on growth rate
        demand_growth = self.config.simulation.demand_growth_rate * (1 + 0.2 * self.market_sentiment)
        self.total_demand_mw *= (1 + demand_growth)

        # Update prices based on supply/demand balance
        utilization = self.total_demand_mw / self.total_capacity_mw if self.total_capacity_mw > 0 else 1.0
        price_adjustment = (utilization - 0.85) * 0.2  # Price pressure above 85% utilization
        self.current_price *= (1 + price_adjustment)

        # Update semiconductor supply (exogenous for now)
        self.semiconductor_supply_level += np.random.normal(0, 0.05)
        self.semiconductor_supply_level = max(0.5, min(1.5, self.semiconductor_supply_level))

        market_state = {
            "current_price": self.current_price,
            "total_capacity_mw": self.total_capacity_mw,
            "total_demand_mw": self.total_demand_mw,
            "semiconductor_supply_level": self.semiconductor_supply_level,
            "sustainability_score": self.sustainability_score,
            "geopolitical_risk_index": self.geopolitical_risk_index
        }
        return market_state 