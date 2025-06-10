"""
Base agent class for the multi-agent datacenter simulation.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import numpy as np
import logging
from enum import Enum

from ..core.config import get_config


class AgentType(Enum):
    """Types of agents in the simulation."""
    HYPERSCALER = "hyperscaler"
    NATION = "nation"
    SEMICONDUCTOR = "semiconductor"
    ENERGY_MARKET = "energy_market"
    REGULATOR = "regulator"


@dataclass
class AgentState:
    """Base state for all agents."""
    agent_id: str
    agent_type: AgentType
    current_time: float = 0.0
    active: bool = True
    cash_balance: float = 0.0
    reputation_score: float = 1.0
    last_action: Optional[str] = None
    decision_history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class MarketConditions:
    """Current market conditions visible to agents."""
    datacenter_demand_growth: float
    semiconductor_supply_shortage: float
    power_availability_constraint: float
    geopolitical_tension_level: float
    average_construction_costs: float
    average_power_costs: float
    carbon_pricing: float
    exchange_rates: Dict[str, float]
    trade_restrictions: Dict[str, List[str]]


class BaseAgent(ABC):
    """Base class for all agents in the simulation."""
    
    def __init__(self, agent_id: str, agent_type: AgentType, initial_params: Dict[str, Any]):
        """Initialize base agent."""
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.config = get_config()
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{agent_id}")
        
        # Initialize state
        self.state = AgentState(
            agent_id=agent_id,
            agent_type=agent_type,
            cash_balance=initial_params.get('initial_cash', 0.0),
            reputation_score=initial_params.get('initial_reputation', 1.0)
        )
        
        # Agent-specific parameters
        self.parameters = initial_params.copy()
        
        # Performance tracking
        self.performance_metrics = {
            'total_revenue': 0.0,
            'total_costs': 0.0,
            'capacity_utilization': 0.0,
            'market_share': 0.0,
            'risk_exposure': 0.0
        }
        
        # Decision-making components
        self.risk_tolerance = initial_params.get('risk_tolerance', 0.5)
        self.planning_horizon = initial_params.get('planning_horizon', 24)  # months
        self.learning_rate = initial_params.get('learning_rate', 0.1)
        
        # Market information
        self.market_knowledge = {}
        self.competitor_tracking = {}
        
    @abstractmethod
    def perceive(self, market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Perceive current market conditions."""
        pass
    
    @abstractmethod
    def decide(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """Make decisions based on current perception."""
        pass
    
    @abstractmethod
    def act(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute decided actions and return results."""
        pass
    
    def update_state(self, time_step: float, results: Dict[str, Any]) -> None:
        """Update agent state based on action results."""
        self.state.current_time = time_step
        self.state.last_action = results.get('action_type')
        
        # Update financial state
        revenue = results.get('revenue', 0.0)
        costs = results.get('costs', 0.0)
        self.state.cash_balance += revenue - costs
        
        # Update performance metrics
        self.performance_metrics['total_revenue'] += revenue
        self.performance_metrics['total_costs'] += costs
        
        # Update reputation based on performance
        performance_impact = results.get('reputation_change', 0.0)
        self.state.reputation_score = max(0.1, min(2.0, 
            self.state.reputation_score + performance_impact))
        
        # Store decision in history
        decision_record = {
            'time': time_step,
            'action': results.get('action_type'),
            'parameters': results.get('action_parameters', {}),
            'outcome': results.get('outcome', {}),
            'performance': {
                'revenue': revenue,
                'costs': costs,
                'net_profit': revenue - costs
            }
        }
        self.state.decision_history.append(decision_record)
        
        # Learn from results
        self._update_learning(results)
    
    def _update_learning(self, results: Dict[str, Any]) -> None:
        """Update agent's learning based on action results."""
        # Simple learning mechanism - adjust risk tolerance based on outcomes
        outcome_score = results.get('outcome_score', 0.0)  # -1 to 1 scale
        
        if outcome_score > 0.5:  # Good outcome
            self.risk_tolerance = min(1.0, self.risk_tolerance + self.learning_rate * 0.1)
        elif outcome_score < -0.5:  # Bad outcome
            self.risk_tolerance = max(0.1, self.risk_tolerance - self.learning_rate * 0.1)
        
        # Update market knowledge
        market_info = results.get('market_feedback', {})
        for key, value in market_info.items():
            if key in self.market_knowledge:
                # Exponential smoothing
                alpha = 0.3
                self.market_knowledge[key] = (alpha * value + 
                                            (1 - alpha) * self.market_knowledge[key])
            else:
                self.market_knowledge[key] = value
    
    def calculate_utility(self, decision_outcome: Dict[str, Any]) -> float:
        """Calculate utility for a given decision outcome."""
        # Base utility function - can be overridden by specific agents
        profit = decision_outcome.get('profit', 0.0)
        risk = decision_outcome.get('risk', 0.0)
        strategic_value = decision_outcome.get('strategic_value', 0.0)
        
        # Risk-adjusted utility
        utility = profit - (risk * (1 - self.risk_tolerance)) + strategic_value
        
        return utility
    
    def evaluate_scenario(self, scenario: Dict[str, Any]) -> float:
        """Evaluate a scenario and return expected utility."""
        # Simple scenario evaluation - can be made more sophisticated
        expected_profit = scenario.get('expected_profit', 0.0)
        probability = scenario.get('probability', 1.0)
        risk_level = scenario.get('risk_level', 0.0)
        
        risk_adjusted_return = expected_profit - (risk_level * (1 - self.risk_tolerance))
        expected_utility = probability * risk_adjusted_return
        
        return expected_utility
    
    def get_financial_health(self) -> Dict[str, float]:
        """Get current financial health indicators."""
        total_profit = self.performance_metrics['total_revenue'] - self.performance_metrics['total_costs']
        
        return {
            'cash_balance': self.state.cash_balance,
            'total_profit': total_profit,
            'revenue_growth': self._calculate_revenue_growth(),
            'profit_margin': self._calculate_profit_margin(),
            'liquidity_ratio': self._calculate_liquidity_ratio()
        }
    
    def _calculate_revenue_growth(self) -> float:
        """Calculate revenue growth rate."""
        if len(self.state.decision_history) < 2:
            return 0.0
        
        recent_revenue = sum(d['performance']['revenue'] 
                           for d in self.state.decision_history[-6:])  # Last 6 periods
        older_revenue = sum(d['performance']['revenue'] 
                          for d in self.state.decision_history[-12:-6])  # Previous 6 periods
        
        if older_revenue == 0:
            return 0.0
        
        return (recent_revenue - older_revenue) / older_revenue
    
    def _calculate_profit_margin(self) -> float:
        """Calculate profit margin."""
        total_revenue = self.performance_metrics['total_revenue']
        if total_revenue == 0:
            return 0.0
        
        total_profit = total_revenue - self.performance_metrics['total_costs']
        return total_profit / total_revenue
    
    def _calculate_liquidity_ratio(self) -> float:
        """Calculate liquidity ratio (simplified)."""
        # Simplified - in reality would consider current assets vs current liabilities
        return max(0.0, self.state.cash_balance / max(1.0, self.performance_metrics['total_costs'] * 0.1))
    
    def get_strategic_position(self) -> Dict[str, Any]:
        """Get current strategic position metrics."""
        return {
            'market_share': self.performance_metrics.get('market_share', 0.0),
            'capacity_utilization': self.performance_metrics.get('capacity_utilization', 0.0),
            'competitive_advantage': self._calculate_competitive_advantage(),
            'strategic_assets': self._count_strategic_assets(),
            'risk_exposure': self.performance_metrics.get('risk_exposure', 0.0)
        }
    
    def _calculate_competitive_advantage(self) -> float:
        """Calculate competitive advantage score."""
        # Simplified calculation based on reputation and market position
        base_advantage = self.state.reputation_score - 1.0  # Above/below average
        market_position = self.performance_metrics.get('market_share', 0.0)
        
        return base_advantage + market_position
    
    def _count_strategic_assets(self) -> int:
        """Count strategic assets (to be overridden by specific agents)."""
        return 0
    
    def should_cooperate(self, other_agent: 'BaseAgent', proposal: Dict[str, Any]) -> bool:
        """Decide whether to cooperate with another agent."""
        # Simple cooperation logic based on reputation and mutual benefit
        if other_agent.state.reputation_score < 0.5:
            return False  # Don't cooperate with unreliable agents
        
        expected_benefit = proposal.get('expected_benefit', 0.0)
        risk_level = proposal.get('risk_level', 0.0)
        
        cooperation_utility = expected_benefit - (risk_level * (1 - self.risk_tolerance))
        
        return cooperation_utility > 0.0
    
    def negotiate(self, other_agent: 'BaseAgent', initial_proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Negotiate with another agent."""
        # Simple negotiation logic
        acceptable_terms = self._evaluate_proposal(initial_proposal)
        
        if acceptable_terms['acceptable']:
            return {
                'accept': True,
                'counter_proposal': initial_proposal,
                'terms': acceptable_terms['terms']
            }
        else:
            counter_proposal = self._generate_counter_proposal(initial_proposal)
            return {
                'accept': False,
                'counter_proposal': counter_proposal,
                'terms': acceptable_terms['terms']
            }
    
    def _evaluate_proposal(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a cooperation proposal."""
        expected_utility = self.evaluate_scenario(proposal)
        minimum_acceptable_utility = self.parameters.get('min_cooperation_utility', 0.1)
        
        return {
            'acceptable': expected_utility >= minimum_acceptable_utility,
            'utility': expected_utility,
            'terms': proposal.get('terms', {})
        }
    
    def _generate_counter_proposal(self, original_proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a counter-proposal."""
        # Simple counter-proposal logic - adjust key parameters
        counter = original_proposal.copy()
        
        # Adjust financial terms in our favor
        if 'cost_sharing' in counter:
            counter['cost_sharing'] = max(0.3, counter['cost_sharing'] - 0.1)
        
        if 'revenue_sharing' in counter:
            counter['revenue_sharing'] = min(0.7, counter['revenue_sharing'] + 0.1)
        
        return counter
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of current agent state."""
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type.value,
            'active': self.state.active,
            'cash_balance': self.state.cash_balance,
            'reputation_score': self.state.reputation_score,
            'performance_metrics': self.performance_metrics.copy(),
            'risk_tolerance': self.risk_tolerance,
            'market_share': self.performance_metrics.get('market_share', 0.0)
        }
    
    def reset(self) -> None:
        """Reset agent to initial state."""
        initial_cash = self.parameters.get('initial_cash', 0.0)
        initial_reputation = self.parameters.get('initial_reputation', 1.0)
        
        self.state = AgentState(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            cash_balance=initial_cash,
            reputation_score=initial_reputation
        )
        
        self.performance_metrics = {
            'total_revenue': 0.0,
            'total_costs': 0.0,
            'capacity_utilization': 0.0,
            'market_share': 0.0,
            'risk_exposure': 0.0
        }
        
        self.market_knowledge = {}
        self.competitor_tracking = {}
    
    def __str__(self) -> str:
        return f"{self.agent_type.value}_{self.agent_id}"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.agent_id}, type={self.agent_type.value})" 