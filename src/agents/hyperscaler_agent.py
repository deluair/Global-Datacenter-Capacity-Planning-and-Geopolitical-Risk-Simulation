"""
Hyperscaler agents representing major cloud providers.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import numpy as np
import random
from scipy.optimize import minimize

from .base_agent import BaseAgent, AgentType
from ..core.config import get_config


@dataclass
class DatacenterAsset:
    """Represents a datacenter owned by a hyperscaler."""
    asset_id: str
    site_id: str
    capacity_mw: float
    utilization_rate: float
    construction_cost: float
    operational_cost_per_month: float
    renewable_energy_share: float
    operational_date: float
    expansion_potential_mw: float


@dataclass
class CapacityPlan:
    """Capacity expansion plan."""
    target_capacity_mw: float
    timeline_months: int
    budget_allocation: float
    geographic_distribution: Dict[str, float]
    risk_assessment: float


class HyperscalerAgent(BaseAgent):
    """Agent representing a hyperscaler (Google, Amazon, Microsoft, Meta)."""
    
    def __init__(self, agent_id: str, company_name: str, initial_params: Dict[str, Any]):
        """Initialize hyperscaler agent."""
        super().__init__(agent_id, AgentType.HYPERSCALER, initial_params)
        
        self.company_name = company_name
        self.config = get_config()
        
        # Hyperscaler-specific attributes
        self.annual_capex_budget = initial_params.get('annual_capex_budget', 50e9)  # $50B default
        self.current_capacity_mw = initial_params.get('current_capacity_mw', 2000.0)
        self.target_growth_rate = initial_params.get('target_growth_rate', 0.30)  # 30% annual growth
        self.geographic_preferences = initial_params.get('geographic_preferences', {
            'USA': 0.45, 'China': 0.20, 'EU': 0.15, 'Asia_Pacific': 0.12, 'Others': 0.08
        })
        
        # Portfolio of datacenter assets
        self.datacenter_assets: List[DatacenterAsset] = []
        self.pending_constructions: List[Dict[str, Any]] = []
        
        # Strategic parameters
        self.ai_workload_focus = initial_params.get('ai_workload_focus', 0.6)  # 60% AI focus
        self.sustainability_commitment = initial_params.get('sustainability_commitment', 0.8)  # 80% renewable target
        self.supply_chain_diversification = initial_params.get('supply_chain_diversification', 0.7)
        
        # Market intelligence
        self.demand_forecasts = {}
        self.competitor_intelligence = {}
        self.supply_chain_status = {}
        
        # Decision-making parameters
        self.capacity_utilization_target = 0.75
        self.minimum_roi_threshold = 0.15  # 15% ROI threshold
        self.risk_assessment_weights = {
            'geopolitical': 0.3,
            'supply_chain': 0.25,
            'regulatory': 0.2,
            'market': 0.15,
            'operational': 0.1
        }
    
    def perceive(self, market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Perceive current market conditions and update intelligence."""
        perception = {
            'demand_growth': market_conditions.get('datacenter_demand_growth', 0.3),
            'supply_shortage': market_conditions.get('semiconductor_supply_shortage', 0.4),
            'power_availability': market_conditions.get('power_availability_constraint', 0.6),
            'geopolitical_risk': market_conditions.get('geopolitical_tension_level', 0.3),
            'construction_costs': market_conditions.get('average_construction_costs', 1000),
        }
        
        return perception
    
    def decide(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """Make strategic decisions based on current perception."""
        decisions = {}
        
        # Capacity planning decision
        capacity_decision = self._decide_capacity_expansion(perception)
        decisions['capacity_expansion'] = capacity_decision
        
        # Supply chain strategy
        supply_decision = self._decide_supply_chain_strategy(perception)
        decisions['supply_chain_strategy'] = supply_decision
        
        return decisions
    
    def act(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute decided actions and return results."""
        results = {
            'action_type': 'hyperscaler_operations',
            'revenue': 0.0,
            'costs': 0.0,
            'capacity_added': 0.0,
        }
        
        # Execute capacity expansion
        if decision.get('capacity_expansion', {}).get('expand', False):
            expansion_results = self._execute_capacity_expansion(decision['capacity_expansion'])
            results.update(expansion_results)
        
        # Calculate operational performance
        operational_results = self._calculate_operational_performance()
        results['revenue'] += operational_results['revenue']
        results['costs'] += operational_results['costs']
        
        return results
    
    def _decide_capacity_expansion(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """Decide on capacity expansion strategy."""
        current_utilization = self._calculate_current_utilization()
        demand_growth = perception['demand_growth']
        
        # Calculate required capacity
        target_capacity = self.current_capacity_mw * (1 + self.target_growth_rate)
        capacity_gap = max(0, target_capacity - self.current_capacity_mw)
        
        # Calculate investment requirements
        construction_cost_per_mw = perception['construction_costs'] * 1000
        total_investment = capacity_gap * construction_cost_per_mw
        
        # Check budget constraints
        available_budget = self.annual_capex_budget * 0.7
        
        if total_investment <= available_budget and current_utilization > 0.75:
            return {
                'expand': True,
                'target_capacity_mw': capacity_gap,
                'investment_amount': total_investment,
            }
        else:
            return {'expand': False}
    
    def _decide_supply_chain_strategy(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """Decide on supply chain strategy."""
        supply_shortage = perception['supply_shortage']
        
        if supply_shortage > 0.3:
            return {
                'diversify': True,
                'strategic_inventory_months': 6,
                'alternative_suppliers': 3
            }
        else:
            return {'diversify': False}
    
    def _execute_capacity_expansion(self, expansion_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute capacity expansion plan."""
        target_capacity = expansion_plan['target_capacity_mw']
        investment = expansion_plan['investment_amount']
        
        # Immediate costs (20% upfront)
        immediate_costs = investment * 0.2
        
        return {
            'capacity_expansion_costs': immediate_costs,
            'future_capacity_committed': target_capacity,
        }
    
    def _calculate_operational_performance(self) -> Dict[str, float]:
        """Calculate current operational performance."""
        total_capacity = sum(asset.capacity_mw for asset in self.datacenter_assets)
        if total_capacity == 0:
            total_capacity = self.current_capacity_mw
        
        # Simplified revenue calculation
        monthly_revenue_per_mw = 100000  # $100k per MW per month
        average_utilization = 0.75
        total_revenue = total_capacity * average_utilization * monthly_revenue_per_mw
        
        # Cost calculation
        total_costs = total_capacity * 30000  # $30k per MW per month
        
        return {
            'revenue': total_revenue,
            'costs': total_costs,
        }
    
    def _calculate_current_utilization(self) -> float:
        """Calculate current capacity utilization."""
        if not self.datacenter_assets:
            return 0.75  # Default assumption
        
        return np.mean([asset.utilization_rate for asset in self.datacenter_assets])
    
    def get_capacity_summary(self) -> Dict[str, Any]:
        """Get summary of capacity and utilization."""
        return {
            'total_capacity_mw': self.current_capacity_mw,
            'utilization_rate': self._calculate_current_utilization(),
            'annual_capex_budget': self.annual_capex_budget,
        }
    
    def get_financial_summary(self) -> Dict[str, Any]:
        """Get financial performance summary."""
        return {
            'annual_capex_budget': self.annual_capex_budget,
            'cash_balance': self.state.cash_balance,
            'total_revenue': self.performance_metrics['total_revenue'],
            'total_costs': self.performance_metrics['total_costs'],
            'market_share': self.performance_metrics.get('market_share', 0.0),
            'roi': self._calculate_roi()
        }
    
    def _calculate_roi(self) -> float:
        """Calculate return on investment."""
        total_investment = self.performance_metrics['total_costs']
        total_profit = self.performance_metrics['total_revenue'] - total_investment
        
        if total_investment > 0:
            return total_profit / total_investment
        return 0.0 