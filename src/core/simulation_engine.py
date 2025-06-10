"""
Core simulation engine for the datacenter capacity planning simulation.
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
import logging
from pathlib import Path
import json
from datetime import datetime, timedelta

from ..agents.hyperscaler_agent import HyperscalerAgent
from ..agents.nation_agent import NationAgent, NationState
from ..core.config import get_config, SimulationConfig, AgentConfig, GeopoliticalConfig
from ..data.generators import DatacenterSiteGenerator, SemiconductorSupplierGenerator, EconomicDataGenerator
from .market_dynamics import Market, MarketImpact


@dataclass
class SimulationState:
    """Overall simulation state."""
    current_time: float = 0.0  # Time in months
    global_demand_growth: float = 0.30
    semiconductor_shortage: float = 0.40
    power_constraint: float = 0.60
    geopolitical_tension: float = 0.30
    carbon_price: float = 85.0
    exchange_rates: Dict[str, float] = field(default_factory=lambda: {
        'EUR_USD': 1.08, 'CNY_USD': 7.20, 'JPY_USD': 150.0
    })


@dataclass
class SimulationResults:
    """Comprehensive results of a simulation run."""
    time_series: pd.DataFrame = field(default_factory=pd.DataFrame)
    agent_performance: Dict[str, pd.DataFrame] = field(default_factory=dict)
    market_evolution: Dict[str, List[Any]] = field(default_factory=dict)
    summary_statistics: Dict[str, Any] = field(default_factory=dict)
    policy_impacts: Dict[str, Any] = field(default_factory=dict)


class SimulationEngine:
    """Main simulation engine orchestrating the multi-agent system."""
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        """Initialize simulation engine."""
        self.config = config or get_config()
        self.logger = logging.getLogger(__name__)
        
        # Simulation state
        self.state = SimulationState()
        self.agents: Dict[str, BaseAgent] = {}
        self.market_data = {}
        
        # Results tracking
        self.results_history: List[Dict[str, Any]] = []
        self.agent_histories: Dict[str, List[Dict[str, Any]]] = {}
        
        # External data
        self.datacenter_sites = None
        self.semiconductor_suppliers = None
        
        # Scenario parameters
        self.scenario_parameters = {}
        
        self.simulation_config = get_config().simulation
        self.agent_config = get_config().agents
        self.geopolitical_config = get_config().geopolitical
        self.economic_config = get_config().economic
        
        self.market = Market(self.config)
        self.agents: Dict[str, Any] = {}
        self.time_step = 0
        self.current_date = datetime.now()
        self.results = SimulationResults()
        
        # Initialize data
        self.site_generator = DatacenterSiteGenerator()
        self.datacenter_sites = self.site_generator.generate_sites(self.config.data.num_datacenter_sites)
        self.supplier_generator = SemiconductorSupplierGenerator()
        self.semiconductor_suppliers = self.supplier_generator.generate_suppliers(self.config.data.num_suppliers)
        
    def initialize(self, scenario: str = "baseline") -> None:
        """Initialize simulation with agents and data."""
        self.logger.info(f"Initializing simulation with scenario: {scenario}")
        
        # Load scenario parameters
        self._load_scenario(scenario)
        
        # Generate or load data
        self._initialize_data()
        
        # Create agents
        self._create_agents()
        
        # Initialize market conditions
        self._initialize_market_conditions()
        
        self.logger.info(f"Simulation initialized with {len(self.agents)} agents")
    
    def run(self, duration_months: Optional[int] = None) -> SimulationResults:
        """Run the simulation for specified duration."""
        duration = duration_months or self.config.simulation.simulation_duration
        time_step = self.config.simulation.time_step
        
        self.logger.info(f"Starting simulation run for {duration} months")
        
        # Reset results tracking
        self.results_history = []
        self.agent_histories = {agent_id: [] for agent_id in self.agents}
        
        # Main simulation loop
        for step in range(int(duration / time_step)):
            current_time = step * time_step
            self.state.current_time = current_time
            
            # Update market conditions
            self._update_market_conditions(current_time)
            
            # Agent perception phase
            agent_perceptions = self._agent_perception_phase()
            
            # Agent decision phase
            agent_decisions = self._agent_decision_phase(agent_perceptions)
            
            # Agent action phase
            action_results = self._agent_action_phase(agent_decisions)
            
            # Update agent states
            self._update_agent_states(current_time, action_results)
            
            # Record results
            self._record_timestep_results(current_time, agent_perceptions, 
                                        agent_decisions, action_results)
            
            # Check for termination conditions
            if self._check_termination_conditions():
                self.logger.info(f"Simulation terminated early at time {current_time}")
                break
            
            if step % 12 == 0:  # Log every 12 months
                self.logger.info(f"Simulation progress: {current_time:.1f} months")
        
        # Generate final results
        results = self._generate_results()
        
        self.logger.info("Simulation completed successfully")
        return results
    
    def run_scenario_analysis(self, scenarios: List[str]) -> Dict[str, SimulationResults]:
        """Run multiple scenarios and compare results."""
        scenario_results = {}
        
        for scenario in scenarios:
            self.logger.info(f"Running scenario: {scenario}")
            
            # Reset and initialize for this scenario
            self.initialize(scenario)
            
            # Run simulation
            results = self.run()
            scenario_results[scenario] = results
            
            # Reset agents for next scenario
            for agent in self.agents.values():
                agent.reset()
        
        return scenario_results
    
    def _load_scenario(self, scenario_name: str) -> None:
        """Load scenario-specific parameters."""
        # Default baseline scenario
        baseline_params = {
            'global_demand_growth': 0.30,
            'semiconductor_shortage': 0.40,
            'power_constraint': 0.60,
            'geopolitical_tension': 0.30,
            'carbon_price': 85.0,
            'trade_war_probability': 0.25,
            'supply_chain_disruption_frequency': 0.15
        }
        
        # Scenario variations
        scenario_variations = {
            'high_growth': {
                'global_demand_growth': 0.45,
                'semiconductor_shortage': 0.50,
                'power_constraint': 0.70
            },
            'trade_war': {
                'geopolitical_tension': 0.70,
                'trade_war_probability': 0.80,
                'semiconductor_shortage': 0.60
            },
            'green_transition': {
                'carbon_price': 150.0,
                'power_constraint': 0.80,
                'renewable_mandate': 0.90
            },
            'supply_crisis': {
                'semiconductor_shortage': 0.80,
                'supply_chain_disruption_frequency': 0.40,
                'geopolitical_tension': 0.60
            }
        }
        
        # Load baseline and apply scenario-specific changes
        self.scenario_parameters = baseline_params.copy()
        if scenario_name in scenario_variations:
            self.scenario_parameters.update(scenario_variations[scenario_name])
        
        # Update simulation state with scenario parameters
        self.state.global_demand_growth = self.scenario_parameters['global_demand_growth']
        self.state.semiconductor_shortage = self.scenario_parameters['semiconductor_shortage']
        self.state.power_constraint = self.scenario_parameters['power_constraint']
        self.state.geopolitical_tension = self.scenario_parameters['geopolitical_tension']
        self.state.carbon_price = self.scenario_parameters['carbon_price']
    
    def _initialize_data(self) -> None:
        """Initialize or load simulation data."""
        # Generate datacenter sites if not already loaded
        if self.datacenter_sites is None:
            self.datacenter_sites = generate_sample_data()
            self.logger.info(f"Generated {len(self.datacenter_sites)} datacenter sites")
    
    def _create_agents(self) -> None:
        """Create and initialize all agents."""
        # Create hyperscaler agents
        hyperscaler_configs = {
            'Google': {
                'annual_capex_budget': 35e9,
                'current_capacity_mw': 2500,
                'ai_workload_focus': 0.70,
                'sustainability_commitment': 0.90
            },
            'Amazon': {
                'annual_capex_budget': 65e9,
                'current_capacity_mw': 4000,
                'ai_workload_focus': 0.50,
                'sustainability_commitment': 0.80
            },
            'Microsoft': {
                'annual_capex_budget': 45e9,
                'current_capacity_mw': 3200,
                'ai_workload_focus': 0.65,
                'sustainability_commitment': 0.85
            },
            'Meta': {
                'annual_capex_budget': 28e9,
                'current_capacity_mw': 1800,
                'ai_workload_focus': 0.75,
                'sustainability_commitment': 0.75
            }
        }
        
        for company, config in hyperscaler_configs.items():
            agent = HyperscalerAgent(
                agent_id=company.lower(),
                company_name=company,
                initial_params=config
            )
            self.agents[agent.agent_id] = agent
        
        self.logger.info(f"Created {len(self.agents)} agents")
    
    def _initialize_market_conditions(self) -> None:
        """Initialize market conditions and external factors."""
        self.market_data = {
            'datacenter_demand_growth': self.state.global_demand_growth,
            'semiconductor_supply_shortage': self.state.semiconductor_shortage,
            'power_availability_constraint': self.state.power_constraint,
            'geopolitical_tension_level': self.state.geopolitical_tension,
            'average_construction_costs': 1000,  # $/kW
            'average_power_costs': 70,  # $/MWh
            'carbon_pricing': self.state.carbon_price,
            'exchange_rates': self.state.exchange_rates.copy(),
            'total_market_capacity': sum(
                agent.current_capacity_mw for agent in self.agents.values()
                if isinstance(agent, HyperscalerAgent)
            )
        }
    
    def _update_market_conditions(self, current_time: float) -> None:
        """Update market conditions based on simulation dynamics."""
        # Add time-varying dynamics
        
        # Demand growth with business cycles
        cycle_factor = 1 + 0.1 * np.sin(2 * np.pi * current_time / 48)  # 4-year cycle
        noise = np.random.normal(0, 0.05)
        self.market_data['datacenter_demand_growth'] = (
            self.state.global_demand_growth * cycle_factor + noise
        )
        
        # Semiconductor shortage with supply chain dynamics
        if np.random.random() < self.scenario_parameters.get('supply_chain_disruption_frequency', 0.15) / 12:
            # Supply chain disruption event
            self.market_data['semiconductor_supply_shortage'] = min(0.9, 
                self.market_data['semiconductor_supply_shortage'] + np.random.uniform(0.1, 0.3))
        else:
            # Gradual recovery
            self.market_data['semiconductor_supply_shortage'] = max(0.1,
                self.market_data['semiconductor_supply_shortage'] - 0.02)
        
        # Geopolitical tension evolution
        if np.random.random() < 0.05:  # 5% chance per month of geopolitical event
            tension_change = np.random.normal(0, 0.15)
            self.state.geopolitical_tension = max(0.1, min(0.9, 
                self.state.geopolitical_tension + tension_change))
            self.market_data['geopolitical_tension_level'] = self.state.geopolitical_tension
        
        # Carbon price evolution
        carbon_trend = 0.002  # 2% monthly increase trend
        carbon_volatility = 0.05
        carbon_change = carbon_trend + np.random.normal(0, carbon_volatility)
        self.state.carbon_price *= (1 + carbon_change)
        self.market_data['carbon_pricing'] = self.state.carbon_price
        
        # Construction cost inflation
        cost_inflation = 0.003  # 3% annual inflation
        self.market_data['average_construction_costs'] *= (1 + cost_inflation / 12)
        
        # Power cost dynamics
        power_volatility = 0.03
        power_change = np.random.normal(0, power_volatility)
        self.market_data['average_power_costs'] *= (1 + power_change)
    
    def _agent_perception_phase(self) -> Dict[str, Dict[str, Any]]:
        """Execute perception phase for all agents."""
        perceptions = {}
        
        for agent_id, agent in self.agents.items():
            try:
                perception = agent.perceive(self.market_data)
                perceptions[agent_id] = perception
            except Exception as e:
                self.logger.error(f"Error in perception phase for agent {agent_id}: {e}")
                perceptions[agent_id] = {}
        
        return perceptions
    
    def _agent_decision_phase(self, perceptions: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Execute decision phase for all agents."""
        decisions = {}
        
        for agent_id, agent in self.agents.items():
            try:
                decision = agent.decide(perceptions[agent_id])
                decisions[agent_id] = decision
            except Exception as e:
                self.logger.error(f"Error in decision phase for agent {agent_id}: {e}")
                decisions[agent_id] = {}
        
        return decisions
    
    def _agent_action_phase(self, decisions: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Execute action phase for all agents."""
        action_results = {}
        
        for agent_id, agent in self.agents.items():
            try:
                result = agent.act(decisions[agent_id])
                action_results[agent_id] = result
            except Exception as e:
                self.logger.error(f"Error in action phase for agent {agent_id}: {e}")
                action_results[agent_id] = {
                    'action_type': 'error',
                    'revenue': 0.0,
                    'costs': 0.0
                }
        
        return action_results
    
    def _update_agent_states(self, current_time: float, action_results: Dict[str, Dict[str, Any]]) -> None:
        """Update agent states based on action results."""
        for agent_id, agent in self.agents.items():
            try:
                agent.update_state(current_time, action_results[agent_id])
            except Exception as e:
                self.logger.error(f"Error updating state for agent {agent_id}: {e}")
    
    def _record_timestep_results(self, current_time: float, perceptions: Dict[str, Dict[str, Any]],
                                decisions: Dict[str, Dict[str, Any]], 
                                action_results: Dict[str, Dict[str, Any]]) -> None:
        """Record results for current timestep."""
        # Record global results
        global_results = {
            'time': current_time,
            'market_demand_growth': self.market_data['datacenter_demand_growth'],
            'semiconductor_shortage': self.market_data['semiconductor_supply_shortage'],
            'geopolitical_tension': self.market_data['geopolitical_tension_level'],
            'carbon_price': self.market_data['carbon_pricing'],
            'construction_costs': self.market_data['average_construction_costs'],
            'total_market_revenue': sum(r.get('revenue', 0) for r in action_results.values()),
            'total_market_costs': sum(r.get('costs', 0) for r in action_results.values()),
        }
        self.results_history.append(global_results)
        
        # Record agent-specific results
        for agent_id, agent in self.agents.items():
            agent_result = {
                'time': current_time,
                'agent_id': agent_id,
                'cash_balance': agent.state.cash_balance,
                'reputation_score': agent.state.reputation_score,
                'revenue': action_results[agent_id].get('revenue', 0),
                'costs': action_results[agent_id].get('costs', 0),
                'action_type': action_results[agent_id].get('action_type', 'none')
            }
            
            # Add agent-specific metrics
            if isinstance(agent, HyperscalerAgent):
                capacity_summary = agent.get_capacity_summary()
                agent_result.update({
                    'capacity_mw': capacity_summary['total_capacity_mw'],
                    'utilization_rate': capacity_summary['utilization_rate'],
                    'capex_budget': capacity_summary['annual_capex_budget']
                })
            
            self.agent_histories[agent_id].append(agent_result)
    
    def _check_termination_conditions(self) -> bool:
        """Check if simulation should terminate early."""
        # Check for market collapse
        total_revenue = sum(r.get('revenue', 0) for r in self.results_history[-1:])
        if total_revenue == 0 and self.state.current_time > 12:  # No revenue for a year
            return True
        
        # Check for extreme market conditions
        if self.market_data['semiconductor_supply_shortage'] > 0.95:
            return True
        
        return False
    
    def _generate_results(self) -> SimulationResults:
        """Generate comprehensive simulation results."""
        # Convert results to DataFrames
        time_series_df = pd.DataFrame(self.results_history)
        
        agent_performance_dfs = {}
        for agent_id, history in self.agent_histories.items():
            agent_performance_dfs[agent_id] = pd.DataFrame(history)
        
        # Generate market evolution DataFrame
        market_evolution_df = time_series_df[[
            'time', 'market_demand_growth', 'semiconductor_shortage',
            'geopolitical_tension', 'carbon_price'
        ]].copy()
        
        # Calculate summary statistics
        summary_stats = self._calculate_summary_statistics(time_series_df, agent_performance_dfs)
        
        # Analyze policy impacts
        policy_impacts = self._analyze_policy_impacts(time_series_df, agent_performance_dfs)
        
        return SimulationResults(
            time_series=time_series_df,
            agent_performance=agent_performance_dfs,
            market_evolution=market_evolution_df,
            summary_statistics=summary_stats,
            policy_impacts=policy_impacts
        )
    
    def _calculate_summary_statistics(self, time_series: pd.DataFrame, 
                                    agent_performance: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Calculate summary statistics for the simulation."""
        stats = {}
        
        # Market-level statistics
        stats['market'] = {
            'total_revenue_final': time_series['total_market_revenue'].iloc[-1],
            'total_costs_final': time_series['total_market_costs'].iloc[-1],
            'average_demand_growth': time_series['market_demand_growth'].mean(),
            'average_shortage_level': time_series['semiconductor_shortage'].mean(),
            'final_carbon_price': time_series['carbon_price'].iloc[-1],
            'carbon_price_growth': (time_series['carbon_price'].iloc[-1] / 
                                  time_series['carbon_price'].iloc[0] - 1)
        }
        
        # Agent-level statistics
        stats['agents'] = {}
        for agent_id, df in agent_performance.items():
            if len(df) > 0:
                stats['agents'][agent_id] = {
                    'final_cash_balance': df['cash_balance'].iloc[-1],
                    'total_revenue': df['revenue'].sum(),
                    'total_costs': df['costs'].sum(),
                    'profit_margin': (df['revenue'].sum() - df['costs'].sum()) / 
                                   max(df['revenue'].sum(), 1),
                    'average_reputation': df['reputation_score'].mean()
                }
                
                # Add hyperscaler-specific metrics
                if 'capacity_mw' in df.columns:
                    stats['agents'][agent_id].update({
                        'final_capacity_mw': df['capacity_mw'].iloc[-1],
                        'capacity_growth': (df['capacity_mw'].iloc[-1] / 
                                          max(df['capacity_mw'].iloc[0], 1) - 1),
                        'average_utilization': df['utilization_rate'].mean()
                    })
        
        return stats
    
    def _analyze_policy_impacts(self, time_series: pd.DataFrame,
                              agent_performance: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze the impact of different policies and market conditions."""
        impacts = {}
        
        # Carbon pricing impact
        carbon_correlation = time_series[['carbon_price', 'total_market_costs']].corr().iloc[0, 1]
        impacts['carbon_pricing'] = {
            'cost_correlation': carbon_correlation,
            'price_elasticity': self._calculate_price_elasticity(time_series),
            'revenue_impact': 'negative' if carbon_correlation > 0.3 else 'minimal'
        }
        
        # Supply chain disruption impact
        shortage_impact = time_series[['semiconductor_shortage', 'total_market_revenue']].corr().iloc[0, 1]
        impacts['supply_chain'] = {
            'revenue_correlation': shortage_impact,
            'disruption_severity': 'high' if shortage_impact < -0.5 else 'moderate',
            'recovery_pattern': self._analyze_recovery_pattern(time_series)
        }
        
        # Geopolitical tension impact
        geo_correlation = time_series[['geopolitical_tension', 'total_market_costs']].corr().iloc[0, 1]
        impacts['geopolitical'] = {
            'cost_impact': geo_correlation,
            'investment_deterrent': 'significant' if geo_correlation > 0.4 else 'limited'
        }
        
        return impacts
    
    def _calculate_price_elasticity(self, time_series: pd.DataFrame) -> float:
        """Calculate price elasticity of demand."""
        # Simplified elasticity calculation
        if len(time_series) < 2:
            return 0.0
        
        price_changes = time_series['carbon_price'].pct_change().dropna()
        demand_changes = time_series['market_demand_growth'].pct_change().dropna()
        
        if len(price_changes) == 0 or price_changes.std() == 0:
            return 0.0
        
        correlation = price_changes.corr(demand_changes)
        return correlation if not np.isnan(correlation) else 0.0
    
    def _analyze_recovery_pattern(self, time_series: pd.DataFrame) -> str:
        """Analyze market recovery patterns from disruptions."""
        # Simplified recovery pattern analysis
        shortage_series = time_series['semiconductor_shortage']
        
        # Find periods of high shortage
        high_shortage_periods = shortage_series > shortage_series.quantile(0.75)
        
        if not high_shortage_periods.any():
            return "no_major_disruptions"
        
        # Look at recovery speed after disruptions
        recovery_speeds = []
        in_disruption = False
        disruption_start = 0
        
        for i, is_disrupted in enumerate(high_shortage_periods):
            if is_disrupted and not in_disruption:
                in_disruption = True
                disruption_start = i
            elif not is_disrupted and in_disruption:
                in_disruption = False
                recovery_length = i - disruption_start
                recovery_speeds.append(recovery_length)
        
        if recovery_speeds:
            avg_recovery = np.mean(recovery_speeds)
            if avg_recovery < 6:  # Less than 6 months
                return "fast_recovery"
            elif avg_recovery < 12:  # Less than 12 months
                return "moderate_recovery"
            else:
                return "slow_recovery"
        
        return "ongoing_disruption"
    
    def save_results(self, results: SimulationResults, output_dir: Path) -> None:
        """Save simulation results to files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save time series data
        results.time_series.to_csv(output_dir / "time_series.csv", index=False)
        
        # Save agent performance data
        for agent_id, df in results.agent_performance.items():
            df.to_csv(output_dir / f"agent_{agent_id}_performance.csv", index=False)
        
        # Save market evolution
        results.market_evolution.to_csv(output_dir / "market_evolution.csv", index=False)
        
        # Save summary statistics and policy impacts as JSON
        with open(output_dir / "summary_statistics.json", 'w') as f:
            json.dump(results.summary_statistics, f, indent=2, default=str)
        
        with open(output_dir / "policy_impacts.json", 'w') as f:
            json.dump(results.policy_impacts, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {output_dir}")
    
    def get_agent_summary(self) -> Dict[str, Any]:
        """Get summary of all agents."""
        summary = {}
        for agent_id, agent in self.agents.items():
            summary[agent_id] = agent.get_state_summary()
        return summary 