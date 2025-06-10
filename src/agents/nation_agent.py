"""
Nation Agent: Comprehensive Government Policy and Strategic Decision-Making.

This agent models national governments making complex policy decisions including:
- Trade policy and international negotiations
- Industrial policy and domestic capacity building  
- Energy policy and climate commitments
- Geopolitical strategy and alliance management
- Regulatory frameworks and competition policy
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import logging

from .base_agent import BaseAgent, AgentType
from ..core.config import get_config
from ..economics.welfare_analysis import WelfareAnalyzer
from ..economics.international_trade import TradeNetworkAnalyzer


class PolicyDomain(Enum):
    """Policy domains for nation agent decision-making."""
    TRADE = "trade"
    INDUSTRIAL = "industrial"
    ENERGY = "energy"
    TECHNOLOGY = "technology"
    COMPETITION = "competition"
    FOREIGN = "foreign"
    FISCAL = "fiscal"
    MONETARY = "monetary"


class GeopoliticalStance(Enum):
    """Geopolitical stance options."""
    COOPERATIVE = "cooperative"
    COMPETITIVE = "competitive"
    PROTECTIONIST = "protectionist"
    ISOLATIONIST = "isolationist"
    AGGRESSIVE = "aggressive"


@dataclass
class PolicyPosition:
    """Represents a policy position in a specific domain."""
    domain: PolicyDomain
    position_value: float  # -1.0 to 1.0 scale
    confidence: float  # 0.0 to 1.0
    domestic_support: float  # 0.0 to 1.0
    international_pressure: float  # -1.0 to 1.0
    implementation_cost: float
    expected_benefit: float
    time_horizon: int  # months
    stakeholder_impacts: Dict[str, float]


@dataclass
class NationState:
    """Current state of the nation."""
    gdp: float
    gdp_growth_rate: float
    unemployment_rate: float
    inflation_rate: float
    debt_to_gdp_ratio: float
    current_account_balance: float
    energy_independence_ratio: float
    technology_competitiveness_index: float
    political_stability_index: float
    democratic_institutions_strength: float
    military_capability_index: float
    soft_power_index: float
    carbon_emissions_per_capita: float
    renewable_energy_share: float
    datacenter_capacity_mw: float
    semiconductor_manufacturing_capacity: float
    strategic_reserves: Dict[str, float]


@dataclass
class DiplomaticRelationship:
    """Bilateral diplomatic relationship."""
    partner_nation: str
    relationship_quality: float  # -1.0 to 1.0
    trade_volume: float
    military_cooperation: float
    technology_sharing: float
    diplomatic_exchanges: int
    shared_interests_alignment: float
    territorial_disputes: bool
    sanctions_in_place: bool
    alliance_membership: List[str]


class NationAgent(BaseAgent):
    """
    Comprehensive nation agent modeling government policy decisions.
    
    This agent makes strategic decisions across multiple policy domains while balancing:
    - Domestic political constraints and public opinion
    - International relations and alliance commitments  
    - Economic competitiveness and national security
    - Long-term strategic objectives and short-term pressures
    """
    
    def __init__(self, nation_name: str, initial_state: NationState,
                 political_system: str = "democracy", **kwargs):
        """
        Initialize nation agent.
        
        Args:
            nation_name: Name of the nation
            initial_state: Initial economic and political state
            political_system: Type of political system (democracy, autocracy, hybrid)
        """
        super().__init__(
            agent_id=f"nation_{nation_name.lower()}",
            agent_type=AgentType.NATION,
            **kwargs
        )
        
        self.nation_name = nation_name
        self.political_system = political_system
        self.state = initial_state
        self.config = get_config()
        
        # Policy framework
        self.policy_positions = self._initialize_policy_positions()
        self.geopolitical_stance = GeopoliticalStance.COOPERATIVE
        self.strategic_objectives = self._define_strategic_objectives()
        
        # Diplomatic relationships
        self.diplomatic_relationships = {}
        self.alliance_memberships = self._initialize_alliances()
        self.trade_agreements = self._initialize_trade_agreements()
        
        # Decision-making parameters
        self.policy_learning_rate = 0.1
        self.political_discount_factor = 0.95  # Future vs present political benefits
        self.domestic_pressure_weight = 0.6
        self.international_pressure_weight = 0.4
        
        # Institutional constraints
        self.institutional_constraints = self._initialize_institutional_constraints()
        self.decision_making_speed = self._calculate_decision_speed()
        
        # Economic analysis tools
        self.welfare_analyzer = WelfareAnalyzer()
        self.trade_analyzer = TradeNetworkAnalyzer()
        
        # Performance tracking
        self.policy_outcomes_history = []
        self.approval_rating_history = []
        self.international_reputation_history = []
        
        self.logger = logging.getLogger(f"{__name__}.{self.agent_id}")
    
    def _initialize_policy_positions(self) -> Dict[PolicyDomain, PolicyPosition]:
        """Initialize baseline policy positions across all domains."""
        positions = {}
        
        # Trade policy position
        positions[PolicyDomain.TRADE] = PolicyPosition(
            domain=PolicyDomain.TRADE,
            position_value=0.2,  # Slightly pro-trade
            confidence=0.7,
            domestic_support=0.6,
            international_pressure=0.3,
            implementation_cost=1e9,
            expected_benefit=5e9,
            time_horizon=24,
            stakeholder_impacts={
                'exporters': 0.8, 'importers': 0.6, 'domestic_producers': -0.3,
                'consumers': 0.4, 'labor_unions': -0.2
            }
        )
        
        # Industrial policy position
        positions[PolicyDomain.INDUSTRIAL] = PolicyPosition(
            domain=PolicyDomain.INDUSTRIAL,
            position_value=0.5,  # Pro-industrial development
            confidence=0.8,
            domestic_support=0.7,
            international_pressure=-0.1,
            implementation_cost=20e9,
            expected_benefit=50e9,
            time_horizon=60,
            stakeholder_impacts={
                'domestic_industry': 0.9, 'technology_firms': 0.8, 'labor_unions': 0.6,
                'taxpayers': -0.3, 'foreign_competitors': -0.7
            }
        )
        
        # Energy policy position
        positions[PolicyDomain.ENERGY] = PolicyPosition(
            domain=PolicyDomain.ENERGY,
            position_value=0.6,  # Pro-renewable energy
            confidence=0.6,
            domestic_support=0.5,
            international_pressure=0.7,
            implementation_cost=100e9,
            expected_benefit=200e9,
            time_horizon=120,
            stakeholder_impacts={
                'renewable_industry': 0.9, 'environmental_groups': 0.8,
                'fossil_fuel_industry': -0.8, 'energy_consumers': -0.2, 'youth_voters': 0.7
            }
        )
        
        # Technology policy position
        positions[PolicyDomain.TECHNOLOGY] = PolicyPosition(
            domain=PolicyDomain.TECHNOLOGY,
            position_value=0.7,  # Pro-technology development
            confidence=0.9,
            domestic_support=0.8,
            international_pressure=0.2,
            implementation_cost=50e9,
            expected_benefit=150e9,
            time_horizon=84,
            stakeholder_impacts={
                'tech_industry': 0.9, 'universities': 0.8, 'high_skilled_workers': 0.7,
                'traditional_industry': -0.2, 'low_skilled_workers': -0.3
            }
        )
        
        # Competition policy position
        positions[PolicyDomain.COMPETITION] = PolicyPosition(
            domain=PolicyDomain.COMPETITION,
            position_value=-0.1,  # Slightly less restrictive
            confidence=0.5,
            domestic_support=0.4,
            international_pressure=-0.2,
            implementation_cost=2e9,
            expected_benefit=10e9,
            time_horizon=36,
            stakeholder_impacts={
                'large_corporations': 0.4, 'small_businesses': 0.6, 'consumers': 0.3,
                'regulators': -0.1, 'antitrust_advocates': -0.7
            }
        )
        
        # Foreign policy position
        positions[PolicyDomain.FOREIGN] = PolicyPosition(
            domain=PolicyDomain.FOREIGN,
            position_value=0.3,  # Moderately cooperative
            confidence=0.6,
            domestic_support=0.5,
            international_pressure=0.4,
            implementation_cost=15e9,
            expected_benefit=30e9,
            time_horizon=48,
            stakeholder_impacts={
                'defense_industry': 0.3, 'diplomatic_corps': 0.6, 'trade_associations': 0.4,
                'isolationist_voters': -0.5, 'internationalist_voters': 0.7
            }
        )
        
        return positions
    
    def _define_strategic_objectives(self) -> Dict[str, float]:
        """Define long-term strategic objectives with weights."""
        return {
            'economic_growth': 0.25,
            'national_security': 0.20,
            'technological_leadership': 0.15,
            'energy_security': 0.10,
            'environmental_sustainability': 0.10,
            'social_stability': 0.10,
            'international_influence': 0.10
        }
    
    def _initialize_alliances(self) -> List[str]:
        """Initialize alliance memberships based on nation."""
        alliance_map = {
            'USA': ['NATO', 'AUKUS', 'QUAD', 'Five_Eyes'],
            'China': ['SCO', 'BRICS', 'RCEP'],
            'Germany': ['NATO', 'EU', 'G7'],
            'Japan': ['QUAD', 'CPTPP', 'G7'],
            'South_Korea': ['KORUS', 'CPTPP'],
            'Taiwan': ['APEC'],
            'Singapore': ['ASEAN', 'CPTPP', 'RCEP'],
            'Netherlands': ['NATO', 'EU', 'G7']
        }
        return alliance_map.get(self.nation_name, [])
    
    def _initialize_trade_agreements(self) -> List[str]:
        """Initialize trade agreement memberships."""
        trade_map = {
            'USA': ['USMCA', 'KORUS'],
            'China': ['RCEP', 'ASEAN_China_FTA'],
            'Germany': ['EU_Single_Market', 'CETA'],
            'Japan': ['CPTPP', 'JSEPA', 'RCEP'],
            'South_Korea': ['KORUS', 'RCEP', 'Korea_EU_FTA'],
            'Taiwan': ['Taiwan_Singapore_EPA'],
            'Singapore': ['CPTPP', 'RCEP', 'USSFTA'],
            'Netherlands': ['EU_Single_Market', 'CETA']
        }
        return trade_map.get(self.nation_name, [])
    
    def _initialize_institutional_constraints(self) -> Dict[str, float]:
        """Initialize institutional constraints on decision-making."""
        if self.political_system == "democracy":
            return {
                'legislative_approval_required': 0.8,
                'public_opinion_sensitivity': 0.7,
                'interest_group_influence': 0.6,
                'judicial_review_constraint': 0.5,
                'media_scrutiny': 0.8,
                'electoral_cycle_pressure': 0.9,
                'coalition_government_constraint': 0.4
            }
        elif self.political_system == "autocracy":
            return {
                'legislative_approval_required': 0.1,
                'public_opinion_sensitivity': 0.3,
                'interest_group_influence': 0.2,
                'judicial_review_constraint': 0.1,
                'media_scrutiny': 0.2,
                'electoral_cycle_pressure': 0.1,
                'elite_consensus_required': 0.7
            }
        else:  # hybrid
            return {
                'legislative_approval_required': 0.5,
                'public_opinion_sensitivity': 0.5,
                'interest_group_influence': 0.4,
                'judicial_review_constraint': 0.3,
                'media_scrutiny': 0.5,
                'electoral_cycle_pressure': 0.6,
                'elite_consensus_required': 0.5
            }
    
    def _calculate_decision_speed(self) -> float:
        """Calculate speed of decision-making based on institutional constraints."""
        constraint_sum = sum(self.institutional_constraints.values())
        max_constraint_sum = len(self.institutional_constraints)
        return 1.0 - (constraint_sum / max_constraint_sum)
    
    def perceive(self, market_state: Dict[str, Any], agent_states: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perceive current economic, political, and geopolitical environment.
        
        Args:
            market_state: Current market conditions
            agent_states: States of other agents
            
        Returns:
            Comprehensive perception of current situation
        """
        perception = {
            'economic_indicators': self._perceive_economic_conditions(market_state),
            'geopolitical_situation': self._perceive_geopolitical_environment(agent_states),
            'domestic_political_conditions': self._perceive_domestic_politics(),
            'international_trade_flows': self._perceive_trade_conditions(market_state),
            'technology_landscape': self._perceive_technology_conditions(market_state),
            'energy_security_status': self._perceive_energy_security(market_state),
            'crisis_indicators': self._identify_crisis_indicators(market_state, agent_states),
            'opportunity_indicators': self._identify_opportunities(market_state, agent_states)
        }
        
        return perception
    
    def _perceive_economic_conditions(self, market_state: Dict[str, Any]) -> Dict[str, float]:
        """Perceive current economic conditions."""
        return {
            'gdp_growth_trend': market_state.get('gdp_growth_rate', 0.02),
            'inflation_rate': market_state.get('inflation_rate', 0.025),
            'unemployment_rate': self.state.unemployment_rate,
            'datacenter_market_growth': market_state.get('demand_growth', 0.15),
            'semiconductor_shortage_severity': market_state.get('semiconductor_shortage', 0.0),
            'energy_prices': market_state.get('energy_price_index', 1.0),
            'trade_balance': self.state.current_account_balance,
            'currency_strength': market_state.get('exchange_rate_volatility', 0.1)
        }
    
    def _perceive_geopolitical_environment(self, agent_states: Dict[str, Any]) -> Dict[str, float]:
        """Perceive geopolitical tensions and relationships."""
        geopolitical_perception = {
            'global_tension_level': 0.0,
            'trade_war_intensity': 0.0,
            'technology_competition_intensity': 0.0,
            'alliance_stability': 0.8,
            'multilateral_cooperation_level': 0.6
        }
        
        # Analyze relationships with other nations
        for agent_id, agent_state in agent_states.items():
            if agent_id.startswith('nation_') and agent_id != self.agent_id:
                other_nation = agent_id.replace('nation_', '').title()
                
                # Calculate tension indicators
                if other_nation in ['China'] and self.nation_name == 'USA':
                    geopolitical_perception['global_tension_level'] += 0.3
                    geopolitical_perception['trade_war_intensity'] += 0.4
                    geopolitical_perception['technology_competition_intensity'] += 0.5
                elif other_nation in ['USA'] and self.nation_name == 'China':
                    geopolitical_perception['global_tension_level'] += 0.3
                    geopolitical_perception['trade_war_intensity'] += 0.4
                    geopolitical_perception['technology_competition_intensity'] += 0.5
        
        # Normalize values
        for key in geopolitical_perception:
            geopolitical_perception[key] = min(1.0, geopolitical_perception[key])
        
        return geopolitical_perception
    
    def _perceive_domestic_politics(self) -> Dict[str, float]:
        """Perceive domestic political conditions."""
        # Simulate approval rating based on economic performance
        economic_performance = (self.state.gdp_growth_rate - 0.02) * 10  # Baseline 2% growth
        approval_rating = 0.5 + economic_performance + np.random.normal(0, 0.1)
        approval_rating = max(0.2, min(0.8, approval_rating))
        
        return {
            'approval_rating': approval_rating,
            'political_stability': self.state.political_stability_index,
            'legislative_support': 0.6,  # Simplified
            'interest_group_pressure': 0.4,
            'upcoming_election_pressure': 0.3,  # Simplified
            'public_opinion_volatility': 0.2
        }
    
    def _perceive_trade_conditions(self, market_state: Dict[str, Any]) -> Dict[str, float]:
        """Perceive international trade conditions."""
        return {
            'global_trade_volume': market_state.get('global_trade_volume', 1.0),
            'supply_chain_disruption_level': market_state.get('supply_chain_disruption', 0.1),
            'tariff_war_escalation': market_state.get('tariff_escalation', 0.0),
            'wto_dispute_intensity': 0.2,  # Simplified
            'regional_trade_integration': 0.7,
            'digital_trade_growth': 0.3
        }
    
    def _perceive_technology_conditions(self, market_state: Dict[str, Any]) -> Dict[str, float]:
        """Perceive technology competition landscape."""
        return {
            'semiconductor_technological_leadership': self.state.technology_competitiveness_index,
            'ai_development_competition': 0.8,
            'quantum_computing_race': 0.6,
            'technology_transfer_restrictions': market_state.get('export_controls', 0.0),
            'research_collaboration_openness': 0.5,
            'intellectual_property_protection': 0.8
        }
    
    def _perceive_energy_security(self, market_state: Dict[str, Any]) -> Dict[str, float]:
        """Perceive energy security conditions."""
        return {
            'energy_independence_level': self.state.energy_independence_ratio,
            'renewable_energy_transition_progress': self.state.renewable_energy_share,
            'critical_minerals_supply_security': 0.6,
            'datacenter_energy_demand_pressure': market_state.get('datacenter_energy_demand', 0.2),
            'grid_stability_risk': 0.1,
            'climate_policy_pressure': 0.7
        }
    
    def _identify_crisis_indicators(self, market_state: Dict[str, Any], 
                                  agent_states: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify potential crisis indicators requiring urgent action."""
        crisis_indicators = []
        
        # Economic crisis indicators
        if self.state.gdp_growth_rate < -0.02:  # Recession
            crisis_indicators.append({
                'type': 'economic_recession',
                'severity': abs(self.state.gdp_growth_rate) * 50,
                'urgency': 0.9,
                'policy_domains_affected': [PolicyDomain.FISCAL, PolicyDomain.MONETARY, PolicyDomain.TRADE]
            })
        
        # Supply chain crisis
        semiconductor_shortage = market_state.get('semiconductor_shortage', 0.0)
        if semiconductor_shortage > 0.3:
            crisis_indicators.append({
                'type': 'supply_chain_crisis',
                'severity': semiconductor_shortage,
                'urgency': 0.8,
                'policy_domains_affected': [PolicyDomain.INDUSTRIAL, PolicyDomain.TRADE, PolicyDomain.TECHNOLOGY]
            })
        
        # Energy security crisis
        if self.state.energy_independence_ratio < 0.3:
            crisis_indicators.append({
                'type': 'energy_security_crisis',
                'severity': 1 - self.state.energy_independence_ratio,
                'urgency': 0.7,
                'policy_domains_affected': [PolicyDomain.ENERGY, PolicyDomain.FOREIGN, PolicyDomain.INDUSTRIAL]
            })
        
        # Geopolitical crisis
        global_tension = self._perceive_geopolitical_environment(agent_states)['global_tension_level']
        if global_tension > 0.7:
            crisis_indicators.append({
                'type': 'geopolitical_crisis',
                'severity': global_tension,
                'urgency': 0.8,
                'policy_domains_affected': [PolicyDomain.FOREIGN, PolicyDomain.TRADE, PolicyDomain.TECHNOLOGY]
            })
        
        return crisis_indicators
    
    def _identify_opportunities(self, market_state: Dict[str, Any], 
                              agent_states: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify strategic opportunities for advancement."""
        opportunities = []
        
        # Technology leadership opportunity
        if self.state.technology_competitiveness_index > 0.8:
            opportunities.append({
                'type': 'technology_leadership',
                'potential_benefit': 0.9,
                'time_sensitivity': 0.7,
                'policy_domains_relevant': [PolicyDomain.TECHNOLOGY, PolicyDomain.INDUSTRIAL, PolicyDomain.TRADE]
            })
        
        # Green energy transition opportunity
        if market_state.get('carbon_price_trend', 0) > 0.1:
            opportunities.append({
                'type': 'green_energy_leadership',
                'potential_benefit': 0.8,
                'time_sensitivity': 0.6,
                'policy_domains_relevant': [PolicyDomain.ENERGY, PolicyDomain.INDUSTRIAL, PolicyDomain.FOREIGN]
            })
        
        # Trade diversification opportunity
        trade_war_intensity = self._perceive_geopolitical_environment(agent_states)['trade_war_intensity']
        if trade_war_intensity > 0.5:
            opportunities.append({
                'type': 'trade_diversification',
                'potential_benefit': 0.7,
                'time_sensitivity': 0.8,
                'policy_domains_relevant': [PolicyDomain.TRADE, PolicyDomain.FOREIGN, PolicyDomain.INDUSTRIAL]
            })
        
        return opportunities
    
    def decide(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make comprehensive policy decisions across multiple domains.
        
        Args:
            perception: Current perception of environment
            
        Returns:
            Policy decisions and actions to implement
        """
        # Analyze current situation
        situation_assessment = self._assess_strategic_situation(perception)
        
        # Generate policy options for each domain
        policy_options = self._generate_policy_options(perception, situation_assessment)
        
        # Evaluate and select optimal policy mix
        selected_policies = self._select_optimal_policies(policy_options, situation_assessment)
        
        # Plan implementation strategy
        implementation_plan = self._plan_policy_implementation(selected_policies)
        
        # Prepare diplomatic initiatives
        diplomatic_actions = self._plan_diplomatic_actions(perception, selected_policies)
        
        decision = {
            'policy_changes': selected_policies,
            'implementation_plan': implementation_plan,
            'diplomatic_actions': diplomatic_actions,
            'resource_allocation': self._allocate_government_resources(selected_policies),
            'communication_strategy': self._plan_public_communication(selected_policies),
            'monitoring_framework': self._establish_monitoring_framework(selected_policies)
        }
        
        return decision
    
    def _assess_strategic_situation(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall strategic situation and priorities."""
        crisis_indicators = perception['crisis_indicators']
        opportunities = perception['opportunity_indicators']
        
        # Calculate crisis severity
        total_crisis_severity = sum(indicator['severity'] * indicator['urgency'] 
                                  for indicator in crisis_indicators)
        
        # Calculate opportunity potential
        total_opportunity_potential = sum(opp['potential_benefit'] * opp['time_sensitivity']
                                        for opp in opportunities)
        
        # Determine strategic stance
        if total_crisis_severity > 0.7:
            strategic_stance = "crisis_management"
        elif total_opportunity_potential > 0.8:
            strategic_stance = "opportunity_maximization"
        else:
            strategic_stance = "steady_state_optimization"
        
        # Identify priority policy domains
        priority_domains = set()
        for indicator in crisis_indicators:
            priority_domains.update(indicator['policy_domains_affected'])
        for opp in opportunities:
            priority_domains.update(opp['policy_domains_relevant'])
        
        return {
            'strategic_stance': strategic_stance,
            'crisis_severity': total_crisis_severity,
            'opportunity_potential': total_opportunity_potential,
            'priority_domains': list(priority_domains),
            'resource_pressure': total_crisis_severity + 0.5 * total_opportunity_potential,
            'time_pressure': max([ind['urgency'] for ind in crisis_indicators] + 
                               [opp['time_sensitivity'] for opp in opportunities] + [0.3])
        }
    
    def _generate_policy_options(self, perception: Dict[str, Any], 
                                situation_assessment: Dict[str, Any]) -> Dict[PolicyDomain, List[Dict[str, Any]]]:
        """Generate policy options for each domain."""
        policy_options = {}
        
        # Trade policy options
        policy_options[PolicyDomain.TRADE] = [
            {
                'name': 'trade_liberalization',
                'position_change': 0.3,
                'cost': 2e9,
                'benefit': 8e9,
                'political_feasibility': 0.6,
                'international_support': 0.8,
                'implementation_time': 18
            },
            {
                'name': 'selective_protectionism', 
                'position_change': -0.4,
                'cost': 5e9,
                'benefit': 3e9,
                'political_feasibility': 0.7,
                'international_support': 0.3,
                'implementation_time': 12
            },
            {
                'name': 'trade_diversification',
                'position_change': 0.1,
                'cost': 3e9,
                'benefit': 6e9,
                'political_feasibility': 0.8,
                'international_support': 0.7,
                'implementation_time': 24
            }
        ]
        
        # Industrial policy options
        policy_options[PolicyDomain.INDUSTRIAL] = [
            {
                'name': 'domestic_semiconductor_investment',
                'position_change': 0.5,
                'cost': 50e9,
                'benefit': 120e9,
                'political_feasibility': 0.8,
                'international_support': 0.4,
                'implementation_time': 60
            },
            {
                'name': 'datacenter_infrastructure_program',
                'position_change': 0.3,
                'cost': 25e9,
                'benefit': 60e9,
                'political_feasibility': 0.9,
                'international_support': 0.6,
                'implementation_time': 36
            },
            {
                'name': 'advanced_manufacturing_initiative',
                'position_change': 0.4,
                'cost': 30e9,
                'benefit': 80e9,
                'political_feasibility': 0.7,
                'international_support': 0.5,
                'implementation_time': 48
            }
        ]
        
        # Energy policy options
        policy_options[PolicyDomain.ENERGY] = [
            {
                'name': 'renewable_energy_acceleration',
                'position_change': 0.4,
                'cost': 100e9,
                'benefit': 180e9,
                'political_feasibility': 0.6,
                'international_support': 0.9,
                'implementation_time': 72
            },
            {
                'name': 'nuclear_energy_expansion',
                'position_change': 0.2,
                'cost': 80e9,
                'benefit': 150e9,
                'political_feasibility': 0.4,
                'international_support': 0.6,
                'implementation_time': 96
            },
            {
                'name': 'energy_efficiency_program',
                'position_change': 0.3,
                'cost': 20e9,
                'benefit': 40e9,
                'political_feasibility': 0.8,
                'international_support': 0.8,
                'implementation_time': 24
            }
        ]
        
        # Technology policy options
        policy_options[PolicyDomain.TECHNOLOGY] = [
            {
                'name': 'ai_research_initiative',
                'position_change': 0.3,
                'cost': 15e9,
                'benefit': 50e9,
                'political_feasibility': 0.8,
                'international_support': 0.5,
                'implementation_time': 36
            },
            {
                'name': 'technology_export_controls',
                'position_change': -0.5,
                'cost': 5e9,
                'benefit': 10e9,
                'political_feasibility': 0.6,
                'international_support': 0.2,
                'implementation_time': 6
            },
            {
                'name': 'international_tech_cooperation',
                'position_change': 0.4,
                'cost': 8e9,
                'benefit': 25e9,
                'political_feasibility': 0.7,
                'international_support': 0.9,
                'implementation_time': 24
            }
        ]
        
        return policy_options
    
    def _select_optimal_policies(self, policy_options: Dict[PolicyDomain, List[Dict[str, Any]]],
                                situation_assessment: Dict[str, Any]) -> Dict[PolicyDomain, Dict[str, Any]]:
        """Select optimal policy mix using multi-criteria decision analysis."""
        selected_policies = {}
        
        # Weight criteria based on strategic situation
        if situation_assessment['strategic_stance'] == "crisis_management":
            criteria_weights = {
                'benefit': 0.3, 'cost': 0.2, 'political_feasibility': 0.3, 
                'implementation_time': 0.2  # Fast implementation crucial in crisis
            }
        elif situation_assessment['strategic_stance'] == "opportunity_maximization":
            criteria_weights = {
                'benefit': 0.4, 'cost': 0.15, 'political_feasibility': 0.25,
                'implementation_time': 0.2
            }
        else:  # steady_state_optimization
            criteria_weights = {
                'benefit': 0.35, 'cost': 0.25, 'political_feasibility': 0.25,
                'implementation_time': 0.15
            }
        
        for domain, options in policy_options.items():
            if domain in situation_assessment['priority_domains'] or len(options) == 0:
                # Use weighted scoring for priority domains
                best_option = None
                best_score = -float('inf')
                
                for option in options:
                    # Normalize metrics to 0-1 scale
                    benefit_score = min(1.0, option['benefit'] / 100e9)
                    cost_score = 1.0 - min(1.0, option['cost'] / 100e9)  # Lower cost is better
                    feasibility_score = option['political_feasibility']
                    time_score = 1.0 - min(1.0, option['implementation_time'] / 120)  # Faster is better
                    
                    total_score = (criteria_weights['benefit'] * benefit_score +
                                 criteria_weights['cost'] * cost_score +
                                 criteria_weights['political_feasibility'] * feasibility_score +
                                 criteria_weights['implementation_time'] * time_score)
                    
                    if total_score > best_score:
                        best_score = total_score
                        best_option = option
                
                if best_option:
                    selected_policies[domain] = best_option
        
        return selected_policies

    def _plan_policy_implementation(self, selected_policies: Dict[PolicyDomain, Dict[str, Any]]) -> Dict[str, Any]:
        """Plan policy implementation strategy."""
        implementation_plan = {}
        
        for domain, policy in selected_policies.items():
            implementation_plan[domain] = {
                'policy_name': policy['name'],
                'position_change': policy['position_change'],
                'cost': policy['cost'],
                'benefit': policy['benefit'],
                'political_feasibility': policy['political_feasibility'],
                'international_support': policy['international_support'],
                'implementation_time': policy['implementation_time'],
                'resource_allocation': self._allocate_government_resources([policy])
            }
        
        return implementation_plan

    def _plan_diplomatic_actions(self, perception: Dict[str, Any], selected_policies: Dict[PolicyDomain, Dict[str, Any]]) -> Dict[str, Any]:
        """Plan diplomatic initiatives based on policy decisions."""
        diplomatic_actions = {}
        
        for domain, policy in selected_policies.items():
            if policy['international_support'] > 0.5:
                partner_nations = self.alliance_memberships if domain in [PolicyDomain.TRADE, PolicyDomain.ENERGY, PolicyDomain.TECHNOLOGY] else []
                for partner in partner_nations:
                    diplomatic_actions[f"{self.agent_id} -> {partner}"] = {
                        'type': 'diplomatic_initiative',
                        'policy_domain': domain,
                        'policy_name': policy['name'],
                        'partner_nation': partner
                    }
        
        return diplomatic_actions

    def _allocate_government_resources(self, selected_policies: Dict[PolicyDomain, Dict[str, Any]]) -> Dict[str, float]:
        """Allocate government resources based on policy decisions."""
        resources = {}
        
        for domain, policy in selected_policies.items():
            resources[domain] = policy['cost']
        
        return resources

    def _plan_public_communication(self, selected_policies: Dict[PolicyDomain, Dict[str, Any]]) -> Dict[str, Any]:
        """Plan public communication strategy based on policy decisions."""
        communication_strategy = {}
        
        for domain, policy in selected_policies.items():
            communication_strategy[domain] = {
                'type': 'public_communication',
                'policy_name': policy['name'],
                'policy_domain': domain,
                'message': f"We are implementing the policy: {policy['name']} to improve {domain.value} policy."
            }
        
        return communication_strategy

    def _establish_monitoring_framework(self, selected_policies: Dict[PolicyDomain, Dict[str, Any]]) -> Dict[str, Any]:
        """Establish monitoring framework for policy implementation."""
        monitoring_framework = {}
        
        for domain, policy in selected_policies.items():
            monitoring_framework[domain] = {
                'type': 'policy_monitoring',
                'policy_name': policy['name'],
                'policy_domain': domain,
                'monitoring_frequency': 'quarterly',
                'performance_metrics': []
            }
        
        return monitoring_framework
    
    def act(self, decision: Dict[str, Any], market_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute policy decisions and update nation state.
        
        Args:
            decision: Policy decisions from decide() method
            market_state: Current market conditions
            
        Returns:
            Policy actions taken and their effects
        """
        actions_taken = {
            'policy_implementations': {},
            'diplomatic_initiatives': [],
            'resource_expenditures': {},
            'state_changes': {},
            'international_effects': {},
            'domestic_effects': {}
        }
        
        # Implement selected policies
        for domain, policy in decision.get('policy_changes', {}).items():
            success_probability = self._calculate_implementation_success(policy, market_state)
            
            if np.random.random() < success_probability:
                # Successfully implement policy
                self._update_policy_position(domain, policy)
                self._spend_government_resources(policy['cost'])
                
                actions_taken['policy_implementations'][domain] = {
                    'policy_name': policy['name'],
                    'implementation_status': 'successful',
                    'cost_incurred': policy['cost'],
                    'expected_benefit': policy['benefit'],
                    'domestic_support_change': self._calculate_domestic_support_change(policy),
                    'international_reputation_change': self._calculate_reputation_change(policy)
                }
                
                # Update nation state based on policy
                self._apply_policy_effects(policy, domain)
                
            else:
                # Failed implementation
                actions_taken['policy_implementations'][domain] = {
                    'policy_name': policy['name'],
                    'implementation_status': 'failed',
                    'cost_incurred': policy['cost'] * 0.3,  # Partial costs incurred
                    'failure_reason': self._determine_failure_reason(policy, market_state),
                    'political_cost': 0.1  # Political capital lost
                }
        
        # Execute diplomatic initiatives
        for initiative_id, initiative in decision.get('diplomatic_actions', {}).items():
            diplomatic_success = self._execute_diplomatic_initiative(initiative)
            actions_taken['diplomatic_initiatives'].append({
                'initiative_id': initiative_id,
                'success': diplomatic_success,
                'partner_nation': initiative.get('partner_nation'),
                'policy_domain': initiative.get('policy_domain'),
                'bilateral_relationship_change': 0.1 if diplomatic_success else -0.05
            })
        
        # Update bilateral relationships
        self._update_diplomatic_relationships(actions_taken['diplomatic_initiatives'])
        
        # Calculate economic effects
        economic_effects = self._calculate_economic_effects(decision['policy_changes'])
        actions_taken['state_changes']['economic'] = economic_effects
        
        # Calculate geopolitical effects
        geopolitical_effects = self._calculate_geopolitical_effects(decision['policy_changes'])
        actions_taken['international_effects'] = geopolitical_effects
        
        # Update approval ratings and political capital
        self._update_political_standing(actions_taken)
        
        # Record performance metrics
        self._record_performance_metrics(actions_taken)
        
        return actions_taken
    
    def _calculate_implementation_success(self, policy: Dict[str, Any], 
                                        market_state: Dict[str, Any]) -> float:
        """Calculate probability of successful policy implementation."""
        base_success = policy['political_feasibility']
        
        # Adjust for institutional constraints
        institutional_penalty = sum(self.institutional_constraints.values()) / len(self.institutional_constraints)
        adjusted_success = base_success * (1 - institutional_penalty * 0.3)
        
        # Adjust for economic conditions
        economic_conditions = self._assess_economic_favorability(market_state)
        adjusted_success *= (0.7 + 0.3 * economic_conditions)
        
        # Adjust for political capital
        political_capital = getattr(self.state, 'political_capital', 0.7)
        adjusted_success *= (0.8 + 0.2 * political_capital)
        
        return max(0.1, min(0.95, adjusted_success))
    
    def _assess_economic_favorability(self, market_state: Dict[str, Any]) -> float:
        """Assess how favorable economic conditions are for policy implementation."""
        gdp_growth = self.state.gdp_growth_rate
        unemployment = self.state.unemployment_rate
        inflation = self.state.inflation_rate
        
        # Higher growth, lower unemployment and moderate inflation are favorable
        growth_score = min(1.0, max(0.0, (gdp_growth + 0.02) / 0.06))  # -2% to 4% range
        unemployment_score = max(0.0, (0.15 - unemployment) / 0.15)  # 0% to 15% range
        inflation_score = max(0.0, 1.0 - abs(inflation - 0.02) / 0.05)  # Optimal around 2%
        
        return (growth_score + unemployment_score + inflation_score) / 3
    
    def _update_policy_position(self, domain: PolicyDomain, policy: Dict[str, Any]) -> None:
        """Update policy position based on implemented policy."""
        if domain in self.policy_positions:
            current_position = self.policy_positions[domain]
            new_value = current_position.position_value + policy['position_change']
            new_value = max(-1.0, min(1.0, new_value))  # Clamp to valid range
            
            self.policy_positions[domain].position_value = new_value
            self.policy_positions[domain].confidence = min(1.0, current_position.confidence + 0.1)
    
    def _spend_government_resources(self, amount: float) -> None:
        """Spend government resources (update fiscal state)."""
        # Simplified fiscal impact
        self.state.debt_to_gdp_ratio += (amount / self.state.gdp) * 0.8  # Not all spending increases debt
    
    def _calculate_domestic_support_change(self, policy: Dict[str, Any]) -> float:
        """Calculate change in domestic political support."""
        base_change = (policy['political_feasibility'] - 0.5) * 0.2
        
        # Add noise for political uncertainty
        noise = np.random.normal(0, 0.05)
        
        return base_change + noise
    
    def _calculate_reputation_change(self, policy: Dict[str, Any]) -> float:
        """Calculate change in international reputation."""
        international_support = policy.get('international_support', 0.5)
        base_change = (international_support - 0.5) * 0.15
        
        return base_change
    
    def _apply_policy_effects(self, policy: Dict[str, Any], domain: PolicyDomain) -> None:
        """Apply policy effects to nation state."""
        policy_name = policy['name']
        
        # Apply domain-specific effects
        if domain == PolicyDomain.INDUSTRIAL:
            if 'semiconductor' in policy_name:
                self.state.semiconductor_manufacturing_capacity *= 1.2
                self.state.technology_competitiveness_index += 0.05
            elif 'datacenter' in policy_name:
                self.state.datacenter_capacity_mw *= 1.3
            elif 'manufacturing' in policy_name:
                self.state.gdp_growth_rate += 0.005  # 0.5% additional growth
        
        elif domain == PolicyDomain.ENERGY:
            if 'renewable' in policy_name:
                self.state.renewable_energy_share += 0.1
                self.state.energy_independence_ratio += 0.05
                self.state.carbon_emissions_per_capita *= 0.95
            elif 'nuclear' in policy_name:
                self.state.energy_independence_ratio += 0.15
                self.state.carbon_emissions_per_capita *= 0.9
            elif 'efficiency' in policy_name:
                self.state.carbon_emissions_per_capita *= 0.98
        
        elif domain == PolicyDomain.TECHNOLOGY:
            if 'ai_research' in policy_name:
                self.state.technology_competitiveness_index += 0.08
            elif 'export_controls' in policy_name:
                self.state.technology_competitiveness_index += 0.02  # Protect domestic tech
            elif 'cooperation' in policy_name:
                self.state.soft_power_index += 0.05
        
        elif domain == PolicyDomain.TRADE:
            if 'liberalization' in policy_name:
                self.state.current_account_balance += self.state.gdp * 0.01
                self.state.gdp_growth_rate += 0.003
            elif 'protectionism' in policy_name:
                self.state.current_account_balance += self.state.gdp * 0.005
                self.state.unemployment_rate *= 0.98  # Protect domestic jobs
            elif 'diversification' in policy_name:
                # Reduce dependence risks without major economic changes
                pass
    
    def _determine_failure_reason(self, policy: Dict[str, Any], 
                                 market_state: Dict[str, Any]) -> str:
        """Determine reason for policy implementation failure."""
        if policy['political_feasibility'] < 0.3:
            return "insufficient_political_support"
        elif policy['cost'] > self.state.gdp * 0.05:  # More than 5% of GDP
            return "insufficient_fiscal_resources"
        elif market_state.get('economic_crisis', False):
            return "unfavorable_economic_conditions"
        else:
            return "implementation_challenges"
    
    def _execute_diplomatic_initiative(self, initiative: Dict[str, Any]) -> bool:
        """Execute diplomatic initiative and return success."""
        # Success probability based on relationship quality and policy alignment
        partner_nation = initiative.get('partner_nation')
        
        if partner_nation in self.diplomatic_relationships:
            relationship_quality = self.diplomatic_relationships[partner_nation].relationship_quality
            base_success = 0.5 + 0.3 * relationship_quality
        else:
            base_success = 0.5  # Neutral relationship
        
        # Adjust for alliance membership
        if partner_nation in self.alliance_memberships:
            base_success += 0.2
        
        return np.random.random() < base_success
    
    def _update_diplomatic_relationships(self, diplomatic_results: List[Dict[str, Any]]) -> None:
        """Update bilateral diplomatic relationships based on initiative results."""
        for result in diplomatic_results:
            partner = result.get('partner_nation')
            if partner and partner in self.diplomatic_relationships:
                relationship_change = result['bilateral_relationship_change']
                current_quality = self.diplomatic_relationships[partner].relationship_quality
                new_quality = max(-1.0, min(1.0, current_quality + relationship_change))
                self.diplomatic_relationships[partner].relationship_quality = new_quality
    
    def _calculate_economic_effects(self, policies: Dict[PolicyDomain, Dict[str, Any]]) -> Dict[str, float]:
        """Calculate aggregate economic effects of all policies."""
        total_cost = sum(policy['cost'] for policy in policies.values())
        total_benefit = sum(policy['benefit'] for policy in policies.values())
        
        # Calculate net fiscal impact
        fiscal_impact = -total_cost + total_benefit * 0.3  # Only 30% of benefits are immediate
        
        # Calculate GDP impact
        gdp_multiplier = 1.5  # Fiscal multiplier
        gdp_impact = fiscal_impact * gdp_multiplier / self.state.gdp
        
        # Calculate employment impact
        employment_elasticity = -0.3  # 1% GDP growth reduces unemployment by 0.3%
        unemployment_impact = gdp_impact * employment_elasticity
        
        return {
            'gdp_impact': gdp_impact,
            'unemployment_impact': unemployment_impact,
            'fiscal_impact': fiscal_impact / self.state.gdp,  # As % of GDP
            'debt_impact': total_cost / self.state.gdp,
            'long_term_growth_impact': sum(policy['benefit'] for policy in policies.values()) * 0.1 / self.state.gdp
        }
    
    def _calculate_geopolitical_effects(self, policies: Dict[PolicyDomain, Dict[str, Any]]) -> Dict[str, float]:
        """Calculate geopolitical effects of policies."""
        effects = {
            'alliance_relations_change': 0.0,
            'rival_relations_change': 0.0,
            'soft_power_change': 0.0,
            'hard_power_change': 0.0,
            'trade_relations_change': 0.0
        }
        
        for domain, policy in policies.items():
            international_support = policy.get('international_support', 0.5)
            
            if domain == PolicyDomain.FOREIGN:
                effects['alliance_relations_change'] += (international_support - 0.5) * 0.2
            elif domain == PolicyDomain.TRADE:
                effects['trade_relations_change'] += (international_support - 0.5) * 0.3
            elif domain == PolicyDomain.TECHNOLOGY:
                if 'export_controls' in policy['name']:
                    effects['rival_relations_change'] -= 0.1
                    effects['hard_power_change'] += 0.05
                else:
                    effects['soft_power_change'] += 0.05
            elif domain == PolicyDomain.ENERGY:
                if 'renewable' in policy['name']:
                    effects['soft_power_change'] += 0.03
        
        return effects
    
    def _update_political_standing(self, actions_taken: Dict[str, Any]) -> None:
        """Update political approval ratings and capital."""
        # Calculate approval rating change
        successful_policies = sum(1 for impl in actions_taken['policy_implementations'].values() 
                                if impl['implementation_status'] == 'successful')
        failed_policies = sum(1 for impl in actions_taken['policy_implementations'].values() 
                            if impl['implementation_status'] == 'failed')
        
        approval_change = successful_policies * 0.02 - failed_policies * 0.05
        
        # Add economic performance effect
        economic_effects = actions_taken.get('state_changes', {}).get('economic', {})
        gdp_impact = economic_effects.get('gdp_impact', 0)
        approval_change += gdp_impact * 2  # Strong correlation between growth and approval
        
        # Update political capital (simplified)
        if not hasattr(self.state, 'political_capital'):
            self.state.political_capital = 0.7
        
        self.state.political_capital = max(0.1, min(1.0, 
            self.state.political_capital + approval_change))
        
        # Record in history
        self.approval_rating_history.append({
            'time_step': len(self.approval_rating_history),
            'approval_rating': self.state.political_capital,
            'approval_change': approval_change
        })
    
    def _record_performance_metrics(self, actions_taken: Dict[str, Any]) -> None:
        """Record policy performance metrics for learning."""
        performance_record = {
            'time_step': len(self.policy_outcomes_history),
            'policies_implemented': list(actions_taken['policy_implementations'].keys()),
            'implementation_success_rate': len([impl for impl in actions_taken['policy_implementations'].values() 
                                              if impl['implementation_status'] == 'successful']) / 
                                            max(1, len(actions_taken['policy_implementations'])),
            'total_cost': sum(impl['cost_incurred'] for impl in actions_taken['policy_implementations'].values()),
            'expected_total_benefit': sum(impl.get('expected_benefit', 0) 
                                        for impl in actions_taken['policy_implementations'].values()),
            'diplomatic_success_rate': len([init for init in actions_taken['diplomatic_initiatives'] 
                                          if init['success']]) / 
                                     max(1, len(actions_taken['diplomatic_initiatives'])),
            'economic_impact': actions_taken.get('state_changes', {}).get('economic', {}),
            'geopolitical_impact': actions_taken.get('international_effects', {})
        }
        
        self.policy_outcomes_history.append(performance_record)
    
    def learn_from_outcomes(self, market_feedback: Dict[str, Any]) -> None:
        """Learn from policy outcomes and adjust future decision-making."""
        if len(self.policy_outcomes_history) < 2:
            return  # Need at least 2 periods to learn
        
        recent_performance = self.policy_outcomes_history[-1]
        
        # Adjust policy learning based on success rates
        success_rate = recent_performance['implementation_success_rate']
        if success_rate > 0.7:
            # High success - can be more ambitious
            self.policy_learning_rate = min(0.2, self.policy_learning_rate * 1.1)
        elif success_rate < 0.3:
            # Low success - be more conservative
            self.policy_learning_rate = max(0.05, self.policy_learning_rate * 0.9)
        
        # Adjust institutional constraint weights based on failure patterns
        failed_policies = [impl for impl in self.policy_outcomes_history[-1]['policies_implemented']
                          if any(p['implementation_status'] == 'failed' 
                                for p in self.policy_outcomes_history[-1].values() 
                                if isinstance(p, dict) and 'implementation_status' in p)]
        
        if len(failed_policies) > 0:
            # Increase weight on political feasibility if failures due to political reasons
            for domain in self.policy_positions:
                if domain.value in failed_policies:
                    self.policy_positions[domain].confidence *= 0.95
        
        # Update strategic objectives based on performance
        economic_impact = recent_performance.get('economic_impact', {})
        if economic_impact.get('gdp_impact', 0) > 0.01:  # Strong positive growth
            self.strategic_objectives['economic_growth'] = min(0.4, 
                self.strategic_objectives['economic_growth'] + 0.02)
        
        geopolitical_impact = recent_performance.get('geopolitical_impact', {})
        if geopolitical_impact.get('alliance_relations_change', 0) > 0.1:
            self.strategic_objectives['international_influence'] = min(0.2,
                self.strategic_objectives['international_influence'] + 0.01)