"""
Welfare Economics Analysis for Datacenter Infrastructure Investment.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from scipy.optimize import minimize, linprog
from scipy.integrate import quad
import logging

from ..core.config import get_config


@dataclass
class WelfareComponents:
    """Components of economic welfare analysis."""
    consumer_surplus: float
    producer_surplus: float
    government_revenue: float
    deadweight_loss: float
    total_welfare: float
    distributional_impact: Dict[str, float]
    regional_effects: Dict[str, float]


@dataclass
class MarketEquilibrium:
    """Market equilibrium state."""
    price: float
    quantity: float
    consumer_surplus: float
    producer_surplus: float
    elasticity_demand: float
    elasticity_supply: float


class DemandCurve:
    """Represents demand curve for datacenter services."""
    
    def __init__(self, intercept: float, slope: float, elasticity: float, 
                 income_effect: float = 0.5, substitute_effect: float = 0.3):
        """
        Initialize demand curve.
        
        Args:
            intercept: Price intercept (maximum willingness to pay)
            slope: Slope of linear demand curve
            elasticity: Price elasticity of demand
            income_effect: Income elasticity
            substitute_effect: Cross-price elasticity with substitutes
        """
        self.intercept = intercept
        self.slope = slope
        self.elasticity = elasticity
        self.income_effect = income_effect
        self.substitute_effect = substitute_effect
        
    def quantity_demanded(self, price: float, income: float = 1.0, 
                         substitute_price: float = 1.0) -> float:
        """Calculate quantity demanded at given price."""
        # Base linear demand
        base_quantity = max(0, (self.intercept - price) / self.slope)
        
        # Income effect
        income_adjustment = (income ** self.income_effect)
        
        # Substitute effect
        substitute_adjustment = (substitute_price ** self.substitute_effect)
        
        return base_quantity * income_adjustment * substitute_adjustment
    
    def inverse_demand(self, quantity: float, income: float = 1.0, 
                      substitute_price: float = 1.0) -> float:
        """Calculate price for given quantity (inverse demand)."""
        adjusted_quantity = quantity / ((income ** self.income_effect) * 
                                      (substitute_price ** self.substitute_effect))
        return max(0, self.intercept - self.slope * adjusted_quantity)
    
    def consumer_surplus(self, equilibrium_price: float, equilibrium_quantity: float,
                        income: float = 1.0, substitute_price: float = 1.0) -> float:
        """Calculate consumer surplus."""
        # Area under demand curve above equilibrium price
        max_price = self.inverse_demand(0, income, substitute_price)
        
        # Triangular approximation for linear demand
        surplus = 0.5 * (max_price - equilibrium_price) * equilibrium_quantity
        
        return max(0, surplus)


class SupplyCurve:
    """Represents supply curve for datacenter capacity."""
    
    def __init__(self, marginal_cost_base: float, capacity_constraint: float,
                 construction_cost: float, operational_cost: float,
                 learning_rate: float = 0.1):
        """
        Initialize supply curve.
        
        Args:
            marginal_cost_base: Base marginal cost
            capacity_constraint: Maximum capacity constraint
            construction_cost: Fixed construction cost per unit
            operational_cost: Variable operational cost
            learning_rate: Learning curve effect
        """
        self.marginal_cost_base = marginal_cost_base
        self.capacity_constraint = capacity_constraint
        self.construction_cost = construction_cost
        self.operational_cost = operational_cost
        self.learning_rate = learning_rate
        self.cumulative_production = 0.0
    
    def marginal_cost(self, quantity: float, time_period: float = 0) -> float:
        """Calculate marginal cost at given quantity."""
        # Base marginal cost with capacity constraints
        if quantity >= self.capacity_constraint:
            return np.inf
        
        # Rising marginal cost as capacity utilization increases
        utilization_factor = quantity / self.capacity_constraint
        congestion_multiplier = 1 + (utilization_factor ** 2)
        
        # Learning curve effect
        learning_multiplier = (1 - self.learning_rate) ** self.cumulative_production
        
        # Time-based cost inflation
        inflation_factor = 1.02 ** time_period  # 2% annual inflation
        
        marginal_cost = (self.marginal_cost_base * congestion_multiplier * 
                        learning_multiplier * inflation_factor + self.operational_cost)
        
        return marginal_cost
    
    def quantity_supplied(self, price: float, time_period: float = 0) -> float:
        """Calculate quantity supplied at given price."""
        # Binary search to find quantity where MC = price
        low, high = 0, self.capacity_constraint
        tolerance = 1e-6
        
        while high - low > tolerance:
            mid = (low + high) / 2
            mc = self.marginal_cost(mid, time_period)
            
            if mc < price:
                low = mid
            else:
                high = mid
        
        return low
    
    def producer_surplus(self, equilibrium_price: float, equilibrium_quantity: float,
                        time_period: float = 0) -> float:
        """Calculate producer surplus."""
        # Integrate (price - MC) from 0 to equilibrium quantity
        def integrand(q):
            return equilibrium_price - self.marginal_cost(q, time_period)
        
        surplus, _ = quad(integrand, 0, equilibrium_quantity)
        return max(0, surplus)


class WelfareAnalyzer:
    """Comprehensive welfare economics analyzer."""
    
    def __init__(self):
        """Initialize welfare analyzer."""
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        
        # Market structure parameters
        self.regional_markets = {
            'North_America': {
                'demand_intercept': 1000,
                'demand_slope': 0.5,
                'elasticity': -1.2,
                'capacity_constraint': 10000,
                'population': 580e6,
                'gdp_per_capita': 65000
            },
            'Europe': {
                'demand_intercept': 800,
                'demand_slope': 0.6,
                'elasticity': -1.5,
                'capacity_constraint': 6000,
                'population': 750e6,
                'gdp_per_capita': 45000
            },
            'Asia_Pacific': {
                'demand_intercept': 600,
                'demand_slope': 0.4,
                'elasticity': -1.8,
                'capacity_constraint': 8000,
                'population': 4600e6,
                'gdp_per_capita': 25000
            },
            'China': {
                'demand_intercept': 500,
                'demand_slope': 0.3,
                'elasticity': -2.0,
                'capacity_constraint': 12000,
                'population': 1400e6,
                'gdp_per_capita': 12000
            }
        }
        
        # Initialize demand and supply curves for each region
        self.demand_curves = {}
        self.supply_curves = {}
        
        for region, params in self.regional_markets.items():
            self.demand_curves[region] = DemandCurve(
                intercept=params['demand_intercept'],
                slope=params['demand_slope'],
                elasticity=params['elasticity']
            )
            
            self.supply_curves[region] = SupplyCurve(
                marginal_cost_base=100,
                capacity_constraint=params['capacity_constraint'],
                construction_cost=1000,
                operational_cost=50
            )
    
    def calculate_market_equilibrium(self, region: str, 
                                   external_factors: Optional[Dict[str, float]] = None) -> MarketEquilibrium:
        """Calculate market equilibrium for a specific region."""
        demand = self.demand_curves[region]
        supply = self.supply_curves[region]
        
        # External factors (tariffs, subsidies, regulations)
        if external_factors is None:
            external_factors = {}
        
        tariff_rate = external_factors.get('tariff_rate', 0.0)
        subsidy_rate = external_factors.get('subsidy_rate', 0.0)
        carbon_tax = external_factors.get('carbon_tax', 0.0)
        
        # Find equilibrium through iteration
        price_range = np.linspace(10, 1000, 1000)
        best_price = None
        min_excess = float('inf')
        
        for price in price_range:
            # Adjust price for external factors
            consumer_price = price * (1 + tariff_rate) + carbon_tax
            producer_price = price * (1 + subsidy_rate)
            
            quantity_demanded = demand.quantity_demanded(consumer_price)
            quantity_supplied = supply.quantity_supplied(producer_price)
            
            excess_demand = abs(quantity_demanded - quantity_supplied)
            
            if excess_demand < min_excess:
                min_excess = excess_demand
                best_price = price
                equilibrium_quantity = min(quantity_demanded, quantity_supplied)
        
        # Calculate surpluses
        consumer_price = best_price * (1 + tariff_rate) + carbon_tax
        producer_price = best_price * (1 + subsidy_rate)
        
        consumer_surplus = demand.consumer_surplus(consumer_price, equilibrium_quantity)
        producer_surplus = supply.producer_surplus(producer_price, equilibrium_quantity)
        
        return MarketEquilibrium(
            price=best_price,
            quantity=equilibrium_quantity,
            consumer_surplus=consumer_surplus,
            producer_surplus=producer_surplus,
            elasticity_demand=demand.elasticity,
            elasticity_supply=1.5  # Assumed supply elasticity
        )
    
    def analyze_policy_impact(self, policy_scenario: Dict[str, Any]) -> WelfareComponents:
        """Analyze welfare impact of policy interventions."""
        baseline_welfare = self._calculate_baseline_welfare()
        policy_welfare = self._calculate_policy_welfare(policy_scenario)
        
        return WelfareComponents(
            consumer_surplus=policy_welfare['consumer_surplus'] - baseline_welfare['consumer_surplus'],
            producer_surplus=policy_welfare['producer_surplus'] - baseline_welfare['producer_surplus'],
            government_revenue=policy_welfare['government_revenue'] - baseline_welfare['government_revenue'],
            deadweight_loss=policy_welfare['deadweight_loss'] - baseline_welfare['deadweight_loss'],
            total_welfare=policy_welfare['total_welfare'] - baseline_welfare['total_welfare'],
            distributional_impact=self._calculate_distributional_impact(baseline_welfare, policy_welfare),
            regional_effects=self._calculate_regional_effects(baseline_welfare, policy_welfare)
        )
    
    def _calculate_baseline_welfare(self) -> Dict[str, float]:
        """Calculate baseline welfare without policy interventions."""
        total_consumer_surplus = 0
        total_producer_surplus = 0
        total_government_revenue = 0
        total_deadweight_loss = 0
        
        for region in self.regional_markets.keys():
            equilibrium = self.calculate_market_equilibrium(region)
            
            total_consumer_surplus += equilibrium.consumer_surplus
            total_producer_surplus += equilibrium.producer_surplus
            # No government intervention in baseline
        
        total_welfare = total_consumer_surplus + total_producer_surplus + total_government_revenue
        
        return {
            'consumer_surplus': total_consumer_surplus,
            'producer_surplus': total_producer_surplus,
            'government_revenue': total_government_revenue,
            'deadweight_loss': total_deadweight_loss,
            'total_welfare': total_welfare
        }
    
    def _calculate_policy_welfare(self, policy_scenario: Dict[str, Any]) -> Dict[str, float]:
        """Calculate welfare under specific policy scenario."""
        total_consumer_surplus = 0
        total_producer_surplus = 0
        total_government_revenue = 0
        total_deadweight_loss = 0
        
        for region in self.regional_markets.keys():
            # Apply region-specific policies
            external_factors = self._get_regional_policy_factors(region, policy_scenario)
            equilibrium = self.calculate_market_equilibrium(region, external_factors)
            
            total_consumer_surplus += equilibrium.consumer_surplus
            total_producer_surplus += equilibrium.producer_surplus
            
            # Calculate government revenue from tariffs/taxes
            tariff_revenue = (equilibrium.quantity * equilibrium.price * 
                            external_factors.get('tariff_rate', 0))
            carbon_tax_revenue = (equilibrium.quantity * 
                                external_factors.get('carbon_tax', 0))
            subsidy_cost = (equilibrium.quantity * equilibrium.price * 
                          external_factors.get('subsidy_rate', 0))
            
            region_gov_revenue = tariff_revenue + carbon_tax_revenue - subsidy_cost
            total_government_revenue += region_gov_revenue
            
            # Estimate deadweight loss from price distortions
            baseline_eq = self.calculate_market_equilibrium(region)
            quantity_distortion = abs(equilibrium.quantity - baseline_eq.quantity)
            price_distortion = abs(equilibrium.price - baseline_eq.price)
            deadweight_loss = 0.5 * quantity_distortion * price_distortion
            total_deadweight_loss += deadweight_loss
        
        total_welfare = (total_consumer_surplus + total_producer_surplus + 
                        total_government_revenue - total_deadweight_loss)
        
        return {
            'consumer_surplus': total_consumer_surplus,
            'producer_surplus': total_producer_surplus,
            'government_revenue': total_government_revenue,
            'deadweight_loss': total_deadweight_loss,
            'total_welfare': total_welfare
        }
    
    def _get_regional_policy_factors(self, region: str, 
                                   policy_scenario: Dict[str, Any]) -> Dict[str, float]:
        """Get policy factors for specific region."""
        factors = {}
        
        # Tariff policies
        if 'trade_war' in policy_scenario:
            trade_war_intensity = policy_scenario['trade_war']
            if region == 'China' and trade_war_intensity > 0:
                factors['tariff_rate'] = 0.25 * trade_war_intensity
            elif region in ['North_America', 'Europe'] and trade_war_intensity > 0:
                factors['tariff_rate'] = 0.15 * trade_war_intensity
        
        # Carbon pricing
        if 'carbon_price' in policy_scenario:
            carbon_intensity = {
                'North_America': 400,  # kg CO2/MWh
                'Europe': 300,
                'Asia_Pacific': 500,
                'China': 600
            }
            factors['carbon_tax'] = (policy_scenario['carbon_price'] * 
                                   carbon_intensity.get(region, 450) / 1000)  # $/MWh
        
        # Infrastructure subsidies
        if 'infrastructure_subsidy' in policy_scenario:
            subsidy_rates = {
                'North_America': 0.05,
                'Europe': 0.08,
                'Asia_Pacific': 0.12,
                'China': 0.15
            }
            factors['subsidy_rate'] = (subsidy_rates.get(region, 0.1) * 
                                     policy_scenario['infrastructure_subsidy'])
        
        # Export controls
        if 'export_controls' in policy_scenario:
            if region in ['Asia_Pacific', 'China']:
                # Export controls increase costs (reduced supply)
                factors['supply_restriction'] = 0.2 * policy_scenario['export_controls']
        
        return factors
    
    def _calculate_distributional_impact(self, baseline: Dict[str, float], 
                                       policy: Dict[str, float]) -> Dict[str, float]:
        """Calculate distributional impact across income groups."""
        # Simplified distributional analysis
        # Assumes datacenter services are normal goods with income elasticity > 1
        
        income_groups = ['low_income', 'middle_income', 'high_income']
        income_shares = [0.3, 0.5, 0.2]  # Share of total income
        consumption_shares = [0.1, 0.4, 0.5]  # Share of datacenter services consumption
        
        consumer_surplus_change = policy['consumer_surplus'] - baseline['consumer_surplus']
        
        distributional_impact = {}
        for i, group in enumerate(income_groups):
            # Higher income groups consume more datacenter services
            group_impact = consumer_surplus_change * consumption_shares[i]
            # Normalize by income share to get per-dollar impact
            distributional_impact[group] = group_impact / income_shares[i]
        
        # Calculate Gini coefficient change
        impacts = list(distributional_impact.values())
        distributional_impact['gini_change'] = self._calculate_gini_change(impacts)
        
        return distributional_impact
    
    def _calculate_regional_effects(self, baseline: Dict[str, float], 
                                  policy: Dict[str, float]) -> Dict[str, float]:
        """Calculate regional effects of policy changes."""
        regional_effects = {}
        
        for region, params in self.regional_markets.items():
            # Calculate regional welfare change
            baseline_eq = self.calculate_market_equilibrium(region)
            policy_eq = self.calculate_market_equilibrium(region, 
                self._get_regional_policy_factors(region, {'carbon_price': 100}))
            
            welfare_change = ((policy_eq.consumer_surplus + policy_eq.producer_surplus) - 
                            (baseline_eq.consumer_surplus + baseline_eq.producer_surplus))
            
            # Normalize by regional GDP
            regional_gdp = params['population'] * params['gdp_per_capita']
            regional_effects[region] = welfare_change / regional_gdp
        
        return regional_effects
    
    def _calculate_gini_change(self, impacts: List[float]) -> float:
        """Calculate change in Gini coefficient from policy impacts."""
        # Simplified Gini calculation
        sorted_impacts = sorted(impacts)
        n = len(sorted_impacts)
        
        if n == 0:
            return 0.0
        
        cumulative_sum = np.cumsum(sorted_impacts)
        total_sum = cumulative_sum[-1]
        
        if total_sum == 0:
            return 0.0
        
        gini = (2 * sum((i + 1) * impact for i, impact in enumerate(sorted_impacts)) / 
                (n * total_sum) - (n + 1) / n)
        
        return gini
    
    def calculate_optimal_carbon_price(self, social_cost_carbon: float = 185,
                                     damage_function_params: Dict[str, float] = None) -> float:
        """Calculate socially optimal carbon price using damage function."""
        if damage_function_params is None:
            damage_function_params = {
                'climate_sensitivity': 3.0,  # °C per doubling of CO2
                'damage_coefficient': 0.0023,  # % GDP loss per °C
                'discount_rate': 0.03
            }
        
        # Optimization problem: maximize social welfare - climate damages
        def objective(carbon_price):
            policy_scenario = {'carbon_price': carbon_price[0]}
            welfare_analysis = self.analyze_policy_impact(policy_scenario)
            
            # Calculate climate damages avoided
            emission_reduction = self._estimate_emission_reduction(carbon_price[0])
            climate_damages_avoided = (emission_reduction * social_cost_carbon)
            
            # Net social benefit = welfare change + climate benefits
            net_benefit = welfare_analysis.total_welfare + climate_damages_avoided
            
            return -net_benefit  # Minimize negative of benefit
        
        # Optimize carbon price
        result = minimize(objective, x0=[100], bounds=[(0, 500)], method='L-BFGS-B')
        
        optimal_price = result.x[0] if result.success else 100  # Default fallback
        
        return optimal_price
    
    def _estimate_emission_reduction(self, carbon_price: float) -> float:
        """Estimate CO2 emission reduction from carbon pricing."""
        # Simplified emission reduction function
        # Based on empirical studies of carbon price elasticity
        
        baseline_emissions = 500e6  # tonnes CO2 annually (estimated global datacenter emissions)
        price_elasticity = -0.3  # 1% price increase -> 0.3% emission reduction
        
        # Price effect relative to baseline (assumed $50/tonne baseline)
        baseline_price = 50
        price_ratio = carbon_price / baseline_price
        
        # Calculate percentage reduction
        reduction_percentage = price_elasticity * np.log(price_ratio)
        
        # Convert to absolute reduction
        emission_reduction = baseline_emissions * abs(reduction_percentage)
        
        return emission_reduction
    
    def analyze_infrastructure_investment_efficiency(self, 
                                                   investment_scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze efficiency of different infrastructure investment scenarios."""
        efficiency_results = {}
        
        for i, scenario in enumerate(investment_scenarios):
            scenario_name = scenario.get('name', f'scenario_{i}')
            
            # Calculate welfare impact
            welfare_impact = self.analyze_policy_impact(scenario)
            
            # Calculate cost-benefit ratio
            investment_cost = scenario.get('investment_cost', 0)
            net_benefit = welfare_impact.total_welfare
            
            if investment_cost > 0:
                benefit_cost_ratio = net_benefit / investment_cost
            else:
                benefit_cost_ratio = float('inf') if net_benefit > 0 else 0
            
            # Calculate distributional progressivity
            distributional_score = self._calculate_progressivity_score(
                welfare_impact.distributional_impact)
            
            # Calculate regional balance
            regional_balance_score = self._calculate_regional_balance_score(
                welfare_impact.regional_effects)
            
            efficiency_results[scenario_name] = {
                'net_benefit': net_benefit,
                'benefit_cost_ratio': benefit_cost_ratio,
                'consumer_surplus_change': welfare_impact.consumer_surplus,
                'producer_surplus_change': welfare_impact.producer_surplus,
                'deadweight_loss': welfare_impact.deadweight_loss,
                'distributional_progressivity': distributional_score,
                'regional_balance': regional_balance_score,
                'overall_efficiency_score': self._calculate_overall_efficiency_score(
                    benefit_cost_ratio, distributional_score, regional_balance_score)
            }
        
        return efficiency_results
    
    def _calculate_progressivity_score(self, distributional_impact: Dict[str, float]) -> float:
        """Calculate progressivity score (higher = more progressive)."""
        if 'low_income' not in distributional_impact:
            return 0.0
        
        low_income_impact = distributional_impact.get('low_income', 0)
        high_income_impact = distributional_impact.get('high_income', 0)
        
        # Progressive if low income benefits more per dollar of income
        if high_income_impact != 0:
            progressivity = low_income_impact / high_income_impact
        else:
            progressivity = 1.0 if low_income_impact > 0 else 0.0
        
        # Normalize to 0-1 scale
        return min(1.0, max(0.0, progressivity))
    
    def _calculate_regional_balance_score(self, regional_effects: Dict[str, float]) -> float:
        """Calculate regional balance score (higher = more balanced)."""
        if not regional_effects:
            return 0.0
        
        effects = list(regional_effects.values())
        
        # Balance measured as inverse of coefficient of variation
        if len(effects) > 1 and np.mean(effects) != 0:
            cv = np.std(effects) / abs(np.mean(effects))
            balance_score = 1 / (1 + cv)  # Higher score = more balanced
        else:
            balance_score = 1.0
        
        return balance_score
    
    def _calculate_overall_efficiency_score(self, benefit_cost_ratio: float, 
                                          progressivity: float, regional_balance: float) -> float:
        """Calculate overall efficiency score combining multiple criteria."""
        # Weighted combination of efficiency metrics
        weights = {
            'economic_efficiency': 0.5,
            'distributional_equity': 0.3,
            'regional_balance': 0.2
        }
        
        # Normalize benefit-cost ratio to 0-1 scale
        normalized_bcr = min(1.0, benefit_cost_ratio / 3.0)  # BCR of 3 = perfect score
        
        overall_score = (weights['economic_efficiency'] * normalized_bcr +
                        weights['distributional_equity'] * progressivity +
                        weights['regional_balance'] * regional_balance)
        
        return overall_score
    
    def generate_welfare_report(self, policy_scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive welfare analysis report."""
        report = {
            'executive_summary': {},
            'baseline_analysis': self._calculate_baseline_welfare(),
            'policy_scenarios': {},
            'optimal_policies': {},
            'recommendations': []
        }
        
        # Analyze each policy scenario
        scenario_results = []
        for scenario in policy_scenarios:
            scenario_name = scenario.get('name', 'unnamed_scenario')
            welfare_impact = self.analyze_policy_impact(scenario)
            
            report['policy_scenarios'][scenario_name] = {
                'welfare_components': welfare_impact,
                'efficiency_metrics': self.analyze_infrastructure_investment_efficiency([scenario])
            }
            
            scenario_results.append({
                'name': scenario_name,
                'total_welfare': welfare_impact.total_welfare,
                'progressivity': self._calculate_progressivity_score(welfare_impact.distributional_impact)
            })
        
        # Find optimal policies
        if scenario_results:
            best_welfare = max(scenario_results, key=lambda x: x['total_welfare'])
            most_progressive = max(scenario_results, key=lambda x: x['progressivity'])
            
            report['optimal_policies'] = {
                'highest_welfare': best_welfare['name'],
                'most_progressive': most_progressive['name'],
                'optimal_carbon_price': self.calculate_optimal_carbon_price()
            }
        
        # Generate recommendations
        report['recommendations'] = self._generate_policy_recommendations(scenario_results)
        
        # Executive summary
        report['executive_summary'] = {
            'total_scenarios_analyzed': len(policy_scenarios),
            'baseline_total_welfare': report['baseline_analysis']['total_welfare'],
            'best_scenario_welfare_gain': (best_welfare['total_welfare'] 
                                         if scenario_results else 0),
            'optimal_carbon_price': report['optimal_policies'].get('optimal_carbon_price', 0)
        }
        
        return report
    
    def _generate_policy_recommendations(self, scenario_results: List[Dict[str, Any]]) -> List[str]:
        """Generate policy recommendations based on analysis results."""
        recommendations = []
        
        if not scenario_results:
            return ["Insufficient data for recommendations"]
        
        # Welfare efficiency recommendations
        best_scenario = max(scenario_results, key=lambda x: x['total_welfare'])
        if best_scenario['total_welfare'] > 0:
            recommendations.append(
                f"Implement {best_scenario['name']} scenario for maximum welfare gains of "
                f"${best_scenario['total_welfare']/1e9:.1f}B annually"
            )
        
        # Distributional equity recommendations
        progressive_scenarios = [s for s in scenario_results if s['progressivity'] > 0.6]
        if progressive_scenarios:
            recommendations.append(
                "Consider progressive policy design to ensure equitable distribution of benefits"
            )
        
        # Carbon pricing recommendations
        recommendations.append(
            "Implement gradual carbon price increases with revenue recycling to minimize "
            "distributional impacts"
        )
        
        # Regional development recommendations
        recommendations.append(
            "Coordinate international infrastructure investment to avoid regional imbalances "
            "and trade tensions"
        )
        
        return recommendations 