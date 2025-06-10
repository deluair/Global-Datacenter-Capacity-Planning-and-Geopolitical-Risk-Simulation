"""
International Trade Analysis for Global Datacenter Infrastructure.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
import networkx as nx
from scipy.optimize import minimize, linprog
from scipy.spatial.distance import pdist, squareform
import logging

from ..core.config import get_config


@dataclass
class TradeFlow:
    """Represents bilateral trade flow between countries."""
    exporter: str
    importer: str
    product_category: str
    volume: float  # Physical units
    value: float   # USD value
    tariff_rate: float
    transport_cost: float
    lead_time_days: int
    quality_premium: float


@dataclass
class ComparativeAdvantage:
    """Comparative advantage indicators for a country-product pair."""
    country: str
    product: str
    revealed_comparative_advantage: float  # RCA index
    export_complexity: float
    opportunity_cost: float
    factor_endowments: Dict[str, float]
    technology_level: float


@dataclass
class TechnologyTransfer:
    """Technology transfer between countries."""
    source_country: str
    destination_country: str
    technology_type: str
    transfer_volume: float
    spillover_effects: float
    learning_coefficient: float
    restrictions: List[str]


class GravityModel:
    """Gravity model for predicting bilateral trade flows."""
    
    def __init__(self):
        """Initialize gravity model."""
        self.config = get_config()
        
        # Standard gravity model parameters (calibrated from literature)
        self.distance_elasticity = -1.1
        self.gdp_elasticity_exporter = 0.8
        self.gdp_elasticity_importer = 0.9
        self.population_elasticity = 0.3
        self.tariff_elasticity = -2.5
        self.common_language_effect = 0.4
        self.colonial_history_effect = 0.3
        self.trade_agreement_effect = 0.6
        
        # Country data
        self.country_data = {
            'USA': {
                'gdp': 25.46e12, 'population': 333e6, 'lat': 39.8, 'lon': -98.5,
                'technology_level': 0.95, 'human_capital': 0.88, 'infrastructure_quality': 0.85
            },
            'China': {
                'gdp': 17.73e12, 'population': 1412e6, 'lat': 35.8, 'lon': 104.2,
                'technology_level': 0.75, 'human_capital': 0.65, 'infrastructure_quality': 0.72
            },
            'Germany': {
                'gdp': 4.26e12, 'population': 83e6, 'lat': 51.2, 'lon': 10.5,
                'technology_level': 0.92, 'human_capital': 0.91, 'infrastructure_quality': 0.93
            },
            'Japan': {
                'gdp': 4.94e12, 'population': 125e6, 'lat': 36.2, 'lon': 138.3,
                'technology_level': 0.90, 'human_capital': 0.86, 'infrastructure_quality': 0.89
            },
            'South_Korea': {
                'gdp': 1.81e12, 'population': 52e6, 'lat': 35.9, 'lon': 127.8,
                'technology_level': 0.88, 'human_capital': 0.84, 'infrastructure_quality': 0.86
            },
            'Taiwan': {
                'gdp': 0.79e12, 'population': 23e6, 'lat': 23.7, 'lon': 120.9,
                'technology_level': 0.87, 'human_capital': 0.82, 'infrastructure_quality': 0.83
            },
            'Singapore': {
                'gdp': 0.40e12, 'population': 6e6, 'lat': 1.4, 'lon': 103.8,
                'technology_level': 0.89, 'human_capital': 0.90, 'infrastructure_quality': 0.95
            },
            'Netherlands': {
                'gdp': 0.91e12, 'population': 17e6, 'lat': 52.1, 'lon': 5.3,
                'technology_level': 0.90, 'human_capital': 0.89, 'infrastructure_quality': 0.91
            }
        }
        
        # Calculate distances
        self.distances = self._calculate_distances()
        
        # Bilateral relationships
        self.bilateral_factors = self._initialize_bilateral_factors()
    
    def _calculate_distances(self) -> Dict[Tuple[str, str], float]:
        """Calculate great circle distances between countries."""
        distances = {}
        countries = list(self.country_data.keys())
        
        for i, country1 in enumerate(countries):
            for j, country2 in enumerate(countries):
                if i != j:
                    lat1, lon1 = self.country_data[country1]['lat'], self.country_data[country1]['lon']
                    lat2, lon2 = self.country_data[country2]['lat'], self.country_data[country2]['lon']
                    
                    # Haversine formula
                    dlat = np.radians(lat2 - lat1)
                    dlon = np.radians(lon2 - lon1)
                    a = (np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * 
                         np.cos(np.radians(lat2)) * np.sin(dlon/2)**2)
                    c = 2 * np.arcsin(np.sqrt(a))
                    distance = 6371 * c  # Earth's radius in km
                    
                    distances[(country1, country2)] = distance
        
        return distances
    
    def _initialize_bilateral_factors(self) -> Dict[Tuple[str, str], Dict[str, float]]:
        """Initialize bilateral relationship factors."""
        factors = {}
        countries = list(self.country_data.keys())
        
        # Common language groups
        common_language = {
            ('USA',): ['USA'],
            ('China',): ['China', 'Taiwan'],
            ('Germany', 'Netherlands'): ['Germany', 'Netherlands'],
            ('Japan',): ['Japan'],
            ('South_Korea',): ['South_Korea'],
            ('Singapore',): ['Singapore']
        }
        
        # Trade agreements
        trade_agreements = [
            ('USA', 'South_Korea'),  # KORUS FTA
            ('Germany', 'Netherlands'),  # EU
            ('Japan', 'Singapore'),  # JSEPA
        ]
        
        for country1 in countries:
            for country2 in countries:
                if country1 != country2:
                    factors[(country1, country2)] = {
                        'common_language': self._check_common_language(country1, country2, common_language),
                        'trade_agreement': (country1, country2) in trade_agreements or (country2, country1) in trade_agreements,
                        'colonial_history': False,  # Simplified
                        'technology_similarity': self._calculate_technology_similarity(country1, country2),
                        'political_stability': self._assess_political_relationship(country1, country2)
                    }
        
        return factors
    
    def _check_common_language(self, country1: str, country2: str, 
                              language_groups: Dict[Tuple[str, ...], List[str]]) -> bool:
        """Check if countries share common language."""
        for group_countries in language_groups.values():
            if country1 in group_countries and country2 in group_countries:
                return True
        return False
    
    def _calculate_technology_similarity(self, country1: str, country2: str) -> float:
        """Calculate technology similarity between countries."""
        tech1 = self.country_data[country1]['technology_level']
        tech2 = self.country_data[country2]['technology_level']
        return 1 - abs(tech1 - tech2)
    
    def _assess_political_relationship(self, country1: str, country2: str) -> float:
        """Assess political relationship quality (0-1 scale)."""
        # Simplified political relationship assessment
        tension_pairs = [
            ('USA', 'China'),
            ('China', 'Taiwan'),
            ('South_Korea', 'China')
        ]
        
        if (country1, country2) in tension_pairs or (country2, country1) in tension_pairs:
            return 0.3  # High tension
        else:
            return 0.8  # Normal relations
    
    def predict_trade_flow(self, exporter: str, importer: str, product_category: str,
                          tariff_rate: float = 0.0, scenario_factors: Optional[Dict[str, float]] = None) -> float:
        """Predict bilateral trade flow using gravity model."""
        if exporter == importer:
            return 0.0
        
        if exporter not in self.country_data or importer not in self.country_data:
            return 0.0
        
        # Basic gravity variables
        gdp_exp = self.country_data[exporter]['gdp']
        gdp_imp = self.country_data[importer]['gdp']
        pop_exp = self.country_data[exporter]['population']
        pop_imp = self.country_data[importer]['population']
        distance = self.distances.get((exporter, importer), 10000)
        
        # Bilateral factors
        bilateral = self.bilateral_factors.get((exporter, importer), {})
        
        # Scenario adjustments
        if scenario_factors is None:
            scenario_factors = {}
        
        geopolitical_tension = scenario_factors.get('geopolitical_tension', 0.0)
        trade_war_intensity = scenario_factors.get('trade_war_intensity', 0.0)
        technology_restrictions = scenario_factors.get('technology_restrictions', 0.0)
        
        # Calculate log trade flow
        log_trade = (
            self.gdp_elasticity_exporter * np.log(gdp_exp) +
            self.gdp_elasticity_importer * np.log(gdp_imp) +
            self.population_elasticity * np.log(pop_exp) +
            self.distance_elasticity * np.log(distance) +
            self.tariff_elasticity * tariff_rate
        )
        
        # Bilateral relationship effects
        if bilateral.get('common_language', False):
            log_trade += self.common_language_effect
        
        if bilateral.get('trade_agreement', False):
            log_trade += self.trade_agreement_effect
        
        if bilateral.get('colonial_history', False):
            log_trade += self.colonial_history_effect
        
        # Technology similarity effect (especially important for high-tech products)
        tech_similarity = bilateral.get('technology_similarity', 0.5)
        if product_category in ['semiconductors', 'datacenter_equipment']:
            log_trade += 0.5 * tech_similarity
        
        # Scenario-specific adjustments
        political_stability = bilateral.get('political_stability', 0.8)
        tension_effect = -geopolitical_tension * (1 - political_stability)
        trade_war_effect = -trade_war_intensity * 0.8
        
        # Technology restrictions (especially severe for advanced semiconductors)
        if product_category == 'advanced_semiconductors':
            tech_restriction_effect = -technology_restrictions * 1.5
        else:
            tech_restriction_effect = -technology_restrictions * 0.5
        
        log_trade += tension_effect + trade_war_effect + tech_restriction_effect
        
        # Convert to levels
        predicted_trade = np.exp(log_trade)
        
        # Product-specific scaling factors
        product_scaling = {
            'semiconductors': 1e9,  # $1B base scale
            'datacenter_equipment': 5e8,  # $500M base scale
            'advanced_semiconductors': 2e9,  # $2B base scale
            'renewable_energy_equipment': 3e8  # $300M base scale
        }
        
        scaling_factor = product_scaling.get(product_category, 1e8)
        
        return predicted_trade * scaling_factor


class TradeNetworkAnalyzer:
    """Analyzer for global trade networks and supply chain dependencies."""
    
    def __init__(self):
        """Initialize trade network analyzer."""
        self.config = get_config()
        self.gravity_model = GravityModel()
        self.logger = logging.getLogger(__name__)
        
        # Product categories relevant to datacenter infrastructure
        self.product_categories = [
            'semiconductors',
            'advanced_semiconductors', 
            'datacenter_equipment',
            'renewable_energy_equipment',
            'rare_earth_materials',
            'optical_components'
        ]
        
        # Initialize trade network
        self.trade_network = nx.DiGraph()
        self._build_baseline_network()
    
    def _build_baseline_network(self) -> None:
        """Build baseline trade network from current data."""
        countries = list(self.gravity_model.country_data.keys())
        
        # Add nodes (countries)
        for country in countries:
            self.trade_network.add_node(country, **self.gravity_model.country_data[country])
        
        # Add edges (trade flows)
        for exporter in countries:
            for importer in countries:
                if exporter != importer:
                    total_trade_value = 0
                    product_flows = {}
                    
                    for product in self.product_categories:
                        flow_value = self.gravity_model.predict_trade_flow(
                            exporter, importer, product
                        )
                        product_flows[product] = flow_value
                        total_trade_value += flow_value
                    
                    if total_trade_value > 1e6:  # Minimum $1M threshold
                        self.trade_network.add_edge(
                            exporter, importer,
                            total_value=total_trade_value,
                            product_flows=product_flows
                        )
    
    def calculate_comparative_advantage(self) -> Dict[str, ComparativeAdvantage]:
        """Calculate revealed comparative advantage for each country-product pair."""
        countries = list(self.gravity_model.country_data.keys())
        comparative_advantages = {}
        
        # Calculate total exports by country and product
        country_exports = {country: {} for country in countries}
        product_totals = {product: 0 for product in self.product_categories}
        total_trade = 0
        
        for exporter in countries:
            for importer in countries:
                if exporter != importer and self.trade_network.has_edge(exporter, importer):
                    edge_data = self.trade_network[exporter][importer]
                    product_flows = edge_data.get('product_flows', {})
                    
                    for product, value in product_flows.items():
                        if product not in country_exports[exporter]:
                            country_exports[exporter][product] = 0
                        country_exports[exporter][product] += value
                        product_totals[product] += value
                        total_trade += value
        
        # Calculate RCA indices
        for country in countries:
            country_total_exports = sum(country_exports[country].values())
            
            for product in self.product_categories:
                product_exports = country_exports[country].get(product, 0)
                
                if country_total_exports > 0 and product_totals[product] > 0 and total_trade > 0:
                    # RCA = (Xij/Xi) / (Xwj/Xw)
                    country_product_share = product_exports / country_total_exports
                    world_product_share = product_totals[product] / total_trade
                    
                    if world_product_share > 0:
                        rca = country_product_share / world_product_share
                    else:
                        rca = 0
                else:
                    rca = 0
                
                # Calculate export complexity (simplified)
                country_data = self.gravity_model.country_data[country]
                complexity_factors = {
                    'technology_level': country_data['technology_level'],
                    'human_capital': country_data['human_capital'],
                    'infrastructure_quality': country_data['infrastructure_quality']
                }
                
                export_complexity = np.mean(list(complexity_factors.values()))
                
                # Calculate opportunity cost (simplified)
                opportunity_cost = self._calculate_opportunity_cost(country, product)
                
                comparative_advantages[f"{country}_{product}"] = ComparativeAdvantage(
                    country=country,
                    product=product,
                    revealed_comparative_advantage=rca,
                    export_complexity=export_complexity,
                    opportunity_cost=opportunity_cost,
                    factor_endowments=complexity_factors,
                    technology_level=country_data['technology_level']
                )
        
        return comparative_advantages
    
    def _calculate_opportunity_cost(self, country: str, product: str) -> float:
        """Calculate opportunity cost of specializing in a product."""
        # Simplified opportunity cost based on factor endowments
        country_data = self.gravity_model.country_data[country]
        
        # Product-specific factor requirements
        factor_requirements = {
            'semiconductors': {'technology_level': 0.85, 'human_capital': 0.80, 'infrastructure_quality': 0.75},
            'advanced_semiconductors': {'technology_level': 0.95, 'human_capital': 0.90, 'infrastructure_quality': 0.85},
            'datacenter_equipment': {'technology_level': 0.75, 'human_capital': 0.70, 'infrastructure_quality': 0.80},
            'renewable_energy_equipment': {'technology_level': 0.70, 'human_capital': 0.65, 'infrastructure_quality': 0.75},
            'rare_earth_materials': {'technology_level': 0.50, 'human_capital': 0.40, 'infrastructure_quality': 0.60},
            'optical_components': {'technology_level': 0.80, 'human_capital': 0.75, 'infrastructure_quality': 0.70}
        }
        
        requirements = factor_requirements.get(product, {})
        
        # Calculate capability gap
        capability_gaps = []
        for factor, requirement in requirements.items():
            country_capability = country_data[factor]
            gap = max(0, requirement - country_capability)
            capability_gaps.append(gap)
        
        # Opportunity cost is the average capability gap
        opportunity_cost = np.mean(capability_gaps) if capability_gaps else 0
        
        return opportunity_cost
    
    def analyze_supply_chain_vulnerability(self, scenario_factors: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Analyze supply chain vulnerability under different scenarios."""
        if scenario_factors is None:
            scenario_factors = {}
        
        vulnerability_analysis = {
            'concentration_risk': {},
            'chokepoint_analysis': {},
            'scenario_impacts': {},
            'resilience_metrics': {}
        }
        
        # Concentration risk analysis
        for product in self.product_categories:
            product_suppliers = []
            total_supply = 0
            
            for country in self.trade_network.nodes():
                country_supply = 0
                for neighbor in self.trade_network.successors(country):
                    edge_data = self.trade_network[country][neighbor]
                    product_flows = edge_data.get('product_flows', {})
                    country_supply += product_flows.get(product, 0)
                
                if country_supply > 0:
                    product_suppliers.append((country, country_supply))
                    total_supply += country_supply
            
            # Calculate Herfindahl-Hirschman Index (HHI)
            if total_supply > 0:
                market_shares = [(supply / total_supply) for _, supply in product_suppliers]
                hhi = sum(share**2 for share in market_shares)
            else:
                hhi = 1.0  # Maximum concentration
            
            # Top suppliers concentration
            product_suppliers.sort(key=lambda x: x[1], reverse=True)
            top3_concentration = sum(supply for _, supply in product_suppliers[:3]) / total_supply if total_supply > 0 else 0
            
            vulnerability_analysis['concentration_risk'][product] = {
                'hhi': hhi,
                'top3_concentration': top3_concentration,
                'num_suppliers': len(product_suppliers),
                'largest_supplier': product_suppliers[0][0] if product_suppliers else None,
                'largest_supplier_share': product_suppliers[0][1] / total_supply if product_suppliers and total_supply > 0 else 0
            }
        
        # Chokepoint analysis (critical nodes/edges)
        vulnerability_analysis['chokepoint_analysis'] = self._identify_chokepoints()
        
        # Scenario impact analysis
        scenarios = [
            {'name': 'trade_war', 'geopolitical_tension': 0.8, 'trade_war_intensity': 0.7},
            {'name': 'tech_restrictions', 'technology_restrictions': 0.9, 'geopolitical_tension': 0.6},
            {'name': 'supply_disruption', 'geopolitical_tension': 0.4, 'supply_shock': 0.8}
        ]
        
        for scenario in scenarios:
            scenario_impact = self._calculate_scenario_impact(scenario)
            vulnerability_analysis['scenario_impacts'][scenario['name']] = scenario_impact
        
        # Calculate overall resilience metrics
        vulnerability_analysis['resilience_metrics'] = self._calculate_resilience_metrics()
        
        return vulnerability_analysis
    
    def _identify_chokepoints(self) -> Dict[str, Any]:
        """Identify critical chokepoints in the trade network."""
        chokepoints = {
            'critical_countries': {},
            'critical_trade_routes': {},
            'systemic_risk_score': 0
        }
        
        # Node criticality (country removal impact)
        for country in self.trade_network.nodes():
            # Create network without this country
            temp_network = self.trade_network.copy()
            temp_network.remove_node(country)
            
            # Calculate connectivity impact
            original_edges = self.trade_network.number_of_edges()
            remaining_edges = temp_network.number_of_edges()
            connectivity_impact = (original_edges - remaining_edges) / original_edges if original_edges > 0 else 0
            
            # Calculate trade volume impact
            removed_trade_volume = 0
            for pred in self.trade_network.predecessors(country):
                if self.trade_network.has_edge(pred, country):
                    removed_trade_volume += self.trade_network[pred][country]['total_value']
            for succ in self.trade_network.successors(country):
                if self.trade_network.has_edge(country, succ):
                    removed_trade_volume += self.trade_network[country][succ]['total_value']
            
            total_trade_volume = sum(data['total_value'] for _, _, data in self.trade_network.edges(data=True))
            trade_impact = removed_trade_volume / total_trade_volume if total_trade_volume > 0 else 0
            
            # Overall criticality score
            criticality_score = 0.6 * connectivity_impact + 0.4 * trade_impact
            
            chokepoints['critical_countries'][country] = {
                'criticality_score': criticality_score,
                'connectivity_impact': connectivity_impact,
                'trade_impact': trade_impact
            }
        
        # Edge criticality (trade route removal impact)
        for exporter, importer, edge_data in self.trade_network.edges(data=True):
            route_importance = edge_data['total_value'] / sum(data['total_value'] for _, _, data in self.trade_network.edges(data=True))
            
            # Check if this is the only route for critical products
            product_flows = edge_data.get('product_flows', {})
            critical_product_dependency = 0
            
            for product, flow_value in product_flows.items():
                if product in ['advanced_semiconductors', 'semiconductors']:
                    # Check alternative suppliers
                    alternative_suppliers = 0
                    for alt_exporter in self.trade_network.nodes():
                        if (alt_exporter != exporter and 
                            self.trade_network.has_edge(alt_exporter, importer)):
                            alt_flows = self.trade_network[alt_exporter][importer].get('product_flows', {})
                            if alt_flows.get(product, 0) > flow_value * 0.1:  # At least 10% of current flow
                                alternative_suppliers += 1
                    
                    if alternative_suppliers < 2:  # Fewer than 2 alternatives
                        critical_product_dependency += flow_value
            
            chokepoints['critical_trade_routes'][(exporter, importer)] = {
                'route_importance': route_importance,
                'critical_product_dependency': critical_product_dependency,
                'alternative_suppliers': alternative_suppliers
            }
        
        # Calculate systemic risk score
        country_criticalities = [data['criticality_score'] for data in chokepoints['critical_countries'].values()]
        route_criticalities = [data['route_importance'] for data in chokepoints['critical_trade_routes'].values()]
        
        systemic_risk = (0.7 * np.mean(country_criticalities) + 0.3 * np.mean(route_criticalities) 
                        if country_criticalities and route_criticalities else 0)
        
        chokepoints['systemic_risk_score'] = systemic_risk
        
        return chokepoints
    
    def _calculate_scenario_impact(self, scenario: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact of specific scenario on trade flows."""
        scenario_factors = {k: v for k, v in scenario.items() if k != 'name'}
        
        # Recalculate trade flows under scenario
        original_total_trade = sum(data['total_value'] for _, _, data in self.trade_network.edges(data=True))
        scenario_total_trade = 0
        
        product_impacts = {}
        country_impacts = {}
        
        for product in self.product_categories:
            original_product_trade = 0
            scenario_product_trade = 0
            
            for exporter in self.trade_network.nodes():
                for importer in self.trade_network.nodes():
                    if exporter != importer:
                        # Original flow
                        if self.trade_network.has_edge(exporter, importer):
                            original_flow = self.trade_network[exporter][importer].get('product_flows', {}).get(product, 0)
                        else:
                            original_flow = 0
                        
                        # Scenario flow
                        scenario_flow = self.gravity_model.predict_trade_flow(
                            exporter, importer, product, scenario_factors=scenario_factors
                        )
                        
                        original_product_trade += original_flow
                        scenario_product_trade += scenario_flow
            
            product_impact = ((scenario_product_trade - original_product_trade) / 
                            original_product_trade if original_product_trade > 0 else 0)
            product_impacts[product] = product_impact
            scenario_total_trade += scenario_product_trade
        
        # Country-level impacts
        for country in self.trade_network.nodes():
            original_country_exports = sum(
                self.trade_network[country][neighbor]['total_value']
                for neighbor in self.trade_network.successors(country)
            )
            
            scenario_country_exports = 0
            for neighbor in self.trade_network.nodes():
                if neighbor != country:
                    for product in self.product_categories:
                        scenario_country_exports += self.gravity_model.predict_trade_flow(
                            country, neighbor, product, scenario_factors=scenario_factors
                        )
            
            country_impact = ((scenario_country_exports - original_country_exports) / 
                            original_country_exports if original_country_exports > 0 else 0)
            country_impacts[country] = country_impact
        
        overall_impact = ((scenario_total_trade - original_total_trade) / 
                         original_total_trade if original_total_trade > 0 else 0)
        
        return {
            'overall_trade_impact': overall_impact,
            'product_impacts': product_impacts,
            'country_impacts': country_impacts,
            'most_affected_product': min(product_impacts.items(), key=lambda x: x[1])[0] if product_impacts else None,
            'most_affected_country': min(country_impacts.items(), key=lambda x: x[1])[0] if country_impacts else None
        }
    
    def _calculate_resilience_metrics(self) -> Dict[str, float]:
        """Calculate overall network resilience metrics."""
        # Network connectivity metrics
        density = nx.density(self.trade_network)
        clustering = nx.average_clustering(self.trade_network.to_undirected())
        
        # Centralization metrics
        in_centrality = nx.in_degree_centrality(self.trade_network)
        out_centrality = nx.out_degree_centrality(self.trade_network)
        
        centralization_in = max(in_centrality.values()) - np.mean(list(in_centrality.values()))
        centralization_out = max(out_centrality.values()) - np.mean(list(out_centrality.values()))
        
        # Robustness to random failures
        robustness_score = self._calculate_robustness_score()
        
        # Supply diversification
        diversification_score = self._calculate_diversification_score()
        
        return {
            'network_density': density,
            'clustering_coefficient': clustering,
            'centralization_score': (centralization_in + centralization_out) / 2,
            'robustness_score': robustness_score,
            'diversification_score': diversification_score,
            'overall_resilience': (0.3 * density + 0.2 * clustering + 
                                 0.2 * (1 - (centralization_in + centralization_out) / 2) +
                                 0.15 * robustness_score + 0.15 * diversification_score)
        }
    
    def _calculate_robustness_score(self) -> float:
        """Calculate network robustness to node removal."""
        original_connectivity = nx.number_connected_components(self.trade_network.to_undirected())
        
        robustness_scores = []
        for country in list(self.trade_network.nodes())[:5]:  # Test top 5 countries
            temp_network = self.trade_network.copy()
            temp_network.remove_node(country)
            remaining_connectivity = nx.number_connected_components(temp_network.to_undirected())
            
            robustness = 1 - (remaining_connectivity - original_connectivity) / len(self.trade_network.nodes())
            robustness_scores.append(max(0, robustness))
        
        return np.mean(robustness_scores) if robustness_scores else 0
    
    def _calculate_diversification_score(self) -> float:
        """Calculate supply chain diversification score."""
        diversification_scores = []
        
        for product in self.product_categories:
            suppliers_count = 0
            total_supply = 0
            supplier_shares = []
            
            for country in self.trade_network.nodes():
                country_supply = sum(
                    self.trade_network[country][neighbor].get('product_flows', {}).get(product, 0)
                    for neighbor in self.trade_network.successors(country)
                )
                
                if country_supply > 0:
                    suppliers_count += 1
                    total_supply += country_supply
                    supplier_shares.append(country_supply)
            
            if suppliers_count > 1 and total_supply > 0:
                # Normalize shares
                supplier_shares = [share / total_supply for share in supplier_shares]
                # Calculate Herfindahl-Hirschman Index
                hhi = sum(share**2 for share in supplier_shares)
                # Diversification is inverse of concentration
                diversification = 1 - hhi
            else:
                diversification = 0
            
            diversification_scores.append(diversification)
        
        return np.mean(diversification_scores) if diversification_scores else 0
    
    def optimize_supply_chain_resilience(self, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize supply chain for resilience under constraints."""
        # This is a simplified optimization problem
        # In practice, this would be a complex multi-objective optimization
        
        optimization_results = {
            'recommended_suppliers': {},
            'diversification_strategy': {},
            'risk_mitigation_measures': [],
            'expected_resilience_improvement': 0
        }
        
        # Analyze current vulnerabilities
        vulnerability_analysis = self.analyze_supply_chain_vulnerability()
        
        # For each product with high concentration risk
        for product, risk_data in vulnerability_analysis['concentration_risk'].items():
            if risk_data['hhi'] > 0.25:  # High concentration threshold
                # Recommend diversification
                current_suppliers = []
                alternative_suppliers = []
                
                for country in self.trade_network.nodes():
                    country_supply = sum(
                        self.trade_network[country][neighbor].get('product_flows', {}).get(product, 0)
                        for neighbor in self.trade_network.successors(country)
                    )
                    
                    if country_supply > 0:
                        current_suppliers.append((country, country_supply))
                    else:
                        # Check potential based on comparative advantage
                        comparative_advantages = self.calculate_comparative_advantage()
                        country_product_key = f"{country}_{product}"
                        if (country_product_key in comparative_advantages and 
                            comparative_advantages[country_product_key].revealed_comparative_advantage > 0.5):
                            alternative_suppliers.append(country)
                
                # Recommend top alternative suppliers
                top_alternatives = alternative_suppliers[:3]  # Top 3 alternatives
                
                optimization_results['recommended_suppliers'][product] = {
                    'current_suppliers': [supplier[0] for supplier in current_suppliers],
                    'recommended_alternatives': top_alternatives,
                    'target_diversification': min(0.8, risk_data['hhi'] + 0.3)  # Reduce concentration
                }
        
        # Generate risk mitigation measures
        chokepoints = vulnerability_analysis['chokepoint_analysis']
        critical_countries = sorted(
            chokepoints['critical_countries'].items(),
            key=lambda x: x[1]['criticality_score'],
            reverse=True
        )[:3]
        
        for country, criticality_data in critical_countries:
            optimization_results['risk_mitigation_measures'].append({
                'type': 'country_dependency_reduction',
                'target_country': country,
                'criticality_score': criticality_data['criticality_score'],
                'recommendation': f"Reduce dependency on {country} by developing alternative suppliers"
            })
        
        # Calculate expected resilience improvement
        current_resilience = vulnerability_analysis['resilience_metrics']['overall_resilience']
        # Estimate improvement based on diversification recommendations
        estimated_improvement = len(optimization_results['recommended_suppliers']) * 0.05  # 5% per diversified product
        optimization_results['expected_resilience_improvement'] = min(0.3, estimated_improvement)  # Cap at 30%
        
        return optimization_results
    
    def generate_trade_policy_recommendations(self, policy_objectives: List[str]) -> Dict[str, Any]:
        """Generate trade policy recommendations based on objectives."""
        recommendations = {
            'multilateral_agreements': [],
            'bilateral_initiatives': [],
            'unilateral_measures': [],
            'investment_priorities': [],
            'risk_assessments': {}
        }
        
        vulnerability_analysis = self.analyze_supply_chain_vulnerability()
        comparative_advantages = self.calculate_comparative_advantage()
        
        # Security-focused recommendations
        if 'supply_chain_security' in policy_objectives:
            critical_products = [
                product for product, risk_data in vulnerability_analysis['concentration_risk'].items()
                if risk_data['hhi'] > 0.4  # Very high concentration
            ]
            
            for product in critical_products:
                recommendations['investment_priorities'].append({
                    'type': 'domestic_capacity_building',
                    'product': product,
                    'rationale': f"High supply concentration risk (HHI: {vulnerability_analysis['concentration_risk'][product]['hhi']:.2f})",
                    'estimated_investment': self._estimate_capacity_investment_cost(product)
                })
        
        # Economic efficiency recommendations
        if 'economic_efficiency' in policy_objectives:
            # Identify products where country has comparative advantage but low market share
            for ca_key, ca_data in comparative_advantages.items():
                if ca_data.revealed_comparative_advantage > 1.2 and ca_data.opportunity_cost < 0.3:
                    recommendations['unilateral_measures'].append({
                        'type': 'export_promotion',
                        'country': ca_data.country,
                        'product': ca_data.product,
                        'comparative_advantage': ca_data.revealed_comparative_advantage,
                        'recommended_action': 'Increase export incentives and trade facilitation'
                    })
        
        # Cooperation and alliance recommendations
        if 'international_cooperation' in policy_objectives:
            # Identify potential partnership opportunities
            technology_leaders = {
                'semiconductors': ['Taiwan', 'South_Korea', 'Japan'],
                'advanced_semiconductors': ['Taiwan', 'South_Korea'],
                'renewable_energy_equipment': ['Germany', 'China'],
                'datacenter_equipment': ['USA', 'China', 'Germany']
            }
            
            for product, leaders in technology_leaders.items():
                for leader in leaders:
                    recommendations['bilateral_initiatives'].append({
                        'type': 'technology_partnership',
                        'partner_country': leader,
                        'product_focus': product,
                        'cooperation_areas': ['R&D collaboration', 'supply chain integration', 'standards harmonization']
                    })
        
        # Risk assessment for different policy scenarios
        policy_scenarios = [
            {'name': 'protectionist', 'trade_war_intensity': 0.8, 'technology_restrictions': 0.7},
            {'name': 'free_trade', 'trade_war_intensity': 0.1, 'technology_restrictions': 0.2},
            {'name': 'selective_decoupling', 'trade_war_intensity': 0.4, 'technology_restrictions': 0.8}
        ]
        
        for scenario in policy_scenarios:
            scenario_impact = self._calculate_scenario_impact(scenario)
            recommendations['risk_assessments'][scenario['name']] = {
                'trade_impact': scenario_impact['overall_trade_impact'],
                'most_affected_sectors': [
                    product for product, impact in scenario_impact['product_impacts'].items()
                    if impact < -0.2  # More than 20% decline
                ],
                'policy_recommendation': self._generate_scenario_policy_recommendation(scenario, scenario_impact)
            }
        
        return recommendations
    
    def _estimate_capacity_investment_cost(self, product: str) -> float:
        """Estimate investment cost for building domestic capacity."""
        # Simplified cost estimation based on product type
        investment_costs = {
            'semiconductors': 20e9,  # $20B for fab capacity
            'advanced_semiconductors': 50e9,  # $50B for cutting-edge fabs
            'datacenter_equipment': 5e9,  # $5B for manufacturing capacity
            'renewable_energy_equipment': 10e9,  # $10B for comprehensive capacity
            'rare_earth_materials': 15e9,  # $15B for mining and processing
            'optical_components': 3e9  # $3B for manufacturing capacity
        }
        
        return investment_costs.get(product, 5e9)  # Default $5B
    
    def _generate_scenario_policy_recommendation(self, scenario: Dict[str, Any], 
                                               impact_analysis: Dict[str, Any]) -> str:
        """Generate policy recommendation for specific scenario."""
        scenario_name = scenario['name']
        overall_impact = impact_analysis['overall_trade_impact']
        
        if scenario_name == 'protectionist':
            if overall_impact < -0.3:
                return "High-risk scenario: Consider graduated response rather than full protection"
            else:
                return "Manageable impact: Can pursue selective protection for critical sectors"
        
        elif scenario_name == 'free_trade':
            if overall_impact > 0.2:
                return "Beneficial scenario: Pursue multilateral trade liberalization"
            else:
                return "Mixed results: Combine trade liberalization with domestic support measures"
        
        elif scenario_name == 'selective_decoupling':
            most_affected = impact_analysis.get('most_affected_sectors', [])
            if len(most_affected) > 2:
                return "Carefully calibrate decoupling to minimize collateral damage"
            else:
                return "Feasible approach: Focus decoupling on truly critical technologies"
        
        return "Monitor developments and adjust policy as needed" 