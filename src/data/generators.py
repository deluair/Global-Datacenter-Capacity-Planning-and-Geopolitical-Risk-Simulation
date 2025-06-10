"""
Data generators for realistic simulation datasets.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
import random
from scipy.stats import norm, gamma, beta, pareto
import json

from ..core.config import get_config


@dataclass
class DatacenterSite:
    """Represents a potential datacenter site."""
    site_id: str
    country: str
    region: str
    city: str
    latitude: float
    longitude: float
    land_cost_per_sqm: float  # USD per square meter
    construction_time_months: int
    regulatory_approval_months: int
    available_power_mw: float
    power_cost_per_mwh: float  # USD per MWh
    renewable_energy_share: float  # 0-1
    grid_stability_score: float  # 0-1
    fiber_connectivity_score: float  # 0-1
    submarine_cable_distance_km: float
    customer_proximity_score: float  # 0-1
    labor_availability_score: float  # 0-1
    average_wage_usd_per_hour: float
    corporate_tax_rate: float  # 0-1
    tax_incentives_available: bool
    political_stability_index: float  # 0-1
    natural_disaster_risk: float  # 0-1
    carbon_intensity_kg_co2_per_mwh: float


@dataclass
class SemiconductorSupplier:
    """Represents a semiconductor supplier."""
    supplier_id: str
    company_name: str
    country: str
    technology_nodes: List[str]  # e.g., ["7nm", "14nm", "28nm"]
    monthly_capacity_wafers: Dict[str, float]
    yield_rates: Dict[str, float]  # 0-1 by technology node
    lead_time_weeks: int
    quality_score: float  # 0-1
    geopolitical_risk_score: float  # 0-1
    price_per_wafer: Dict[str, float]  # USD by technology node
    export_restrictions: List[str]  # List of restricted destination countries
    alternative_suppliers: List[str]  # Supplier IDs of alternatives
    switching_cost_multiplier: float  # Cost multiplier for switching


@dataclass
class EconomicIndicators:
    """Economic indicators for a specific country and time period."""
    country: str
    date: str
    gdp_growth_rate: float
    inflation_rate: float
    exchange_rate_usd: float
    interest_rate: float
    unemployment_rate: float
    energy_price_index: float
    manufacturing_pmi: float
    political_risk_score: float
    trade_openness_index: float


class DatacenterSiteGenerator:
    """Generates realistic datacenter site data."""
    
    def __init__(self, seed: Optional[int] = None):
        if seed:
            np.random.seed(seed)
            random.seed(seed)
        
        self.config = get_config()
        
        # City data with realistic characteristics
        self.major_cities = {
            "USA": [
                ("Ashburn", 39.0438, -77.4874, "Virginia", 850, 0.65),
                ("Santa Clara", 37.3541, -121.9552, "California", 1200, 0.85),
                ("Dallas", 32.7767, -96.7970, "Texas", 650, 0.45),
                ("Chicago", 41.8781, -87.6298, "Illinois", 750, 0.55),
                ("Atlanta", 33.7490, -84.3880, "Georgia", 600, 0.40),
            ],
            "China": [
                ("Beijing", 39.9042, 116.4074, "Beijing", 400, 0.25),
                ("Shanghai", 31.2304, 121.4737, "Shanghai", 450, 0.30),
                ("Shenzhen", 22.5431, 114.0579, "Guangdong", 500, 0.35),
                ("Guangzhou", 23.1291, 113.2644, "Guangdong", 420, 0.28),
                ("Hangzhou", 30.2741, 120.1551, "Zhejiang", 380, 0.32),
            ],
            "Germany": [
                ("Frankfurt", 50.1109, 8.6821, "Hesse", 950, 0.75),
                ("Munich", 48.1351, 11.5820, "Bavaria", 1100, 0.80),
                ("Berlin", 52.5200, 13.4050, "Berlin", 800, 0.70),
                ("Hamburg", 53.5511, 9.9937, "Hamburg", 850, 0.72),
            ],
            "Singapore": [
                ("Singapore", 1.3521, 103.8198, "Singapore", 1800, 0.95),
            ],
            "Ireland": [
                ("Dublin", 53.3498, -6.2603, "Dublin", 1400, 0.90),
            ],
            "Japan": [
                ("Tokyo", 35.6762, 139.6503, "Tokyo", 1600, 0.80),
                ("Osaka", 34.6937, 135.5023, "Osaka", 1400, 0.75),
            ],
        }
    
    def generate_sites(self, num_sites: int) -> List[DatacenterSite]:
        """Generate realistic datacenter sites."""
        sites = []
        
        for i in range(num_sites):
            # Select country based on realistic distribution
            country = np.random.choice(
                list(self.major_cities.keys()),
                p=[0.45, 0.25, 0.12, 0.08, 0.05, 0.05]  # USA, China, Germany, Singapore, Ireland, Japan
            )
            
            # Select city within country
            city_data = random.choice(self.major_cities[country])
            city, lat, lon, region, base_land_cost, renewable_share = city_data
            
            # Add some geographic variation
            lat_offset = np.random.normal(0, 0.5)
            lon_offset = np.random.normal(0, 0.5)
            
            # Generate realistic site characteristics
            site = DatacenterSite(
                site_id=f"DC_{country}_{i:04d}",
                country=country,
                region=region,
                city=city,
                latitude=lat + lat_offset,
                longitude=lon + lon_offset,
                land_cost_per_sqm=max(50, np.random.normal(base_land_cost, base_land_cost * 0.3)),
                construction_time_months=int(np.random.gamma(18, 1.2)),  # 12-36 months typically
                regulatory_approval_months=int(np.random.gamma(8, 1.5)),  # 6-24 months
                available_power_mw=max(10, np.random.exponential(150)),  # Power availability
                power_cost_per_mwh=self._generate_power_cost(country),
                renewable_energy_share=min(1.0, max(0, np.random.beta(
                    renewable_share * 5, (1 - renewable_share) * 5
                ))),
                grid_stability_score=self._generate_grid_stability(country),
                fiber_connectivity_score=self._generate_connectivity_score(country, city),
                submarine_cable_distance_km=self._generate_cable_distance(lat, lon),
                customer_proximity_score=self._generate_proximity_score(country, city),
                labor_availability_score=self._generate_labor_score(country),
                average_wage_usd_per_hour=self._generate_wage(country),
                corporate_tax_rate=self._generate_tax_rate(country),
                tax_incentives_available=np.random.random() < 0.3,  # 30% chance
                political_stability_index=self._generate_political_stability(country),
                natural_disaster_risk=self._generate_disaster_risk(lat, lon),
                carbon_intensity_kg_co2_per_mwh=self._generate_carbon_intensity(country, renewable_share)
            )
            
            sites.append(site)
        
        return sites
    
    def _generate_power_cost(self, country: str) -> float:
        """Generate realistic power costs by country."""
        base_costs = {
            "USA": 65, "China": 45, "Germany": 120, 
            "Singapore": 110, "Ireland": 95, "Japan": 140
        }
        base = base_costs.get(country, 80)
        return max(20, np.random.normal(base, base * 0.25))
    
    def _generate_grid_stability(self, country: str) -> float:
        """Generate grid stability scores."""
        base_stability = {
            "USA": 0.85, "China": 0.75, "Germany": 0.92,
            "Singapore": 0.95, "Ireland": 0.88, "Japan": 0.90
        }
        base = base_stability.get(country, 0.70)
        return min(1.0, max(0.3, np.random.beta(base * 10, (1 - base) * 10)))
    
    def _generate_connectivity_score(self, country: str, city: str) -> float:
        """Generate fiber connectivity scores."""
        major_hubs = ["Ashburn", "Santa Clara", "Frankfurt", "Singapore", "Dublin", "Tokyo"]
        if city in major_hubs:
            return min(1.0, np.random.beta(8, 2))  # High connectivity
        else:
            return min(1.0, np.random.beta(3, 4))  # Lower connectivity
    
    def _generate_cable_distance(self, lat: float, lon: float) -> float:
        """Generate distance to nearest submarine cable."""
        # Simplified model - coastal areas have shorter distances
        if abs(lat) > 45:  # Higher latitudes
            return np.random.exponential(500)
        else:
            return np.random.exponential(200)
    
    def _generate_proximity_score(self, country: str, city: str) -> float:
        """Generate customer proximity scores."""
        major_markets = ["Ashburn", "Santa Clara", "Frankfurt", "Singapore", "Dublin", "Tokyo", "Beijing", "Shanghai"]
        if city in major_markets:
            return min(1.0, np.random.beta(6, 2))
        else:
            return min(1.0, np.random.beta(2, 3))
    
    def _generate_labor_score(self, country: str) -> float:
        """Generate labor availability scores."""
        base_scores = {
            "USA": 0.80, "China": 0.85, "Germany": 0.75,
            "Singapore": 0.70, "Ireland": 0.72, "Japan": 0.68
        }
        base = base_scores.get(country, 0.65)
        return min(1.0, max(0.3, np.random.beta(base * 8, (1 - base) * 8)))
    
    def _generate_wage(self, country: str) -> float:
        """Generate average wages by country."""
        base_wages = {
            "USA": 35, "China": 12, "Germany": 28,
            "Singapore": 25, "Ireland": 30, "Japan": 24
        }
        base = base_wages.get(country, 20)
        return max(8, np.random.normal(base, base * 0.2))
    
    def _generate_tax_rate(self, country: str) -> float:
        """Generate corporate tax rates."""
        base_rates = {
            "USA": 0.21, "China": 0.25, "Germany": 0.30,
            "Singapore": 0.17, "Ireland": 0.125, "Japan": 0.30
        }
        base = base_rates.get(country, 0.25)
        return min(0.5, max(0.1, np.random.normal(base, 0.03)))
    
    def _generate_political_stability(self, country: str) -> float:
        """Generate political stability indices."""
        base_stability = {
            "USA": 0.75, "China": 0.65, "Germany": 0.90,
            "Singapore": 0.95, "Ireland": 0.88, "Japan": 0.85
        }
        base = base_stability.get(country, 0.60)
        return min(1.0, max(0.2, np.random.beta(base * 10, (1 - base) * 10)))
    
    def _generate_disaster_risk(self, lat: float, lon: float) -> float:
        """Generate natural disaster risk based on location."""
        # Higher risk in certain geographic regions
        risk_zones = [
            (30, 45, 120, 150),  # Japan region - high earthquake risk
            (20, 35, -100, -80), # Gulf of Mexico - hurricane risk
            (35, 45, -125, -115), # California - earthquake risk
        ]
        
        base_risk = 0.1
        for min_lat, max_lat, min_lon, max_lon in risk_zones:
            if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
                base_risk = 0.3
                break
        
        return min(1.0, max(0.05, np.random.beta(base_risk * 5, (1 - base_risk) * 5)))
    
    def _generate_carbon_intensity(self, country: str, renewable_share: float) -> float:
        """Generate carbon intensity based on energy mix."""
        base_intensity = {
            "USA": 400, "China": 550, "Germany": 350,
            "Singapore": 380, "Ireland": 300, "Japan": 450
        }
        base = base_intensity.get(country, 450)
        
        # Adjust for renewable share
        adjusted = base * (1 - renewable_share * 0.8)
        return max(50, np.random.normal(adjusted, adjusted * 0.2))


class SemiconductorSupplierGenerator:
    """Generates realistic semiconductor supplier data."""
    
    def __init__(self, seed: Optional[int] = None):
        if seed:
            np.random.seed(seed)
            random.seed(seed)
        
        # Technology node categories
        self.tech_nodes = {
            "advanced": ["3nm", "5nm", "7nm"],
            "mature": ["14nm", "28nm", "40nm"],
            "legacy": ["65nm", "130nm", "180nm"]
        }
        
        # Major suppliers with realistic characteristics
        self.major_suppliers = {
            "TSMC": {
                "country": "Taiwan",
                "specialization": "advanced",
                "market_share": 0.54,
                "quality_score": 0.95
            },
            "Samsung": {
                "country": "South Korea",
                "specialization": "advanced",
                "market_share": 0.17,
                "quality_score": 0.90
            },
            "Intel": {
                "country": "USA",
                "specialization": "mature",
                "market_share": 0.08,
                "quality_score": 0.88
            },
            "GlobalFoundries": {
                "country": "USA",
                "specialization": "mature",
                "market_share": 0.06,
                "quality_score": 0.82
            },
            "SMIC": {
                "country": "China",
                "specialization": "legacy",
                "market_share": 0.05,
                "quality_score": 0.75
            }
        }
    
    def generate_suppliers(self, num_suppliers: int) -> List[SemiconductorSupplier]:
        """Generate realistic semiconductor suppliers."""
        suppliers = []
        
        # First, create major suppliers
        for name, data in self.major_suppliers.items():
            supplier = self._create_major_supplier(name, data)
            suppliers.append(supplier)
        
        # Then create additional smaller suppliers
        remaining = num_suppliers - len(self.major_suppliers)
        for i in range(remaining):
            supplier = self._create_minor_supplier(f"SUP_{i:03d}")
            suppliers.append(supplier)
        
        # Set up alternative supplier relationships
        self._setup_alternative_suppliers(suppliers)
        
        return suppliers
    
    def _create_major_supplier(self, name: str, data: Dict) -> SemiconductorSupplier:
        """Create a major semiconductor supplier."""
        specialization = data["specialization"]
        
        # Determine technology nodes based on specialization
        if specialization == "advanced":
            nodes = self.tech_nodes["advanced"] + self.tech_nodes["mature"][:2]
        elif specialization == "mature":
            nodes = self.tech_nodes["mature"] + self.tech_nodes["legacy"][:1]
        else:
            nodes = self.tech_nodes["legacy"] + self.tech_nodes["mature"][-1:]
        
        # Generate capacities and prices
        capacities = {}
        prices = {}
        yield_rates = {}
        
        for node in nodes:
            if node in self.tech_nodes["advanced"]:
                base_capacity = 5000 * data["market_share"]
                base_price = 8000
                base_yield = 0.85
            elif node in self.tech_nodes["mature"]:
                base_capacity = 15000 * data["market_share"]
                base_price = 3000
                base_yield = 0.92
            else:
                base_capacity = 25000 * data["market_share"]
                base_price = 800
                base_yield = 0.96
            
            capacities[node] = max(100, np.random.normal(base_capacity, base_capacity * 0.2))
            prices[node] = max(100, np.random.normal(base_price, base_price * 0.15))
            yield_rates[node] = min(0.98, max(0.5, np.random.normal(base_yield, 0.05)))
        
        # Export restrictions based on geopolitical factors
        export_restrictions = []
        if data["country"] in ["Taiwan", "South Korea", "USA"]:
            if np.random.random() < 0.7:  # 70% chance of restrictions to China
                export_restrictions.append("China")
        elif data["country"] == "China":
            if np.random.random() < 0.4:  # 40% chance of restrictions from others
                export_restrictions.extend(["USA", "Japan"])
        
        return SemiconductorSupplier(
            supplier_id=name.upper().replace(" ", "_"),
            company_name=name,
            country=data["country"],
            technology_nodes=nodes,
            monthly_capacity_wafers=capacities,
            yield_rates=yield_rates,
            lead_time_weeks=int(np.random.normal(12, 3)),
            quality_score=data["quality_score"],
            geopolitical_risk_score=self._calculate_geopolitical_risk(data["country"]),
            price_per_wafer=prices,
            export_restrictions=export_restrictions,
            alternative_suppliers=[],  # Set later
            switching_cost_multiplier=np.random.uniform(1.2, 2.5)
        )
    
    def _create_minor_supplier(self, supplier_id: str) -> SemiconductorSupplier:
        """Create a minor semiconductor supplier."""
        # Random country selection weighted by semiconductor industry presence
        countries = ["Taiwan", "South Korea", "China", "Japan", "Singapore", "Malaysia", "Thailand"]
        weights = [0.25, 0.20, 0.15, 0.12, 0.10, 0.10, 0.08]
        country = np.random.choice(countries, p=weights)
        
        # Smaller suppliers typically focus on mature/legacy nodes
        specialization = np.random.choice(["mature", "legacy"], p=[0.6, 0.4])
        
        if specialization == "mature":
            nodes = random.sample(self.tech_nodes["mature"] + self.tech_nodes["legacy"], 
                                random.randint(2, 4))
        else:
            nodes = random.sample(self.tech_nodes["legacy"], random.randint(1, 3))
        
        # Generate smaller capacities and competitive prices
        capacities = {}
        prices = {}
        yield_rates = {}
        
        for node in nodes:
            if node in self.tech_nodes["mature"]:
                base_capacity = np.random.exponential(2000)
                base_price = np.random.normal(2800, 400)
                base_yield = np.random.normal(0.88, 0.05)
            else:
                base_capacity = np.random.exponential(5000)
                base_price = np.random.normal(750, 150)
                base_yield = np.random.normal(0.93, 0.03)
            
            capacities[node] = max(50, base_capacity)
            prices[node] = max(100, base_price)
            yield_rates[node] = min(0.95, max(0.6, base_yield))
        
        return SemiconductorSupplier(
            supplier_id=supplier_id,
            company_name=f"Semiconductor Corp {supplier_id[-3:]}",
            country=country,
            technology_nodes=nodes,
            monthly_capacity_wafers=capacities,
            yield_rates=yield_rates,
            lead_time_weeks=int(np.random.normal(8, 2)),
            quality_score=np.random.uniform(0.6, 0.85),
            geopolitical_risk_score=self._calculate_geopolitical_risk(country),
            price_per_wafer=prices,
            export_restrictions=self._generate_export_restrictions(country),
            alternative_suppliers=[],
            switching_cost_multiplier=np.random.uniform(1.1, 1.8)
        )
    
    def _calculate_geopolitical_risk(self, country: str) -> float:
        """Calculate geopolitical risk score for a country."""
        risk_scores = {
            "Taiwan": 0.65,  # High due to China tensions
            "South Korea": 0.30,  # Moderate due to North Korea
            "China": 0.55,  # High due to trade tensions
            "USA": 0.20,  # Low domestic risk
            "Japan": 0.25,  # Low risk
            "Singapore": 0.15,  # Very low risk
            "Malaysia": 0.35,  # Moderate risk
            "Thailand": 0.40   # Moderate risk
        }
        base_risk = risk_scores.get(country, 0.50)
        return min(1.0, max(0.1, np.random.normal(base_risk, 0.1)))
    
    def _generate_export_restrictions(self, country: str) -> List[str]:
        """Generate export restrictions based on country."""
        restrictions = []
        
        if country in ["Taiwan", "South Korea", "Japan"]:
            if np.random.random() < 0.6:
                restrictions.append("China")
        elif country == "China":
            if np.random.random() < 0.3:
                restrictions.extend(["USA", "Japan"])
        
        return restrictions
    
    def _setup_alternative_suppliers(self, suppliers: List[SemiconductorSupplier]) -> None:
        """Set up alternative supplier relationships."""
        supplier_dict = {s.supplier_id: s for s in suppliers}
        
        for supplier in suppliers:
            alternatives = []
            for other in suppliers:
                if (other.supplier_id != supplier.supplier_id and 
                    set(supplier.technology_nodes) & set(other.technology_nodes)):
                    # Probability based on technology overlap and geographic diversity
                    overlap = len(set(supplier.technology_nodes) & set(other.technology_nodes))
                    geo_diversity = 1.0 if other.country != supplier.country else 0.5
                    prob = min(0.8, overlap * 0.2 * geo_diversity)
                    
                    if np.random.random() < prob:
                        alternatives.append(other.supplier_id)
            
            supplier.alternative_suppliers = alternatives[:5]  # Limit to 5 alternatives


class EconomicDataGenerator:
    """Generates economic indicators and time series data."""
    
    def __init__(self, seed: Optional[int] = None):
        if seed:
            np.random.seed(seed)
        
        self.config = get_config()
    
    def generate_economic_indicators(self, countries: List[str], 
                                   start_date: str, periods: int) -> List[EconomicIndicators]:
        """Generate economic indicators for multiple countries over time."""
        indicators = []
        
        # Base economic characteristics by country
        base_economics = {
            "USA": {
                "gdp_growth": 0.025, "inflation": 0.025, "exchange_rate": 1.0,
                "interest_rate": 0.05, "unemployment": 0.04, "energy_price": 100,
                "manufacturing_pmi": 52, "political_risk": 0.15, "trade_openness": 0.85
            },
            "China": {
                "gdp_growth": 0.055, "inflation": 0.02, "exchange_rate": 7.2,
                "interest_rate": 0.035, "unemployment": 0.05, "energy_price": 80,
                "manufacturing_pmi": 51, "political_risk": 0.35, "trade_openness": 0.75
            },
            "Germany": {
                "gdp_growth": 0.015, "inflation": 0.02, "exchange_rate": 0.85,
                "interest_rate": 0.02, "unemployment": 0.035, "energy_price": 150,
                "manufacturing_pmi": 49, "political_risk": 0.10, "trade_openness": 0.95
            },
            "Singapore": {
                "gdp_growth": 0.035, "inflation": 0.025, "exchange_rate": 1.35,
                "interest_rate": 0.03, "unemployment": 0.025, "energy_price": 120,
                "manufacturing_pmi": 50, "political_risk": 0.05, "trade_openness": 0.98
            },
            "Ireland": {
                "gdp_growth": 0.045, "inflation": 0.025, "exchange_rate": 0.85,
                "interest_rate": 0.02, "unemployment": 0.04, "energy_price": 130,
                "manufacturing_pmi": 52, "political_risk": 0.08, "trade_openness": 0.92
            },
            "Japan": {
                "gdp_growth": 0.01, "inflation": 0.01, "exchange_rate": 110,
                "interest_rate": 0.005, "unemployment": 0.025, "energy_price": 140,
                "manufacturing_pmi": 49, "political_risk": 0.12, "trade_openness": 0.80
            }
        }
        
        # Generate time series for each country
        dates = pd.date_range(start=start_date, periods=periods, freq='M')
        
        for country in countries:
            base = base_economics.get(country, base_economics["USA"])  # Default to USA
            
            # Initialize state variables for time series
            state = {key: value for key, value in base.items()}
            
            for i, date in enumerate(dates):
                # Add time-series dynamics with realistic correlations
                if i > 0:
                    # GDP growth with business cycle
                    cycle = 0.01 * np.sin(2 * np.pi * i / 48)  # 4-year cycle
                    shock = np.random.normal(0, 0.005)
                    state["gdp_growth"] = max(-0.05, min(0.15, 
                        0.7 * state["gdp_growth"] + 0.3 * base["gdp_growth"] + cycle + shock))
                    
                    # Inflation with persistence
                    inflation_shock = np.random.normal(0, 0.003)
                    state["inflation"] = max(0, min(0.10,
                        0.8 * state["inflation"] + 0.2 * base["inflation"] + inflation_shock))
                    
                    # Exchange rate with volatility
                    fx_shock = np.random.normal(0, 0.02)
                    state["exchange_rate"] = max(0.1, 
                        state["exchange_rate"] * (1 + fx_shock))
                    
                    # Interest rate responding to inflation and growth
                    taylor_rule = base["interest_rate"] + 1.5 * (state["inflation"] - 0.02) + 0.5 * state["gdp_growth"]
                    rate_shock = np.random.normal(0, 0.002)
                    state["interest_rate"] = max(0, min(0.15,
                        0.9 * state["interest_rate"] + 0.1 * taylor_rule + rate_shock))
                    
                    # Unemployment (Okun's law relationship)
                    unemployment_change = -0.3 * (state["gdp_growth"] - 0.025) + np.random.normal(0, 0.003)
                    state["unemployment"] = max(0.01, min(0.15,
                        state["unemployment"] + unemployment_change))
                    
                    # Energy prices with volatility
                    energy_shock = np.random.normal(0, 0.05)
                    state["energy_price"] = max(20,
                        state["energy_price"] * (1 + energy_shock))
                    
                    # Manufacturing PMI
                    pmi_shock = np.random.normal(0, 2)
                    state["manufacturing_pmi"] = max(30, min(70,
                        0.7 * state["manufacturing_pmi"] + 0.3 * base["manufacturing_pmi"] + pmi_shock))
                    
                    # Political risk (slow-moving)
                    if np.random.random() < 0.05:  # 5% chance of political event
                        political_shock = np.random.normal(0, 0.1)
                        state["political_risk"] = max(0, min(1,
                            state["political_risk"] + political_shock))
                    
                    # Trade openness (policy-driven)
                    if np.random.random() < 0.02:  # 2% chance of trade policy change
                        trade_shock = np.random.normal(0, 0.05)
                        state["trade_openness"] = max(0.3, min(1,
                            state["trade_openness"] + trade_shock))
                
                indicator = EconomicIndicators(
                    country=country,
                    date=date.strftime('%Y-%m-%d'),
                    gdp_growth_rate=state["gdp_growth"],
                    inflation_rate=state["inflation"],
                    exchange_rate_usd=state["exchange_rate"],
                    interest_rate=state["interest_rate"],
                    unemployment_rate=state["unemployment"],
                    energy_price_index=state["energy_price"],
                    manufacturing_pmi=state["manufacturing_pmi"],
                    political_risk_score=state["political_risk"],
                    trade_openness_index=state["trade_openness"]
                )
                
                indicators.append(indicator)
        
        return indicators


def save_generated_data(sites: List[DatacenterSite], 
                       suppliers: List[SemiconductorSupplier],
                       indicators: List[EconomicIndicators],
                       data_dir: Path) -> None:
    """Save generated data to files."""
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Save datacenter sites
    sites_df = pd.DataFrame([vars(site) for site in sites])
    sites_df.to_csv(data_dir / "datacenter_sites.csv", index=False)
    
    # Save semiconductor suppliers
    suppliers_data = []
    for supplier in suppliers:
        data = vars(supplier).copy()
        # Convert nested dicts to JSON strings for CSV storage
        data['technology_nodes'] = json.dumps(data['technology_nodes'])
        data['monthly_capacity_wafers'] = json.dumps(data['monthly_capacity_wafers'])
        data['yield_rates'] = json.dumps(data['yield_rates'])
        data['price_per_wafer'] = json.dumps(data['price_per_wafer'])
        data['export_restrictions'] = json.dumps(data['export_restrictions'])
        data['alternative_suppliers'] = json.dumps(data['alternative_suppliers'])
        suppliers_data.append(data)
    
    suppliers_df = pd.DataFrame(suppliers_data)
    suppliers_df.to_csv(data_dir / "semiconductor_suppliers.csv", index=False)
    
    # Save economic indicators
    indicators_df = pd.DataFrame([vars(indicator) for indicator in indicators])
    indicators_df.to_csv(data_dir / "economic_indicators.csv", index=False)
    
    print(f"Data saved to {data_dir}")
    print(f"- {len(sites)} datacenter sites")
    print(f"- {len(suppliers)} semiconductor suppliers")
    print(f"- {len(indicators)} economic indicator records")


if __name__ == "__main__":
    # Generate sample data
    config = get_config()
    
    # Create generators
    site_gen = DatacenterSiteGenerator(seed=42)
    supplier_gen = SemiconductorSupplierGenerator(seed=42)
    econ_gen = EconomicDataGenerator(seed=42)
    
    # Generate data
    sites = site_gen.generate_sites(config.data.num_datacenter_sites)
    suppliers = supplier_gen.generate_suppliers(config.data.num_suppliers)
    
    countries = ["USA", "China", "Germany", "Singapore", "Ireland", "Japan"]
    indicators = econ_gen.generate_economic_indicators(countries, "2020-01-01", 60)
    
    # Save data
    save_generated_data(sites, suppliers, indicators, config.data.data_directory) 