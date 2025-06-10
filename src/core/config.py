"""
Configuration management for the datacenter simulation system.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from pathlib import Path
import json


class SimulationConfig(BaseModel):
    """Main simulation configuration."""
    
    # Simulation parameters
    simulation_duration: int = Field(default=60, description="Simulation duration in months")
    time_step: float = Field(default=1.0, description="Time step in months")
    random_seed: Optional[int] = Field(default=42, description="Random seed for reproducibility")
    
    # Economic parameters
    discount_rate: float = Field(default=0.08, description="Annual discount rate")
    inflation_rate: float = Field(default=0.025, description="Annual inflation rate")
    carbon_price: float = Field(default=85.0, description="Carbon price per ton CO2")
    
    # Market parameters
    global_datacenter_growth_rate: float = Field(default=0.30, description="Annual datacenter demand growth")
    ai_workload_share: float = Field(default=0.50, description="Share of AI workloads by 2025")
    power_constraint_severity: float = Field(default=0.75, description="Power availability constraint factor")


class AgentConfig(BaseModel):
    """Configuration for different agent types."""
    
    # Hyperscaler configurations
    hyperscaler_capex_budgets: Dict[str, float] = Field(
        default={
            "Google": 35.0e9,  # $35B annual capex
            "Amazon": 65.0e9,  # $65B annual capex
            "Microsoft": 45.0e9,  # $45B annual capex
            "Meta": 28.0e9,     # $28B annual capex
        },
        description="Annual capex budgets in USD"
    )
    
    # Semiconductor company configurations
    semiconductor_capacities: Dict[str, Dict[str, float]] = Field(
        default={
            "TSMC": {"advanced_nodes": 15000, "mature_nodes": 25000},  # Wafers per month
            "Samsung": {"advanced_nodes": 8000, "mature_nodes": 18000},
            "Intel": {"advanced_nodes": 5000, "mature_nodes": 22000},
        },
        description="Manufacturing capacity by process node"
    )
    
    # Nation configurations
    nation_energy_capacities: Dict[str, float] = Field(
        default={
            "USA": 4500.0,      # Available datacenter power capacity in MW
            "China": 3200.0,
            "Germany": 800.0,
            "Singapore": 600.0,
            "Ireland": 400.0,
            "Japan": 1200.0,
        },
        description="Available datacenter power capacity by country"
    )


class GeopoliticalConfig(BaseModel):
    """Geopolitical scenario parameters."""
    
    trade_war_probability: float = Field(default=0.25, description="Probability of trade war escalation")
    export_control_severity: float = Field(default=0.60, description="Semiconductor export control severity")
    
    tariff_rates: Dict[str, Dict[str, float]] = Field(
        default={
            "USA": {"China": 0.25, "EU": 0.05, "Others": 0.10},
            "China": {"USA": 0.30, "EU": 0.08, "Others": 0.12},
            "EU": {"USA": 0.06, "China": 0.15, "Others": 0.08},
        },
        description="Bilateral tariff rates"
    )
    
    technology_restrictions: List[str] = Field(
        default=["advanced_semiconductors", "quantum_computing", "ai_accelerators"],
        description="Technologies subject to export controls"
    )


class EconomicConfig(BaseModel):
    """Economic modeling parameters."""
    
    # Supply chain parameters
    supplier_concentration_risk: float = Field(default=0.70, description="Supply chain concentration risk factor")
    disruption_frequency: float = Field(default=0.15, description="Annual probability of major disruption")
    inventory_buffer_months: float = Field(default=3.0, description="Strategic inventory buffer")
    
    # Financial parameters
    exchange_rate_volatility: float = Field(default=0.12, description="Annual exchange rate volatility")
    political_risk_premium: Dict[str, float] = Field(
        default={
            "USA": 0.02, "EU": 0.03, "China": 0.08, "Singapore": 0.04,
            "India": 0.12, "Vietnam": 0.15, "Others": 0.20
        },
        description="Political risk premiums by region"
    )
    
    # Energy economics
    renewable_energy_targets: Dict[str, float] = Field(
        default={
            "USA": 0.80, "EU": 0.90, "China": 0.70, "Singapore": 0.85,
            "India": 0.60, "Others": 0.50
        },
        description="Renewable energy targets by 2030"
    )


class DataConfig(BaseModel):
    """Data generation and storage configuration."""
    
    # Dataset sizes
    num_datacenter_sites: int = Field(default=10000, description="Number of potential datacenter sites")
    num_suppliers: int = Field(default=500, description="Number of semiconductor suppliers")
    num_countries: int = Field(default=50, description="Number of countries in simulation")
    
    # Data paths
    data_directory: Path = Field(default=Path("data"), description="Data storage directory")
    output_directory: Path = Field(default=Path("output"), description="Simulation output directory")
    cache_directory: Path = Field(default=Path("cache"), description="Cache directory")
    
    # Database configuration
    database_url: str = Field(default="sqlite:///datacenter_simulation.db", description="Database connection URL")


class Config:
    """Main configuration manager."""
    
    def __init__(self, config_file: Optional[Path] = None):
        """Initialize configuration from file or defaults."""
        if config_file and config_file.exists():
            with open(config_file, 'r') as f:
                config_data = json.load(f)
        else:
            config_data = {}
        
        self.simulation = SimulationConfig(**config_data.get('simulation', {}))
        self.agents = AgentConfig(**config_data.get('agents', {}))
        self.geopolitical = GeopoliticalConfig(**config_data.get('geopolitical', {}))
        self.economic = EconomicConfig(**config_data.get('economic', {}))
        self.data = DataConfig(**config_data.get('data', {}))
    
    def save(self, config_file: Path) -> None:
        """Save configuration to file."""
        config_data = {
            'simulation': self.simulation.model_dump(),
            'agents': self.agents.model_dump(),
            'geopolitical': self.geopolitical.model_dump(),
            'economic': self.economic.model_dump(),
            'data': self.data.model_dump(),
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2, default=str)
    
    def validate(self) -> List[str]:
        """Validate configuration and return any errors."""
        errors = []
        
        # Validate budget constraints
        total_capex = sum(self.agents.hyperscaler_capex_budgets.values())
        if total_capex < 100e9:  # $100B minimum realistic total
            errors.append("Total hyperscaler capex appears too low for realistic simulation")
        
        # Validate capacity constraints
        total_power = sum(self.agents.nation_energy_capacities.values())
        if total_power < 5000:  # 5GW minimum
            errors.append("Total available power capacity appears insufficient")
        
        # Validate economic parameters
        if self.economic.disruption_frequency > 1.0:
            errors.append("Disruption frequency cannot exceed 100%")
        
        return errors


# Global configuration instance
config = Config()


def load_config(config_file: Optional[Path] = None) -> Config:
    """Load configuration from file."""
    global config
    config = Config(config_file)
    return config


def get_config() -> Config:
    """Get current configuration."""
    return config 