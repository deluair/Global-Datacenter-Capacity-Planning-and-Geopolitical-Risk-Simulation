#!/usr/bin/env python3
"""
Simple test to verify all modules can be imported correctly.
"""

try:
    print("Testing core imports...")
    from src.core.config import get_config
    print("✓ Config module imported successfully")
    
    from src.agents.base_agent import BaseAgent, AgentType
    print("✓ Base agent module imported successfully")
    
    from src.agents.hyperscaler_agent import HyperscalerAgent
    print("✓ Hyperscaler agent module imported successfully")
    
    from src.agents.nation_agent import NationAgent, PolicyDomain, NationState
    print("✓ Nation agent module imported successfully")
    
    print("\nTesting data generators...")
    from src.data.generators import DatacenterSiteGenerator, SemiconductorSupplierGenerator
    print("✓ Data generators imported successfully")
    
    print("\nTesting economics modules...")
    from src.economics.welfare_analysis import WelfareAnalyzer
    print("✓ Welfare analysis module imported successfully")
    
    from src.economics.international_trade import TradeNetworkAnalyzer
    print("✓ International trade module imported successfully")
    
    print("\nTesting simulation engine...")
    from src.core.simulation_engine import SimulationEngine
    print("✓ Simulation engine imported successfully")
    
    print("\n🎉 ALL IMPORTS SUCCESSFUL!")
    print("The comprehensive Global Datacenter Capacity Planning simulation is ready to run.")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Unexpected error: {e}") 