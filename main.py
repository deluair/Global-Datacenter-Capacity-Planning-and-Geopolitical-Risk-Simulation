#!/usr/bin/env python3
"""
Main entry point for the Global Datacenter Capacity Planning and Geopolitical Risk Simulation.

This comprehensive simulation addresses the complex interactions between:
- Datacenter infrastructure capacity planning
- Semiconductor supply chain constraints  
- Geopolitical tensions and trade restrictions
- Energy market dynamics and sustainability requirements
- Economic welfare and international trade impacts

Usage:
    python main.py --scenario baseline --duration 60 --output results/
    python main.py --scenario trade_war --analysis policy --export-data
    python main.py --dashboard  # Launch interactive dashboard
"""

import argparse
import logging
import sys
from pathlib import Path
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

# Import core simulation components
from src.core.simulation_engine import SimulationEngine, SimulationResults
from src.core.config import load_config, get_config
from src.data.generators import (
    DatacenterSiteGenerator, 
    SemiconductorSupplierGenerator, 
    EconomicDataGenerator,
    save_generated_data
)
from src.economics.welfare_analysis import WelfareAnalyzer
from src.economics.international_trade import TradeNetworkAnalyzer

# Import visualization components
try:
    from src.visualization.dashboard import DatacenterDashboard
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False


def setup_logging(log_level: str = "INFO", log_file: Optional[Path] = None) -> None:
    """Setup comprehensive logging configuration."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )
    
    # Set specific logger levels
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("plotly").setLevel(logging.WARNING)


def generate_simulation_data(config_path: Optional[Path] = None) -> None:
    """Generate comprehensive simulation datasets."""
    logger = logging.getLogger(__name__)
    logger.info("Generating comprehensive simulation datasets...")
    
    # Load configuration
    if config_path:
        config = load_config(config_path)
    else:
        config = get_config()
    
    # Create data generators with reproducible seeds
    site_generator = DatacenterSiteGenerator(seed=42)
    supplier_generator = SemiconductorSupplierGenerator(seed=43)
    economic_generator = EconomicDataGenerator(seed=44)
    
    logger.info(f"Generating {config.data.num_datacenter_sites} datacenter sites...")
    datacenter_sites = site_generator.generate_sites(config.data.num_datacenter_sites)
    
    logger.info(f"Generating {config.data.num_suppliers} semiconductor suppliers...")
    semiconductor_suppliers = supplier_generator.generate_suppliers(config.data.num_suppliers)
    
    logger.info("Generating economic indicators time series...")
    countries = ["USA", "China", "Germany", "Japan", "South_Korea", "Taiwan", 
                "Singapore", "Ireland", "Netherlands", "India", "Vietnam"]
    economic_indicators = economic_generator.generate_economic_indicators(
        countries, "2020-01-01", 72  # 6 years of monthly data
    )
    
    # Save all generated data
    config.data.data_directory.mkdir(parents=True, exist_ok=True)
    save_generated_data(datacenter_sites, semiconductor_suppliers, 
                       economic_indicators, config.data.data_directory)
    
    logger.info(f"Successfully generated and saved simulation data to {config.data.data_directory}")


def run_baseline_simulation(duration: int = 60, scenario: str = "baseline", 
                          output_dir: Path = Path("output")) -> SimulationResults:
    """Run the baseline datacenter capacity planning simulation."""
    logger = logging.getLogger(__name__)
    logger.info(f"Running {scenario} simulation for {duration} months...")
    
    # Initialize simulation engine
    simulation_engine = SimulationEngine()
    
    # Initialize with scenario
    simulation_engine.initialize(scenario)
    
    # Run simulation
    start_time = time.time()
    results = simulation_engine.run(duration)
    end_time = time.time()
    
    logger.info(f"Simulation completed in {end_time - start_time:.2f} seconds")
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    simulation_engine.save_results(results, output_dir / scenario)
    
    # Print summary
    logger.info("=== SIMULATION SUMMARY ===")
    logger.info(f"Scenario: {scenario}")
    logger.info(f"Duration: {duration} months")
    logger.info(f"Final Market Revenue: ${results.summary_statistics['market']['total_revenue_final']/1e9:.2f}B")
    logger.info(f"Average Demand Growth: {results.summary_statistics['market']['average_demand_growth']:.1%}")
    
    for agent_id, agent_stats in results.summary_statistics['agents'].items():
        logger.info(f"{agent_id.title()}: Cash=${agent_stats['final_cash_balance']/1e9:.1f}B, "
                   f"Capacity={agent_stats.get('final_capacity_mw', 0)/1000:.1f}GW")
    
    return results


def run_scenario_analysis(scenarios: List[str], duration: int = 60, 
                         output_dir: Path = Path("output")) -> Dict[str, SimulationResults]:
    """Run comprehensive scenario analysis."""
    logger = logging.getLogger(__name__)
    logger.info(f"Running scenario analysis for {len(scenarios)} scenarios...")
    
    simulation_engine = SimulationEngine()
    scenario_results = simulation_engine.run_scenario_analysis(scenarios)
    
    # Save comparative results
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_file = output_dir / "scenario_comparison.json"
    
    comparison_data = {}
    for scenario, results in scenario_results.items():
        comparison_data[scenario] = {
            'summary_statistics': results.summary_statistics,
            'final_metrics': {
                'total_revenue': results.time_series['total_market_revenue'].iloc[-1],
                'total_costs': results.time_series['total_market_costs'].iloc[-1],
                'demand_growth': results.time_series['market_demand_growth'].mean(),
                'supply_shortage': results.time_series['semiconductor_shortage'].iloc[-1]
            }
        }
    
    with open(comparison_file, 'w') as f:
        json.dump(comparison_data, f, indent=2, default=str)
    
    logger.info(f"Scenario comparison saved to {comparison_file}")
    
    # Print comparative summary
    logger.info("=== SCENARIO COMPARISON ===")
    for scenario, results in scenario_results.items():
        final_revenue = results.time_series['total_market_revenue'].iloc[-1]
        avg_shortage = results.time_series['semiconductor_shortage'].mean()
        logger.info(f"{scenario}: Revenue=${final_revenue/1e9:.1f}B, Shortage={avg_shortage:.1%}")
    
    return scenario_results


def run_economic_analysis(simulation_results: SimulationResults, 
                         output_dir: Path = Path("output")) -> Dict[str, Any]:
    """Run comprehensive economic welfare and trade analysis."""
    logger = logging.getLogger(__name__)
    logger.info("Running comprehensive economic analysis...")
    
    # Initialize economic analyzers
    welfare_analyzer = WelfareAnalyzer()
    trade_analyzer = TradeNetworkAnalyzer()
    
    # Define policy scenarios for analysis
    policy_scenarios = [
        {
            'name': 'carbon_pricing',
            'carbon_price': 150,  # $150/tonne CO2
            'description': 'Aggressive carbon pricing policy'
        },
        {
            'name': 'trade_liberalization', 
            'tariff_reduction': 0.5,
            'trade_facilitation': 0.3,
            'description': 'Multilateral trade liberalization'
        },
        {
            'name': 'tech_sovereignty',
            'technology_restrictions': 0.8,
            'domestic_subsidy': 0.2,
            'description': 'Technology sovereignty and domestic capacity building'
        },
        {
            'name': 'green_transition',
            'carbon_price': 200,
            'renewable_mandate': 0.9,
            'green_subsidy': 0.15,
            'description': 'Comprehensive green transition'
        }
    ]
    
    # Generate welfare analysis report
    welfare_report = welfare_analyzer.generate_welfare_report(policy_scenarios)
    
    # Analyze supply chain vulnerabilities
    vulnerability_analysis = trade_analyzer.analyze_supply_chain_vulnerability({
        'geopolitical_tension': 0.6,
        'trade_war_intensity': 0.4,
        'technology_restrictions': 0.7
    })
    
    # Calculate comparative advantage
    comparative_advantages = trade_analyzer.calculate_comparative_advantage()
    
    # Generate trade policy recommendations
    trade_recommendations = trade_analyzer.generate_trade_policy_recommendations([
        'supply_chain_security', 
        'economic_efficiency', 
        'international_cooperation'
    ])
    
    # Optimize supply chain resilience
    resilience_optimization = trade_analyzer.optimize_supply_chain_resilience({
        'max_concentration': 0.3,  # Maximum 30% concentration in any single supplier
        'min_suppliers': 3,        # Minimum 3 suppliers per critical component
        'geographic_diversification': 0.6  # Target geographic diversification
    })
    
    # Compile comprehensive economic analysis
    economic_analysis = {
        'welfare_analysis': welfare_report,
        'supply_chain_vulnerability': vulnerability_analysis,
        'comparative_advantages': {
            ca_key: {
                'country': ca.country,
                'product': ca.product,
                'rca_index': ca.revealed_comparative_advantage,
                'export_complexity': ca.export_complexity,
                'opportunity_cost': ca.opportunity_cost
            }
            for ca_key, ca in comparative_advantages.items()
        },
        'trade_policy_recommendations': trade_recommendations,
        'resilience_optimization': resilience_optimization,
        'analysis_timestamp': datetime.now().isoformat(),
        'simulation_metadata': {
            'duration_months': len(simulation_results.time_series),
            'agents_analyzed': list(simulation_results.agent_performance.keys()),
            'final_market_size': simulation_results.time_series['total_market_revenue'].iloc[-1]
        }
    }
    
    # Save economic analysis results
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_file = output_dir / "economic_analysis.json"
    
    with open(analysis_file, 'w') as f:
        json.dump(economic_analysis, f, indent=2, default=str)
    
    logger.info(f"Economic analysis saved to {analysis_file}")
    
    # Print key economic insights
    logger.info("=== ECONOMIC ANALYSIS INSIGHTS ===")
    
    # Welfare insights
    optimal_carbon_price = welfare_report['optimal_policies']['optimal_carbon_price']
    logger.info(f"Optimal Carbon Price: ${optimal_carbon_price:.0f}/tonne CO2")
    
    best_welfare_scenario = welfare_report['optimal_policies']['highest_welfare']
    logger.info(f"Highest Welfare Scenario: {best_welfare_scenario}")
    
    # Supply chain insights
    overall_resilience = vulnerability_analysis['resilience_metrics']['overall_resilience']
    logger.info(f"Overall Supply Chain Resilience: {overall_resilience:.2f}/1.0")
    
    systemic_risk = vulnerability_analysis['chokepoint_analysis']['systemic_risk_score']
    logger.info(f"Systemic Risk Score: {systemic_risk:.2f}/1.0")
    
    # Trade insights
    critical_dependencies = len([
        product for product, risk in vulnerability_analysis['concentration_risk'].items()
        if risk['hhi'] > 0.4
    ])
    logger.info(f"Critical Supply Chain Dependencies: {critical_dependencies} products")
    
    return economic_analysis


def launch_dashboard() -> None:
    """Launch the interactive dashboard."""
    logger = logging.getLogger(__name__)
    
    if not DASHBOARD_AVAILABLE:
        logger.error("Dashboard dependencies not available. Install streamlit and plotly.")
        sys.exit(1)
    
    logger.info("Launching interactive dashboard...")
    dashboard = DatacenterDashboard()
    dashboard.run_dashboard()


def export_research_package(results: SimulationResults, economic_analysis: Dict[str, Any],
                          output_dir: Path = Path("research_package")) -> None:
    """Export comprehensive research package for academic use."""
    logger = logging.getLogger(__name__)
    logger.info("Exporting comprehensive research package...")
    
    # Create research package structure
    package_dir = output_dir / f"datacenter_simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    package_dir.mkdir(parents=True, exist_ok=True)
    
    # Data directory
    data_dir = package_dir / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Analysis directory
    analysis_dir = package_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    
    # Documentation directory
    docs_dir = package_dir / "documentation"
    docs_dir.mkdir(exist_ok=True)
    
    # Export simulation data
    results.time_series.to_csv(data_dir / "time_series_data.csv", index=False)
    
    for agent_id, df in results.agent_performance.items():
        df.to_csv(data_dir / f"agent_{agent_id}_performance.csv", index=False)
    
    # Export economic analysis
    with open(analysis_dir / "economic_analysis.json", 'w') as f:
        json.dump(economic_analysis, f, indent=2, default=str)
    
    with open(analysis_dir / "summary_statistics.json", 'w') as f:
        json.dump(results.summary_statistics, f, indent=2, default=str)
    
    # Generate methodology documentation
    methodology_doc = f"""
# Global Datacenter Capacity Planning Simulation: Methodology

## Overview
This research package contains the complete results and analysis from a comprehensive 
multi-agent simulation of global datacenter capacity planning under geopolitical risk.

## Simulation Architecture
- **Duration**: {len(results.time_series)} months
- **Agents**: {len(results.agent_performance)} hyperscaler companies
- **Market Scope**: Global datacenter infrastructure
- **Economic Framework**: Welfare economics and international trade theory

## Key Components
1. **Multi-Agent System**: Hyperscaler agents with strategic decision-making
2. **Economic Analysis**: Welfare economics and comparative advantage analysis  
3. **Supply Chain Modeling**: Semiconductor supply chain vulnerability assessment
4. **Geopolitical Risk**: Trade restrictions and technology export controls
5. **Sustainability Modeling**: Carbon pricing and renewable energy transitions

## Data Files
- `time_series_data.csv`: Main simulation time series
- `agent_*_performance.csv`: Individual agent performance data
- `economic_analysis.json`: Comprehensive economic welfare analysis
- `summary_statistics.json`: Simulation summary statistics

## Replication Instructions
1. Install dependencies: `pip install -r requirements.txt`
2. Run simulation: `python main.py --scenario baseline --duration 60`
3. Generate analysis: `python main.py --analysis economic --export-data`

## Citation
If using this simulation for research, please cite:
[Citation information would go here]

## Contact
[Contact information would go here]
"""
    
    with open(docs_dir / "methodology.md", 'w') as f:
        f.write(methodology_doc)
    
    # Generate requirements file
    requirements = """
numpy>=1.24.3
pandas>=2.0.3
scipy>=1.11.1
matplotlib>=3.7.2
seaborn>=0.12.2
plotly>=5.15.0
streamlit>=1.25.0
networkx>=3.1
scikit-learn>=1.3.0
pydantic>=2.1.1
"""
    
    with open(package_dir / "requirements.txt", 'w') as f:
        f.write(requirements)
    
    # Create README
    readme_content = f"""
# Global Datacenter Capacity Planning Simulation

Research package generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Explore data in the `data/` directory
3. Review analysis in the `analysis/` directory  
4. Read methodology in `documentation/methodology.md`

## Key Results
- Total Market Revenue: ${results.time_series['total_market_revenue'].iloc[-1]/1e9:.1f}B
- Average Demand Growth: {results.time_series['market_demand_growth'].mean():.1%}
- Supply Chain Resilience: {economic_analysis['supply_chain_vulnerability']['resilience_metrics']['overall_resilience']:.2f}/1.0

See documentation for complete methodology and analysis details.
"""
    
    with open(package_dir / "README.md", 'w') as f:
        f.write(readme_content)
    
    logger.info(f"Research package exported to {package_dir}")


def main():
    """Main entry point with comprehensive CLI interface."""
    parser = argparse.ArgumentParser(
        description="Global Datacenter Capacity Planning and Geopolitical Risk Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --generate-data
  %(prog)s --scenario baseline --duration 60
  %(prog)s --scenario-analysis baseline high_growth trade_war
  %(prog)s --analysis economic --export-data
  %(prog)s --dashboard
        """
    )
    
    # Main operation modes
    parser.add_argument('--generate-data', action='store_true',
                       help='Generate simulation datasets')
    parser.add_argument('--dashboard', action='store_true',
                       help='Launch interactive dashboard')
    
    # Simulation parameters
    parser.add_argument('--scenario', default='baseline',
                       choices=['baseline', 'high_growth', 'trade_war', 'green_transition', 'supply_crisis'],
                       help='Simulation scenario to run')
    parser.add_argument('--duration', type=int, default=60,
                       help='Simulation duration in months')
    parser.add_argument('--scenario-analysis', nargs='+', 
                       help='Run analysis across multiple scenarios')
    
    # Analysis options
    parser.add_argument('--analysis', choices=['economic', 'policy', 'trade'],
                       help='Type of post-simulation analysis to run')
    parser.add_argument('--export-data', action='store_true',
                       help='Export comprehensive research package')
    
    # Configuration and output
    parser.add_argument('--config', type=Path,
                       help='Configuration file path')
    parser.add_argument('--output', type=Path, default=Path('output'),
                       help='Output directory')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--log-file', type=Path,
                       help='Log file path')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    logger.info("=== GLOBAL DATACENTER CAPACITY PLANNING SIMULATION ===")
    logger.info(f"Started at {datetime.now()}")
    
    try:
        # Load configuration
        if args.config:
            load_config(args.config)
        
        # Data generation mode
        if args.generate_data:
            generate_simulation_data(args.config)
            return
        
        # Dashboard mode
        if args.dashboard:
            launch_dashboard()
            return
        
        # Simulation modes
        results = None
        economic_analysis = None
        
        if args.scenario_analysis:
            # Multi-scenario analysis
            scenario_results = run_scenario_analysis(args.scenario_analysis, args.duration, args.output)
            # Use baseline results for further analysis
            results = scenario_results.get('baseline', list(scenario_results.values())[0])
        else:
            # Single scenario
            results = run_baseline_simulation(args.duration, args.scenario, args.output)
        
        # Post-simulation analysis
        if args.analysis:
            if args.analysis in ['economic', 'policy', 'trade']:
                economic_analysis = run_economic_analysis(results, args.output)
            
            if args.analysis == 'policy':
                logger.info("Policy analysis completed - see economic_analysis.json for policy recommendations")
            
            if args.analysis == 'trade':
                logger.info("Trade analysis completed - see supply chain and trade flow analysis")
        
        # Export research package
        if args.export_data:
            if economic_analysis is None:
                logger.info("Running economic analysis for research package...")
                economic_analysis = run_economic_analysis(results, args.output)
            
            export_research_package(results, economic_analysis, args.output / "research_package")
        
        logger.info("=== SIMULATION COMPLETED SUCCESSFULLY ===")
        logger.info(f"Results saved to {args.output}")
        
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Simulation failed with error: {e}")
        raise


if __name__ == "__main__":
    main() 