"""
Interactive dashboard for datacenter simulation visualization.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, Any, Optional

from ..core.simulation_engine import SimulationEngine, SimulationResults
from ..core.config import get_config


class DatacenterDashboard:
    """Interactive dashboard for simulation results."""
    
    def __init__(self):
        """Initialize dashboard."""
        self.config = get_config()
        self.simulation_engine = None
        self.results = None
    
    def run_dashboard(self) -> None:
        """Run the Streamlit dashboard."""
        st.set_page_config(
            page_title="Datacenter Capacity Planning Simulation",
            page_icon="🏢",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🏢 Global Datacenter Capacity Planning Simulation")
        st.markdown("---")
        
        # Sidebar controls
        self._render_sidebar()
        
        # Main content
        if self.results is not None:
            self._render_main_content()
        else:
            self._render_welcome_screen()
    
    def _render_sidebar(self) -> None:
        """Render sidebar controls."""
        st.sidebar.header("Simulation Controls")
        
        # Scenario selection
        scenario = st.sidebar.selectbox(
            "Select Scenario",
            ["baseline", "high_growth", "trade_war", "green_transition", "supply_crisis"],
            help="Choose a predefined scenario to simulate"
        )
        
        # Simulation parameters
        st.sidebar.subheader("Parameters")
        
        duration = st.sidebar.slider(
            "Simulation Duration (months)",
            min_value=12,
            max_value=120,
            value=60,
            step=6
        )
        
        demand_growth = st.sidebar.slider(
            "Global Demand Growth Rate",
            min_value=0.1,
            max_value=0.6,
            value=0.3,
            step=0.05,
            format="%.1f%%"
        )
        
        supply_shortage = st.sidebar.slider(
            "Semiconductor Supply Shortage",
            min_value=0.0,
            max_value=0.8,
            value=0.4,
            step=0.05,
            format="%.1f%%"
        )
        
        geopolitical_tension = st.sidebar.slider(
            "Geopolitical Tension Level",
            min_value=0.0,
            max_value=0.8,
            value=0.3,
            step=0.05,
            format="%.1f%%"
        )
        
        # Run simulation button
        if st.sidebar.button("Run Simulation", type="primary"):
            with st.spinner("Running simulation..."):
                self._run_simulation(scenario, duration, {
                    'demand_growth': demand_growth,
                    'supply_shortage': supply_shortage,
                    'geopolitical_tension': geopolitical_tension
                })
                st.success("Simulation completed!")
                st.rerun()
        
        # Download results
        if self.results is not None:
            st.sidebar.markdown("---")
            st.sidebar.subheader("Export Results")
            
            if st.sidebar.button("Download CSV"):
                self._download_results()
    
    def _render_welcome_screen(self) -> None:
        """Render welcome screen when no simulation has been run."""
        st.markdown("""
        ## Welcome to the Datacenter Capacity Planning Simulation
        
        This simulation models the complex interactions between:
        - **Hyperscaler companies** (Google, Amazon, Microsoft, Meta)
        - **Semiconductor supply chains** and manufacturing constraints
        - **Geopolitical tensions** and trade restrictions
        - **Energy markets** and sustainability requirements
        - **Global demand** for datacenter capacity
        
        ### How to use:
        1. Select a scenario from the sidebar
        2. Adjust simulation parameters as needed
        3. Click "Run Simulation" to begin
        4. Explore the results in the interactive visualizations
        
        ### Scenarios Available:
        - **Baseline**: Current market conditions
        - **High Growth**: Accelerated AI demand growth
        - **Trade War**: Increased geopolitical tensions
        - **Green Transition**: Carbon pricing and renewable mandates
        - **Supply Crisis**: Severe semiconductor shortages
        """)
        
        # Add some sample visualizations or mockups
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sample Market Evolution")
            # Create sample data for demo
            sample_data = pd.DataFrame({
                'Month': range(60),
                'Demand Growth': 0.3 + 0.1 * np.sin(np.arange(60) * 0.1) + np.random.normal(0, 0.02, 60),
                'Supply Shortage': 0.4 * (1 - np.arange(60) / 120) + np.random.normal(0, 0.05, 60)
            })
            
            fig = px.line(sample_data, x='Month', y=['Demand Growth', 'Supply Shortage'],
                         title="Market Conditions Over Time")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Sample Agent Performance")
            # Sample agent data
            agents = ['Google', 'Amazon', 'Microsoft', 'Meta']
            sample_performance = pd.DataFrame({
                'Agent': agents,
                'Capacity (MW)': [2500, 4000, 3200, 1800],
                'Revenue ($B)': [45, 75, 58, 32]
            })
            
            fig = px.bar(sample_performance, x='Agent', y='Capacity (MW)',
                        title="Current Datacenter Capacity by Agent")
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_main_content(self) -> None:
        """Render main dashboard content with simulation results."""
        # Key metrics
        self._render_key_metrics()
        
        # Time series visualizations
        self._render_time_series()
        
        # Agent performance
        self._render_agent_performance()
        
        # Market analysis
        self._render_market_analysis()
        
        # Geopolitical impact
        self._render_geopolitical_analysis()
    
    def _render_key_metrics(self) -> None:
        """Render key performance metrics."""
        st.subheader("📊 Key Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_revenue = self.results.summary_statistics['market']['total_revenue_final']
            st.metric(
                "Total Market Revenue",
                f"${total_revenue/1e9:.1f}B",
                delta=f"+{(total_revenue/1e9)*0.15:.1f}B"
            )
        
        with col2:
            avg_demand = self.results.summary_statistics['market']['average_demand_growth']
            st.metric(
                "Avg Demand Growth",
                f"{avg_demand:.1%}",
                delta=f"{(avg_demand-0.3):.1%}"
            )
        
        with col3:
            final_shortage = self.results.time_series['semiconductor_shortage'].iloc[-1]
            st.metric(
                "Supply Shortage",
                f"{final_shortage:.1%}",
                delta=f"{(final_shortage-0.4):.1%}",
                delta_color="inverse"
            )
        
        with col4:
            total_capacity = sum(
                stats.get('final_capacity_mw', 0) 
                for stats in self.results.summary_statistics['agents'].values()
            )
            st.metric(
                "Total Capacity",
                f"{total_capacity/1000:.1f} GW",
                delta=f"+{total_capacity*0.1/1000:.1f} GW"
            )
    
    def _render_time_series(self) -> None:
        """Render time series visualizations."""
        st.subheader("📈 Market Evolution Over Time")
        
        # Market conditions
        fig_market = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Demand Growth', 'Supply Shortage', 'Geopolitical Tension', 'Total Revenue'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": True}]]
        )
        
        # Demand growth
        fig_market.add_trace(
            go.Scatter(x=self.results.time_series['time'], 
                      y=self.results.time_series['market_demand_growth'],
                      name='Demand Growth', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Supply shortage
        fig_market.add_trace(
            go.Scatter(x=self.results.time_series['time'], 
                      y=self.results.time_series['semiconductor_shortage'],
                      name='Supply Shortage', line=dict(color='red')),
            row=1, col=2
        )
        
        # Geopolitical tension
        fig_market.add_trace(
            go.Scatter(x=self.results.time_series['time'], 
                      y=self.results.time_series['geopolitical_tension'],
                      name='Geopolitical Tension', line=dict(color='orange')),
            row=2, col=1
        )
        
        # Total revenue
        fig_market.add_trace(
            go.Scatter(x=self.results.time_series['time'], 
                      y=self.results.time_series['total_market_revenue']/1e9,
                      name='Revenue ($B)', line=dict(color='green')),
            row=2, col=2
        )
        
        fig_market.update_layout(height=600, showlegend=False)
        fig_market.update_xaxes(title_text="Time (Months)")
        fig_market.update_yaxes(title_text="Rate", row=1, col=1)
        fig_market.update_yaxes(title_text="Shortage Level", row=1, col=2)
        fig_market.update_yaxes(title_text="Tension Level", row=2, col=1)
        fig_market.update_yaxes(title_text="Revenue ($B)", row=2, col=2)
        
        st.plotly_chart(fig_market, use_container_width=True)
    
    def _render_agent_performance(self) -> None:
        """Render agent performance analysis."""
        st.subheader("🏢 Agent Performance Analysis")
        
        # Agent comparison
        col1, col2 = st.columns(2)
        
        with col1:
            # Final capacity comparison
            agent_data = []
            for agent_id, stats in self.results.summary_statistics['agents'].items():
                agent_data.append({
                    'Agent': agent_id.title(),
                    'Capacity (MW)': stats.get('final_capacity_mw', 0),
                    'Revenue ($B)': stats.get('total_revenue', 0) / 1e9,
                    'Profit Margin': stats.get('profit_margin', 0) * 100
                })
            
            agent_df = pd.DataFrame(agent_data)
            
            fig_capacity = px.bar(agent_df, x='Agent', y='Capacity (MW)',
                                title="Final Datacenter Capacity by Agent",
                                color='Agent')
            st.plotly_chart(fig_capacity, use_container_width=True)
        
        with col2:
            # Revenue performance
            fig_revenue = px.bar(agent_df, x='Agent', y='Revenue ($B)',
                               title="Total Revenue by Agent",
                               color='Agent')
            st.plotly_chart(fig_revenue, use_container_width=True)
        
        # Agent performance over time
        st.subheader("Performance Trends")
        
        agent_selector = st.selectbox(
            "Select Agent for Detailed Analysis",
            list(self.results.agent_performance.keys())
        )
        
        if agent_selector:
            agent_data = self.results.agent_performance[agent_selector]
            
            fig_agent = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Cash Balance', 'Capacity Growth', 'Revenue', 'Utilization'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Cash balance
            fig_agent.add_trace(
                go.Scatter(x=agent_data['time'], y=agent_data['cash_balance']/1e9,
                          name='Cash Balance ($B)'),
                row=1, col=1
            )
            
            # Capacity
            if 'capacity_mw' in agent_data.columns:
                fig_agent.add_trace(
                    go.Scatter(x=agent_data['time'], y=agent_data['capacity_mw'],
                              name='Capacity (MW)'),
                    row=1, col=2
                )
            
            # Revenue
            fig_agent.add_trace(
                go.Scatter(x=agent_data['time'], y=agent_data['revenue']/1e6,
                          name='Monthly Revenue ($M)'),
                row=2, col=1
            )
            
            # Utilization
            if 'utilization_rate' in agent_data.columns:
                fig_agent.add_trace(
                    go.Scatter(x=agent_data['time'], y=agent_data['utilization_rate'],
                              name='Utilization Rate'),
                    row=2, col=2
                )
            
            fig_agent.update_layout(height=600, showlegend=False)
            fig_agent.update_xaxes(title_text="Time (Months)")
            
            st.plotly_chart(fig_agent, use_container_width=True)
    
    def _render_market_analysis(self) -> None:
        """Render market analysis."""
        st.subheader("🌍 Market Analysis")
        
        # Policy impact analysis
        if hasattr(self.results, 'policy_impacts'):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Carbon Pricing Impact")
                carbon_impact = self.results.policy_impacts.get('carbon_pricing', {})
                
                st.metric("Cost Correlation", f"{carbon_impact.get('cost_correlation', 0):.2f}")
                st.metric("Price Elasticity", f"{carbon_impact.get('price_elasticity', 0):.2f}")
                
                if carbon_impact.get('revenue_impact') == 'negative':
                    st.warning("Carbon pricing has negative impact on revenue")
                else:
                    st.success("Carbon pricing impact is minimal")
            
            with col2:
                st.markdown("### Supply Chain Impact")
                supply_impact = self.results.policy_impacts.get('supply_chain', {})
                
                st.metric("Revenue Correlation", f"{supply_impact.get('revenue_correlation', 0):.2f}")
                
                disruption_level = supply_impact.get('disruption_severity', 'moderate')
                if disruption_level == 'high':
                    st.error("High supply chain disruption severity")
                else:
                    st.info(f"Supply chain disruption: {disruption_level}")
                
                recovery = supply_impact.get('recovery_pattern', 'unknown')
                st.info(f"Recovery pattern: {recovery.replace('_', ' ').title()}")
        
        # Correlation analysis
        st.markdown("### Market Correlations")
        
        correlation_data = self.results.time_series[[
            'market_demand_growth', 'semiconductor_shortage', 
            'geopolitical_tension', 'total_market_revenue'
        ]].corr()
        
        fig_corr = px.imshow(correlation_data, 
                           title="Market Variables Correlation Matrix",
                           color_continuous_scale='RdBu',
                           aspect="auto")
        st.plotly_chart(fig_corr, use_container_width=True)
    
    def _render_geopolitical_analysis(self) -> None:
        """Render geopolitical risk analysis."""
        st.subheader("🌐 Geopolitical Risk Analysis")
        
        # Risk timeline
        fig_risk = go.Figure()
        
        fig_risk.add_trace(go.Scatter(
            x=self.results.time_series['time'],
            y=self.results.time_series['geopolitical_tension'],
            mode='lines+markers',
            name='Geopolitical Tension',
            line=dict(color='red', width=3)
        ))
        
        # Add risk level annotations
        high_risk_periods = self.results.time_series['geopolitical_tension'] > 0.6
        if high_risk_periods.any():
            fig_risk.add_hline(y=0.6, line_dash="dash", line_color="red",
                             annotation_text="High Risk Threshold")
        
        fig_risk.update_layout(
            title="Geopolitical Risk Evolution",
            xaxis_title="Time (Months)",
            yaxis_title="Risk Level",
            height=400
        )
        
        st.plotly_chart(fig_risk, use_container_width=True)
        
        # Risk impact summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_tension = self.results.time_series['geopolitical_tension'].mean()
            if avg_tension > 0.5:
                st.error(f"High average tension: {avg_tension:.1%}")
            else:
                st.success(f"Moderate tension: {avg_tension:.1%}")
        
        with col2:
            max_tension = self.results.time_series['geopolitical_tension'].max()
            st.metric("Peak Tension", f"{max_tension:.1%}")
        
        with col3:
            tension_volatility = self.results.time_series['geopolitical_tension'].std()
            st.metric("Risk Volatility", f"{tension_volatility:.1%}")
    
    def _run_simulation(self, scenario: str, duration: int, params: Dict[str, Any]) -> None:
        """Run the simulation with given parameters."""
        # Initialize simulation engine
        self.simulation_engine = SimulationEngine()
        
        # Update configuration with custom parameters
        self.simulation_engine.state.global_demand_growth = params['demand_growth']
        self.simulation_engine.state.semiconductor_shortage = params['supply_shortage']
        self.simulation_engine.state.geopolitical_tension = params['geopolitical_tension']
        
        # Initialize and run
        self.simulation_engine.initialize(scenario)
        self.results = self.simulation_engine.run(duration)
    
    def _download_results(self) -> None:
        """Provide download functionality for results."""
        # Convert results to CSV format
        csv_data = self.results.time_series.to_csv(index=False)
        
        st.sidebar.download_button(
            label="Download Time Series",
            data=csv_data,
            file_name="simulation_results.csv",
            mime="text/csv"
        )


def main():
    """Main function to run the dashboard."""
    dashboard = DatacenterDashboard()
    dashboard.run_dashboard()


if __name__ == "__main__":
    main() 