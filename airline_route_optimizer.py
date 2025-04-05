import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import streamlit as st
from itertools import islice
from typing import List, Dict

class RouteOptimizer:
    """Optimizes airline routes using graph theory metrics"""

    def __init__(self, data_path: str):
        self._load_data(data_path)
        self._build_networks()
        self._compute_layouts()

    def _load_data(self, data_path: str):
        """Load and preprocess flight data"""
        raw = pd.read_csv(data_path)
        self.data = {
            'distance': raw[['origin', 'dest', 'distance']],
            'time': self._calculate_times(raw),
            'composite': self._calculate_composite_scores(raw)
        }

    def _calculate_times(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert hours+minutes to total minutes"""
        return df.assign(
            time=df['hour'] * 60 + df['minute']
        )[['origin', 'dest', 'time']]

    def _calculate_composite_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate combined distance/time scores"""
        max_dist = df['distance'].max()
        return df.assign(
            composite_score=0.7*(df['distance']/max_dist) +
            0.3*((df['hour']*60 + df['minute'])/1440)
        )[['origin', 'dest', 'composite_score']]

    def _build_networks(self):
        """Construct all network graphs"""
        self.graphs = {
            'distance': self._create_graph('distance'),
            'time': self._create_graph('time'),
            'composite': self._create_graph('composite_score')
        }

    def _create_graph(self, weight_attr: str) -> nx.Graph:
        """Build weighted network graph"""
        G = nx.Graph()
        edges = [tuple(x) for x in self.data[
            'distance' if weight_attr == 'distance'
            else 'time' if weight_attr == 'time'
            else 'composite'
        ].values]
        G.add_weighted_edges_from(edges, weight=weight_attr)
        return G

    def _compute_layouts(self):
        """Precompute node positions for visualization"""
        self.layouts = {
            name: nx.spring_layout(G, k=0.15, iterations=50)
            for name, G in self.graphs.items()
        }

    def find_routes(self, source: str, target: str,
                   k: int = 3, metric: str = 'composite') -> Dict:
        """Find top k optimal routes"""
        weight_attr = 'composite_score' if metric == 'composite' else metric
        path_gen = nx.shortest_simple_paths(
            self.graphs[metric],
            source,
            target,
            weight=weight_attr
        )

        return {
            f"path_{i+1}": self._calculate_route_metrics(path, metric)
            for i, path in enumerate(islice(path_gen, k))
        }

    def _calculate_route_metrics(self, path: List[str], metric: str) -> Dict:
        """Calculate all metrics for a given route"""
        return {
            'route': path,
            'distance': self._sum_edge_weights(path, 'distance'),
            'time': self._sum_edge_weights(path, 'time'),
            'composite_score': self._sum_edge_weights(path, 'composite')
        }

    def _sum_edge_weights(self, path: List[str], metric: str) -> float:
        """Sum weights along a path for given metric"""
        G = self.graphs[metric]
        weight_key = {
            'distance': 'distance',
            'time': 'time',
            'composite': 'composite_score'
        }[metric]

        return sum(
            G[u][v][weight_key]
            for u, v in zip(path, path[1:])
        )

    def visualize(self, paths: List[List[str]], metric: str = 'composite'):
        """Create interactive visualization"""
        fig = go.Figure()
        self._add_base_network(fig, metric)
        self._highlight_paths(fig, paths, metric)
        self._finalize_layout(fig, metric)
        st.plotly_chart(fig, use_container_width=True)

    def _add_base_network(self, fig: go.Figure, metric: str):
        """Add nodes and edges to visualization with improved readability"""
        G = self.graphs[metric]
        pos = self.layouts[metric]

        # Add edges first (background)
        edge_x, edge_y = [], []
        for u, v in G.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.8, color='rgba(200,200,200,0.4)'),
            hoverinfo='none',
            mode='lines'
        ))

        # Calculate centrality measures
        betweenness = nx.betweenness_centrality(G)
        max_bc = max(betweenness.values()) if betweenness else 1

        # Add nodes with better text visibility
        node_x = [pos[n][0] for n in G.nodes()]
        node_y = [pos[n][1] for n in G.nodes()]

        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            marker=dict(
                size=[15 + 25*(betweenness[n]/max_bc) for n in G.nodes()],
                color=[betweenness[n] for n in G.nodes()],
                colorscale='Plasma',
                showscale=True,
                colorbar=dict(
                    title='Betweenness<br>Centrality',
                    title_font=dict(size=12),
                    tickfont=dict(size=10),
                    thickness=20,
                    len=0.5,
                    xanchor='left',
                    yanchor='middle'
                ),
                line=dict(width=2, color='white')
            ),
            hovertext=[
                f"<b>{node}</b><br>"
                f"Betweenness: {betweenness[node]:.3f}<br>"
                f"Connections: {G.degree(node)}"
                for node in G.nodes()
            ],
            hoverinfo='text',
            text=list(G.nodes()),
            textposition='middle center',
            textfont=dict(
                color='black',
                size=12,
                family='Arial'
            )
        ))

    def _highlight_paths(self, fig: go.Figure, paths: List[List[str]], metric: str):
        """Highlight optimal paths and endpoints on visualization"""
        pos = self.layouts[metric]
        colors = ['#FF0000', '#00AA00', '#0000FF']

        # Get origin and destination from first path
        origin = paths[0][0]
        destination = paths[0][-1]

        # Highlight endpoints with special colors
        endpoint_colors = {
            origin: '#F542F2',    # Pink for origin
            destination: '#00FFFB' # Sky blue for destination
        }

        # Add special markers for endpoints
        fig.add_trace(go.Scatter(
            x=[pos[origin][0], pos[destination][0]],
            y=[pos[origin][1], pos[destination][1]],
            mode='markers',
            marker=dict(
                size=25,
                color=[endpoint_colors[origin], endpoint_colors[destination]],
                line=dict(width=3, color='white')
            ),
            hoverinfo='none',
            showlegend=False
        ))

        # Add path traces
        for i, path in enumerate(paths[:3]):
            path_edges = list(zip(path, path[1:]))
            path_x, path_y = [], []
            for u, v in path_edges:
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                path_x.extend([x0, x1, None])
                path_y.extend([y0, y1, None])

            fig.add_trace(go.Scatter(
                x=path_x, y=path_y,
                line=dict(width=3, color=colors[i]),
                mode='lines',
                name=f'Path {i+1}',
                hoverinfo='none'
            ))

    def _finalize_layout(self, fig: go.Figure, metric: str):
        """Configure visualization layout"""
        fig.update_layout(
            title=f'<b>Optimal Routes ({metric.title()})</b>',
            showlegend=True,
            hovermode='closest',
            margin=dict(b=25, l=25, r=25, t=50),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='rgba(250,250,250,1)',
            height=800
        )

    def analyze(self) -> Dict:
        """Compute network metrics"""
        metrics = {}
        for name, G in self.graphs.items():
            weight_attr = 'composite_score' if name == 'composite' else name
            try:
                mst_weight = sum(d[weight_attr] for _, _, d in nx.minimum_spanning_edges(G, data=True))
            except KeyError:
                mst_weight = float('nan')

            metrics[name] = {
                'clustering': nx.average_clustering(G),
                'diameter': nx.diameter(G) if nx.is_connected(G) else float('inf'),
                'density': nx.density(G),
                'assortativity': nx.degree_assortativity_coefficient(G),
                'mst_weight': mst_weight
            }
        return metrics

def main():
    st.set_page_config(layout="wide", page_title="Airline Route Optimizer")

    st.title("✈️ Airline Route Optimizer")
    st.markdown("""
    This tool finds optimal flight routes between airports using graph theory metrics.
    Select your origin and destination airports below to see recommended routes.
    """)

    # Initialize the optimizer
    if 'optimizer' not in st.session_state:
        with st.spinner("Loading flight data and building networks..."):
            st.session_state.optimizer = RouteOptimizer("flights.csv")
            st.session_state.airports = sorted(st.session_state.optimizer.graphs['distance'].nodes())

    # Create sidebar controls
    with st.sidebar:
        st.header("Route Parameters")
        src = st.selectbox("Origin Airport", st.session_state.airports)
        dest = st.selectbox("Destination Airport", st.session_state.airports)

        st.markdown("---")
        st.header("Network Information")

        # Enhanced node color information with colored badges
        st.markdown("### Node Color Information")
        st.markdown("""
        <div style="display: flex; flex-direction: column; gap: 8px;">
            <div style="display: flex; align-items: center;">
                <div style="width: 20px; height: 20px; background-color: #F542F2; border-radius: 50%; margin-right: 10px;"></div>
                <span>Origin Airport</span>
            </div>
            <div style="display: flex; align-items: center;">
                <div style="width: 20px; height: 20px; background-color: #00FFFB; border-radius: 50%; margin-right: 10px;"></div>
                <span>Destination Airport</span>
            </div>
            <div style="display: flex; align-items: center;">
                <div style="width: 20px; height: 20px; background: linear-gradient(to right, #FF0000, #FFA500); border-radius: 50%; margin-right: 10px;"></div>
                <span>Critical hubs (high betweenness)</span>
            </div>
            <div style="display: flex; align-items: center;">
                <div style="width: 20px; height: 20px; background-color: #FFFF00; border-radius: 50%; margin-right: 10px;"></div>
                <span>Important connectors</span>
            </div>
            <div style="display: flex; align-items: center;">
                <div style="width: 20px; height: 20px; background-color: #800080; border-radius: 50%; margin-right: 10px;"></div>
                <span>Less central airports</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")
        st.markdown("**Size** indicates importance (larger = more central)")

        st.markdown("---")
        st.header("Algorithm Details")
        st.markdown("""
        ### Route Optimization Mathematics

        **1. Composite Score Calculation:**<br>
        The composite score provides a single, balanced metric that combines both flight distance and duration into one value, weighing:<br>
        ```
        composite_score = 0.7*(distance/max_distance) + 0.3*(flight_time/1440)
        ```
        - The time term is normalized by 1440 (theoretical max flight time in a day, i.e. 24 hours = 1440 minutes).

        **2. Shortest Path Algorithm:**<br>
        Uses Yen's algorithm to find k shortest paths with weights based on selected metric.

        **3. Betweenness Centrality:**
        ```
        CB(v) = Σ(s≠v≠t) σst(v)/σst
        ```
        - Where σst is total shortest paths from s to t, and σst(v) are paths through v.

        **4. Network Metrics:**
        - **Clustering Coefficient:** Measures degree of node clustering
        - **Diameter:** Longest shortest path in the network
        - **Density:** Ratio of actual edges to possible edges
        - **Assortativity:** Degree correlation between connected nodes
        - **MST Weight:** Total weight of minimum spanning tree
        """, unsafe_allow_html=True)

    if st.button("Find Optimal Routes"):
        with st.spinner("Calculating optimal routes..."):
            # Find routes (default to composite metric)
            routes = st.session_state.optimizer.find_routes(src, dest)

            # Display routes with smaller font and full information
            st.subheader("Optimal Routes")
            cols = st.columns(3)
            for i, (name, data) in enumerate(routes.items()):
                with cols[i]:
                    st.markdown(f"<h4 style='font-size: 20px;'>Path {i+1}:</h4>", unsafe_allow_html=True)
                    st.markdown(f"<p style='font-size: 18px;'>Route: {' → '.join(data['route'])}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='font-size: 16px;'>Distance: {data['distance']:,.0f} miles</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='font-size: 16px;'>Time: {data['time']//60}h {data['time']%60}m</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='font-size: 16px;'>Composite Score: {data['composite_score']:.3f}</p>", unsafe_allow_html=True)

            # Visualize with metric tabs
            st.subheader("Route Visualization")

            # Metric selection tabs
            tab1, tab2, tab3 = st.tabs(["Composite", "Distance", "Time"])

            with tab1:
                st.session_state.optimizer.visualize(
                    [data['route'] for data in routes.values()],
                    metric='composite'
                )
            with tab2:
                st.session_state.optimizer.visualize(
                    [data['route'] for data in routes.values()],
                    metric='distance'
                )
            with tab3:
                st.session_state.optimizer.visualize(
                    [data['route'] for data in routes.values()],
                    metric='time'
                )

            # Show metrics in table format
            st.subheader("Network Metrics")
            stats = st.session_state.optimizer.analyze()

            # Create a DataFrame for the metrics
            metrics_df = pd.DataFrame({
                'Metric': ['Composite', 'Distance', 'Time'],
                'Clustering': [stats['composite']['clustering'], stats['distance']['clustering'], stats['time']['clustering']],
                'Diameter': [stats['composite']['diameter'], stats['distance']['diameter'], stats['time']['diameter']],
                'Density': [stats['composite']['density'], stats['distance']['density'], stats['time']['density']],
                'Assortativity': [stats['composite']['assortativity'], stats['distance']['assortativity'], stats['time']['assortativity']],
                'MST Weight': [stats['composite']['mst_weight'], stats['distance']['mst_weight'], stats['time']['mst_weight']]
            })

            # Display the table with improved formatting
            st.dataframe(
                metrics_df.style.format({
                    'Clustering': '{:.3f}',
                    'Density': '{:.4f}',
                    'Assortativity': '{:.3f}',
                    'MST Weight': '{:,.1f}'
                }),
                use_container_width=True
            )

if __name__ == "__main__":
    main()
