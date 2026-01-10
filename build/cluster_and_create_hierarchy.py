#!/usr/bin/env python3
"""
Perform K-means clustering on the graph and create hierarchical nodes
"""
import json
import numpy as np
from sklearn.cluster import KMeans
from pathlib import Path
from typing import Dict, List, Any
import pickle
import networkx as nx
from datetime import datetime
import argparse


def load_graph(json_path: str) -> Dict[str, Any]:
    """Load graph data"""
    with open(json_path, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)
    return graph_data


def extract_embeddings_and_ids(nodes: List[Dict]) -> tuple:
    """Extract embeddings and IDs from nodes"""
    embeddings = []
    node_ids = []
    
    for node in nodes:
        if 'embedding' in node and node['embedding']:
            embeddings.append(node['embedding'])
            node_ids.append(node['id'])
    
    return np.array(embeddings), node_ids


def perform_kmeans(embeddings: np.ndarray, n_clusters: int = None) -> np.ndarray:
    """Perform K-means clustering"""
    # If number of clusters not specified, use heuristic method
    if n_clusters is None:
        n_samples = len(embeddings)
        # Heuristic: average 5-10 nodes per cluster
        n_clusters = max(2, min(n_samples // 5, 50))
    
    print(f"  Performing K-means clustering, number of clusters: {n_clusters}")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    
    return labels, kmeans.cluster_centers_


def create_cluster_nodes(graph_data: Dict, labels: np.ndarray, 
                        node_ids: List[str], cluster_centers: np.ndarray) -> Dict[str, Any]:
    """Create new graph with hierarchical cluster nodes"""
    
    # Deep copy original graph data
    new_graph = {
        'meta': graph_data['meta'].copy(),
        'nodes': graph_data['nodes'].copy(),
        'edges': graph_data['edges'].copy() if 'edges' in graph_data else []
    }
    
    # Update metadata
    new_graph['meta']['clustered'] = True
    new_graph['meta']['cluster_created_at'] = datetime.now().isoformat() + 'Z'
    new_graph['meta']['n_clusters'] = int(max(labels) + 1)
    
    # Create node_id to index mapping
    node_id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
    
    # Organize nodes by cluster
    clusters = {}
    for idx, label in enumerate(labels):
        cluster_id = int(label)
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(node_ids[idx])
    
    # Get maximum node ID number
    max_node_num = 0
    for node in new_graph['nodes']:
        node_id = node['id']
        if node_id.startswith('N'):
            try:
                num = int(node_id[1:])
                max_node_num = max(max_node_num, num)
            except:
                pass
    
    # Create high-level cluster nodes
    cluster_nodes = []
    cluster_node_mapping = {}  # cluster_id -> cluster_node_id
    
    for cluster_id, member_node_ids in clusters.items():
        cluster_node_id = f"C{cluster_id}"
        cluster_node_mapping[cluster_id] = cluster_node_id
        
        # Calculate average embedding within cluster (using original node embeddings)
        cluster_embeddings = []
        for member_id in member_node_ids:
            for node in new_graph['nodes']:
                if node['id'] == member_id and 'embedding' in node:
                    cluster_embeddings.append(node['embedding'])
                    break
        
        avg_embedding = np.mean(cluster_embeddings, axis=0).tolist()
        
        # Collect all session_ids, event_ids and other info within cluster
        all_session_ids = set()
        all_event_ids = set()
        all_people = set()
        
        for member_id in member_node_ids:
            for node in new_graph['nodes']:
                if node['id'] == member_id:
                    all_session_ids.update(node.get('session_ids', []))
                    all_event_ids.update(node.get('event_ids', []))
                    all_people.update(node.get('people', []))
                    break
        
        # Create cluster node
        cluster_node = {
            'id': cluster_node_id,
            'type': 'cluster',
            'cluster_id': cluster_id,
            'member_nodes': member_node_ids,
            'n_members': len(member_node_ids),
            'embedding': avg_embedding,
            'session_ids': sorted(list(all_session_ids)),
            'event_ids': sorted(list(all_event_ids)),
            'people': sorted(list(all_people)),
            'summaries': [f"Cluster {cluster_id} containing {len(member_node_ids)} nodes"]
        }
        
        cluster_nodes.append(cluster_node)
    
    # Add cluster nodes to graph
    new_graph['cluster_nodes'] = cluster_nodes
    
    # Create edges between cluster nodes and member nodes
    cluster_edges = []
    for cluster_node in cluster_nodes:
        cluster_node_id = cluster_node['id']
        for member_node_id in cluster_node['member_nodes']:
            edge = {
                'source': cluster_node_id,
                'target': member_node_id,
                'type': 'contains',
                'evidence': [f"Node {member_node_id} belongs to cluster {cluster_node['cluster_id']}"]
            }
            cluster_edges.append(edge)
    
    new_graph['cluster_edges'] = cluster_edges
    
    print(f"  Created {len(cluster_nodes)} cluster nodes")
    print(f"  Created {len(cluster_edges)} cluster edges")
    
    return new_graph


def create_networkx_graph(graph_data: Dict) -> nx.Graph:
    """Create NetworkX graph including cluster nodes and edges"""
    G = nx.Graph()
    
    # Add original nodes
    for node in graph_data['nodes']:
        G.add_node(node['id'], **node)
    
    # Add cluster nodes
    if 'cluster_nodes' in graph_data:
        for node in graph_data['cluster_nodes']:
            G.add_node(node['id'], **node)
    
    # Add original edges
    if 'edges' in graph_data:
        for edge in graph_data['edges']:
            G.add_edge(edge['source'], edge['target'], **edge)
    
    # Add cluster edges
    if 'cluster_edges' in graph_data:
        for edge in graph_data['cluster_edges']:
            G.add_edge(edge['source'], edge['target'], **edge)
    
    return G


def process_single_graph(input_path: str, output_dir: Path, n_clusters: int = None):
    """Process a single graph file"""
    print(f"\nProcessing graph: {input_path}")
    
    # Load graph
    graph_data = load_graph(input_path)
    
    # Extract embeddings
    embeddings, node_ids = extract_embeddings_and_ids(graph_data['nodes'])
    print(f"  Extracted embeddings from {len(embeddings)} nodes")
    
    if len(embeddings) < 2:
        print(f"  Too few nodes, skipping clustering")
        return
    
    # Perform K-means clustering
    labels, cluster_centers = perform_kmeans(embeddings, n_clusters)
    
    # Create new graph
    new_graph_data = create_cluster_nodes(graph_data, labels, node_ids, cluster_centers)
    
    # Save JSON file
    input_filename = Path(input_path).name
    output_json_path = output_dir / input_filename
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(new_graph_data, f, ensure_ascii=False, indent=2)
    print(f"  Saved JSON to: {output_json_path}")
    
    # Create and save NetworkX graph
    G = create_networkx_graph(new_graph_data)
    output_gpickle_path = output_dir / input_filename.replace('.json', '.gpickle')
    with open(output_gpickle_path, 'wb') as f:
        pickle.dump(G, f)
    print(f"  Saved gpickle to: {output_gpickle_path}")
    
    # Print clustering statistics
    print(f"  Clustering statistics:")
    for cluster_node in new_graph_data['cluster_nodes']:
        print(f"    Cluster {cluster_node['cluster_id']}: {cluster_node['n_members']} nodes")


def main():
    parser = argparse.ArgumentParser(description='Perform K-means clustering on graphs and create cluster nodes')
    parser.add_argument('--input_dir', type=str, 
                       default='./graphs',
                       help='Input directory containing graph files')
    parser.add_argument('--output_dir', type=str,
                       default='./output',
                       help='Output directory for processed graphs')
    parser.add_argument('--n_clusters', type=int, default=None,
                       help='Number of clusters (if not specified, auto-calculated)')
    parser.add_argument('--pattern', type=str, default='*.json',
                       help='File pattern to process')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Find all graph files
    graph_files = sorted(list(input_dir.glob(args.pattern)))
    print(f"\nFound {len(graph_files)} graph files")
    
    # Process each graph
    for graph_file in graph_files:
        try:
            process_single_graph(str(graph_file), output_dir, args.n_clusters)
        except Exception as e:
            print(f"Error processing {graph_file}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nComplete! All processed graphs saved to: {output_dir}")


if __name__ == '__main__':
    main()
