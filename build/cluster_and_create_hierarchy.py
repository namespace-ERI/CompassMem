#!/usr/bin/env python3
"""
对图进行K-means聚类，并创建高层节点
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
    """加载图数据"""
    with open(json_path, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)
    return graph_data


def extract_embeddings_and_ids(nodes: List[Dict]) -> tuple:
    """提取节点的embedding和ID"""
    embeddings = []
    node_ids = []
    
    for node in nodes:
        if 'embedding' in node and node['embedding']:
            embeddings.append(node['embedding'])
            node_ids.append(node['id'])
    
    return np.array(embeddings), node_ids


def perform_kmeans(embeddings: np.ndarray, n_clusters: int = None) -> np.ndarray:
    """执行K-means聚类"""
    # 如果没有指定聚类数，使用启发式方法
    if n_clusters is None:
        n_samples = len(embeddings)
        # 启发式：每个簇平均10-20个节点
        n_clusters = max(2, min(n_samples // 5, 50))
    
    print(f"  执行K-means聚类，聚类数: {n_clusters}")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    
    return labels, kmeans.cluster_centers_


def create_cluster_nodes(graph_data: Dict, labels: np.ndarray, 
                        node_ids: List[str], cluster_centers: np.ndarray) -> Dict[str, Any]:
    """创建包含高层聚类节点的新图"""
    
    # 深拷贝原始图数据
    new_graph = {
        'meta': graph_data['meta'].copy(),
        'nodes': graph_data['nodes'].copy(),
        'edges': graph_data['edges'].copy() if 'edges' in graph_data else []
    }
    
    # 更新元数据
    new_graph['meta']['clustered'] = True
    new_graph['meta']['cluster_created_at'] = datetime.now().isoformat() + 'Z'
    new_graph['meta']['n_clusters'] = int(max(labels) + 1)
    
    # 创建node_id到索引的映射
    node_id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
    
    # 按簇组织节点
    clusters = {}
    for idx, label in enumerate(labels):
        cluster_id = int(label)
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(node_ids[idx])
    
    # 获取当前最大的节点ID数字
    max_node_num = 0
    for node in new_graph['nodes']:
        node_id = node['id']
        if node_id.startswith('N'):
            try:
                num = int(node_id[1:])
                max_node_num = max(max_node_num, num)
            except:
                pass
    
    # 创建高层节点
    cluster_nodes = []
    cluster_node_mapping = {}  # cluster_id -> cluster_node_id
    
    for cluster_id, member_node_ids in clusters.items():
        cluster_node_id = f"C{cluster_id}"
        cluster_node_mapping[cluster_id] = cluster_node_id
        
        # 计算簇内embedding的平均值（使用原始节点的embedding）
        cluster_embeddings = []
        for member_id in member_node_ids:
            for node in new_graph['nodes']:
                if node['id'] == member_id and 'embedding' in node:
                    cluster_embeddings.append(node['embedding'])
                    break
        
        avg_embedding = np.mean(cluster_embeddings, axis=0).tolist()
        
        # 收集簇内所有session_ids, event_ids等信息
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
        
        # 创建聚类节点
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
    
    # 将高层节点添加到图中
    new_graph['cluster_nodes'] = cluster_nodes
    
    # 创建高层节点与成员节点之间的边
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
    
    print(f"  创建了 {len(cluster_nodes)} 个高层聚类节点")
    print(f"  创建了 {len(cluster_edges)} 条聚类边")
    
    return new_graph


def create_networkx_graph(graph_data: Dict) -> nx.Graph:
    """创建NetworkX图（包含聚类节点和边）"""
    G = nx.Graph()
    
    # 添加原始节点
    for node in graph_data['nodes']:
        G.add_node(node['id'], **node)
    
    # 添加聚类节点
    if 'cluster_nodes' in graph_data:
        for node in graph_data['cluster_nodes']:
            G.add_node(node['id'], **node)
    
    # 添加原始边
    if 'edges' in graph_data:
        for edge in graph_data['edges']:
            G.add_edge(edge['source'], edge['target'], **edge)
    
    # 添加聚类边
    if 'cluster_edges' in graph_data:
        for edge in graph_data['cluster_edges']:
            G.add_edge(edge['source'], edge['target'], **edge)
    
    return G


def process_single_graph(input_path: str, output_dir: Path, n_clusters: int = None):
    """处理单个图文件"""
    print(f"\n处理图: {input_path}")
    
    # 加载图
    graph_data = load_graph(input_path)
    
    # 提取embeddings
    embeddings, node_ids = extract_embeddings_and_ids(graph_data['nodes'])
    print(f"  提取了 {len(embeddings)} 个节点的embedding")
    
    if len(embeddings) < 2:
        print(f"  节点数太少，跳过聚类")
        return
    
    # 执行K-means聚类
    labels, cluster_centers = perform_kmeans(embeddings, n_clusters)
    
    # 创建新图
    new_graph_data = create_cluster_nodes(graph_data, labels, node_ids, cluster_centers)
    
    # 保存JSON文件
    input_filename = Path(input_path).name
    output_json_path = output_dir / input_filename
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(new_graph_data, f, ensure_ascii=False, indent=2)
    print(f"  保存JSON到: {output_json_path}")
    
    # 创建并保存NetworkX图
    G = create_networkx_graph(new_graph_data)
    output_gpickle_path = output_dir / input_filename.replace('.json', '.gpickle')
    with open(output_gpickle_path, 'wb') as f:
        pickle.dump(G, f)
    print(f"  保存gpickle到: {output_gpickle_path}")
    
    # 打印聚类统计信息
    print(f"  聚类统计:")
    for cluster_node in new_graph_data['cluster_nodes']:
        print(f"    Cluster {cluster_node['cluster_id']}: {cluster_node['n_members']} 个节点")


def main():
    parser = argparse.ArgumentParser(description='对图进行K-means聚类并创建高层节点')
    parser.add_argument('--input_dir', type=str, 
                       default='/share/project/zyt/hyy/Memory/build_graph/graphs_llm_bge',
                       help='输入图文件所在目录')
    parser.add_argument('--output_dir', type=str,
                       default='/share/project/zyt/hyy/Memory/build_graph/graphs_llm_clustered_bge',
                       help='输出图文件所在目录')
    parser.add_argument('--n_clusters', type=int, default=None,
                       help='聚类数量（如果不指定，自动计算）')
    parser.add_argument('--pattern', type=str, default='*.json',
                       help='要处理的文件模式')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {output_dir}")
    
    # 查找所有图文件
    graph_files = sorted(list(input_dir.glob(args.pattern)))
    print(f"\n找到 {len(graph_files)} 个图文件")
    
    # 处理每个图
    for graph_file in graph_files:
        try:
            process_single_graph(str(graph_file), output_dir, args.n_clusters)
        except Exception as e:
            print(f"处理 {graph_file} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n完成！所有处理后的图已保存到: {output_dir}")


if __name__ == '__main__':
    main()

