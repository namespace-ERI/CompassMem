#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build per-item event graphs from two-stage extraction JSONL.

- Each JSON line contains a single session's extraction result with events and relations
- Build a per-session subgraph first (nodes = events, edges = relations)
- Per item (e.g., locomo_item10), merge subgraphs into one graph
- Do not merge across different items. Output one graph file per item.

Node Merging Strategies:
1. Without LLM (default):
   - Merge nodes across sessions using embedding similarity >= threshold
   - Fast but may have false positives
   
2. With LLM (--use-llm-judge):
   - Disable embedding-based auto-merge
   - Use LLM to judge if nodes refer to the same event (merge) or are related (add edge)
   - Higher accuracy, discovers relationships

Outputs:
- <outdir>/<item_id>.gpickle  (pickle-serialized NetworkX MultiDiGraph)
- <outdir>/<item_id>.json     (JSON with nodes/edges for interoperability)
- <outdir>/<item_id>.graphml  (GraphML for tooling/visualization)

Embeddings:
- Default: TF-IDF (no external downloads).
- E5 uses HuggingFace Transformers backend; we format inputs as 'query: <text>'

Usage examples:

  # Without LLM (fast, embedding-based merge)
  python build.py \
    --input data.jsonl \
    --outdir graphs \
    --embed e5 \
    --sbert_model /path/to/e5 \
    --threshold 0.95 \
    --device cuda:0

  # With LLM (accurate, semantic-based merge and relation discovery)
  python build.py \
    --input data.jsonl \
    --outdir graphs \
    --embed e5 \
    --sbert_model /path/to/e5 \
    --use-llm-judge \
    --llm-model /path/to/llm \
    --llm-similarity-threshold 0.80 \
    --device cuda:1
    # Note: --threshold is ignored when --use-llm-judge is enabled
    # Use --device to specify which GPU to load the embedding model (e.g., cuda:0, cuda:1)
"""
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import networkx as nx
import pickle
import numpy as np

# ML deps
try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.neighbors import NearestNeighbors  # type: ignore
    from sklearn.preprocessing import normalize  # type: ignore
except Exception as e:
    raise RuntimeError(
        "scikit-learn is required for TF-IDF embeddings and neighbor search.\n"
        "Please install with: pip install scikit-learn"
    ) from e

# Optional: transformers+torch for E5/SBERT-style pooling
_TRANSFORMERS_AVAILABLE = False
_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel  # type: ignore
    _TRANSFORMERS_AVAILABLE = True
    _TORCH_AVAILABLE = True
except Exception:
    _TRANSFORMERS_AVAILABLE = False
    try:
        import torch  # noqa: F401
        _TORCH_AVAILABLE = True
    except Exception:
        _TORCH_AVAILABLE = False

# Optional: OpenAI client for LLM-based event similarity judgment (supports vLLM backend and external APIs)
_OPENAI_AVAILABLE = False
try:
    from openai import OpenAI  # type: ignore
    _OPENAI_AVAILABLE = True
except Exception:
    OpenAI = None  # type: ignore
    _OPENAI_AVAILABLE = False


@dataclass
class EventNode:
    key: str                 # unique key within item, e.g., "locomo_item10_session1::E1"
    session_id: str
    event_id: str            # original id like "E1"
    summary: str
    people: List[str]
    utterance_refs: List[str]
    time_explicit: str | None
    text: str                # text used for embedding
    embedding: np.ndarray | None = None


@dataclass
class RelationEdge:
    source_key: str  # node key
    target_key: str  # node key
    type: str
    evidence: List[str]


def parse_item_id(session_id: str) -> str:
    """
    Extract item_id from session_id.
    
    Supports two formats:
    1. locomo_item10_session1 -> locomo_item10
    2. doc123_session1 -> doc123
    3. hash_session1 -> hash
    
    Logic: Remove the last part if it matches 'sessionN' pattern
    """
    parts = session_id.split("_")
    
    # If last part looks like "sessionN", remove it
    if len(parts) >= 2 and parts[-1].startswith("session"):
        return "_".join(parts[:-1])
    
    # Fallback: return as-is
    return session_id


def load_sessions_grouped_by_item(jsonl_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            extraction = obj.get("extraction") or {}
            session_id = extraction.get("session_id") or obj.get("session_id")
            if not session_id:
                # skip malformed line
                continue
            item_id = parse_item_id(session_id)
            grouped.setdefault(item_id, []).append(extraction)
    return grouped


def build_session_subgraph(extraction: Dict[str, Any]) -> Tuple[nx.MultiDiGraph, Dict[str, EventNode], List[RelationEdge]]:
    """
    Returns:
      - subgraph with nodes/edges for this session
      - nodes dict keyed by node_key
      - edges list (RelationEdge)
    """
    G = nx.MultiDiGraph()
    session_id = extraction.get("session_id")
    events = extraction.get("events", [])
    relations = extraction.get("relations", [])

    nodes: Dict[str, EventNode] = {}
    edges: List[RelationEdge] = []

    for ev in events:
        ev_id = ev.get("id")
        key = f"{session_id}::{ev_id}"
        summary = ev.get("summary", "")
        people = list(ev.get("people", []) or [])
        utt = list(ev.get("utterance_refs", []) or [])
        time_explicit = None
        t = ev.get("time") or {}
        if isinstance(t, dict):
            time_explicit = t.get("explicit")
        # Build text used for embedding: summary + people + utt refs (lightweight)
        extra = " ".join(people + utt)
        text = summary if not extra else f"{summary} | {extra}"

        node = EventNode(
            key=key,
            session_id=session_id,
            event_id=str(ev_id),
            summary=summary,
            people=people,
            utterance_refs=utt,
            time_explicit=time_explicit,
            text=text,
        )
        nodes[key] = node
        G.add_node(key,
                   session_id=session_id,
                   event_id=str(ev_id),
                   summary=summary,
                   people=people,
                   utterance_refs=utt,
                   time_explicit=time_explicit,
                   text=text,
                   embedding=None)

    for rel in relations:
        src = rel.get("source")
        tgt = rel.get("target")
        rtype = rel.get("type")
        evid = list(rel.get("evidence", []) or [])
        if not src or not tgt:
            continue
        src_key = f"{session_id}::{src}"
        tgt_key = f"{session_id}::{tgt}"
        if src_key not in nodes or tgt_key not in nodes:
            # Missing node, skip
            continue
        edges.append(RelationEdge(source_key=src_key, target_key=tgt_key, type=str(rtype), evidence=evid))
        G.add_edge(src_key, tgt_key, type=str(rtype), evidence=evid)

    return G, nodes, edges


def _torch_device(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    if _TORCH_AVAILABLE and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _average_pool(last_hidden_states, attention_mask):
    # last_hidden_states: (B, L, H); attention_mask: (B, L)
    mask = attention_mask.unsqueeze(-1).to(last_hidden_states.dtype)
    masked = last_hidden_states * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def compute_embeddings(texts: List[str], method: str = "e5", sbert_model_name: str = "", batch_size: int = 64, device: str | None = None) -> np.ndarray:
    method = method.lower()
    if method == "e5":
        if not (_TRANSFORMERS_AVAILABLE and _TORCH_AVAILABLE):
            raise RuntimeError(
                "transformers + torch are required for --embed e5. Install with: pip install transformers torch"
            )
        if not sbert_model_name:
            raise RuntimeError(
                "E5 embedding selected but no model path/name provided.\n"
                "Please pass --sbert_model with an embedding model path (e.g., /share/project/zyt/hyy/Model/bge-m3 or other path)."
            )
        # bge-m3模型不需要添加前缀
        processed_texts = [t.strip() for t in texts]
        tok = AutoTokenizer.from_pretrained(sbert_model_name)
        mdl = AutoModel.from_pretrained(sbert_model_name)
        mdl.eval()
        dev = _torch_device(device)
        mdl.to(dev)
        all_embs: List[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(processed_texts), batch_size):
                batch = processed_texts[i:i + batch_size]
                batch_dict = tok(batch, max_length=512, padding=True, truncation=True, return_tensors='pt')
                batch_dict = {k: v.to(dev) for k, v in batch_dict.items()}
                outputs = mdl(**batch_dict)
                embs = _average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                embs = F.normalize(embs, p=2, dim=1)
                all_embs.append(embs.cpu().numpy().astype(np.float32))
        return np.concatenate(all_embs, axis=0)

    elif method == "sbert":
        # Implement generic mean pooling with transformers backend (no sentence-transformers)
        if not (_TRANSFORMERS_AVAILABLE and _TORCH_AVAILABLE):
            raise RuntimeError(
                "transformers + torch are required for --embed sbert. Install with: pip install transformers torch"
            )
        model_name = sbert_model_name or "sentence-transformers/all-MiniLM-L6-v2"
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModel.from_pretrained(model_name)
        mdl.eval()
        dev = _torch_device(device)
        mdl.to(dev)
        all_embs: List[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_dict = tok(batch, max_length=512, padding=True, truncation=True, return_tensors='pt')
                batch_dict = {k: v.to(dev) for k, v in batch_dict.items()}
                outputs = mdl(**batch_dict)
                embs = _average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                embs = F.normalize(embs, p=2, dim=1)
                all_embs.append(embs.cpu().numpy().astype(np.float32))
        return np.concatenate(all_embs, axis=0)

    else:
        # TF-IDF with L2-normalization; cosine similarity ~ dot product
        vectorizer = TfidfVectorizer(max_features=4096, ngram_range=(1, 2))
        X = vectorizer.fit_transform(texts)
        X = normalize(X)
        return X.toarray().astype(np.float32)


def cluster_nodes_by_threshold(embeddings: np.ndarray, node_keys: List[str], similarity_threshold: float, use_llm: bool = False) -> Tuple[Dict[str, int], List[Tuple[str, str, float]]]:
    """
    Build clusters where any pair with cosine similarity >= threshold are connected;
    clusters are connected components under that relation.

    We do this using NearestNeighbors with metric='cosine' and radius = 1 - threshold.

    Args:
        embeddings: Node embeddings
        node_keys: Node key list
        similarity_threshold: Similarity threshold for merging
        use_llm: If True, skip auto-merging and only collect similar pairs for LLM judgment

    Returns:
        - mapping: node_key -> cluster_id (0..K-1). If use_llm=True, each node is its own cluster
        - top_similar_pairs: list of (key1, key2, similarity) for each node's most similar neighbor
    """
    if len(node_keys) == 0:
        return {}, []
    if len(node_keys) == 1:
        return {node_keys[0]: 0}, []

    n = len(node_keys)
    
    # Calculate full similarity matrix
    similarity_matrix = embeddings @ embeddings.T
    
    # Find top-1 most similar node for each node (for LLM judgment or display)
    top_similar_pairs: List[Tuple[str, str, float]] = []
    seen_pairs = set()
    
    for i in range(n):
        similarities = similarity_matrix[i].copy()
        similarities[i] = -1  # Exclude self
        max_idx = np.argmax(similarities)
        max_sim = similarities[max_idx]
        
        # Only include if similarity is meaningful (> 0.5)
        if max_sim > 0.5:
            # Avoid duplicate pairs (i, j) and (j, i)
            pair_key = tuple(sorted([node_keys[i], node_keys[max_idx]]))
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                top_similar_pairs.append((node_keys[i], node_keys[max_idx], float(max_sim)))
    
    # If using LLM, don't auto-merge - each node stays independent
    if use_llm:
        print(f"\n=== Using LLM for merge decisions ===")
        print(f"Skipping auto-merge based on embedding similarity")
        print(f"Found {len(top_similar_pairs)} similar pairs for LLM judgment")
        
        # Each node is its own cluster
        cluster_map = {key: i for i, key in enumerate(node_keys)}
        return cluster_map, top_similar_pairs
    
    # Original auto-merge logic based on threshold
    # cosine distance = 1 - cosine_similarity
    radius = max(0.0, 1.0 - float(similarity_threshold))
    nbrs = NearestNeighbors(metric="cosine", radius=radius, algorithm="brute")
    nbrs.fit(embeddings)
    # radius_neighbors returns distances and indices
    distances, indices = nbrs.radius_neighbors(embeddings, radius=radius, return_distance=True)

    # Build adjacency list (exclude self if distance == 0 is included)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Debug: collect similarity statistics
    similarity_stats = []
    merge_pairs = []
    
    for i in range(n):
        for dist, j in zip(distances[i], indices[i]):
            if j == i:
                continue
            # cosine_distance <= radius -> similarity >= threshold
            similarity = 1.0 - dist  # convert cosine distance to cosine similarity
            similarity_stats.append(similarity)
            merge_pairs.append((node_keys[i], node_keys[j], similarity))
            union(i, int(j))
    
    # Print similarity statistics
    if similarity_stats:
        print(f"\n=== Similarity Statistics (threshold={similarity_threshold}) ===")
        print(f"Total pairs considered for merging: {len(similarity_stats)}")
        print(f"Similarity range: {min(similarity_stats):.4f} - {max(similarity_stats):.4f}")
        print(f"Mean similarity: {sum(similarity_stats)/len(similarity_stats):.4f}")
        
        # Show top 10 highest similarities
        merge_pairs.sort(key=lambda x: x[2], reverse=True)
        print(f"\nTop 10 highest similarities:")
        for i, (key1, key2, sim) in enumerate(merge_pairs[:10]):
            print(f"  {i+1:2d}. {sim:.4f} - {key1} <-> {key2}")
        
        # Show distribution
        bins = [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 1.0]
        print(f"\nSimilarity distribution:")
        for i in range(len(bins)-1):
            count = sum(1 for s in similarity_stats if bins[i] <= s < bins[i+1])
            if count > 0:
                print(f"  {bins[i]:.3f}-{bins[i+1]:.3f}: {count:3d} pairs")
        print("=" * 60)

    # Compress and assign cluster ids
    root_to_cluster: Dict[int, int] = {}
    cluster_map: Dict[str, int] = {}
    next_cid = 0
    for i, key in enumerate(node_keys):
        r = find(i)
        if r not in root_to_cluster:
            root_to_cluster[r] = next_cid
            next_cid += 1
        cluster_map[key] = root_to_cluster[r]

    return cluster_map, top_similar_pairs


# LLM-based event similarity judgment prompt
EVENT_SIMILARITY_PROMPT = (
    "You are an expert at analyzing events and determining if they refer to the same real-world occurrence or have significant overlap.\n\n"
    "Given two event descriptions extracted from different dialog sessions, determine:\n"
    "1. Whether they describe the SAME event (same occurrence at the same time)\n"
    "2. Whether they have SIGNIFICANT OVERLAP (mention or relate to the same real-world situation/topic)\n\n"
    "Consider these factors:\n"
    "- Do they involve the same people/participants?\n"
    "- Do they describe the same actions, situations, or topics?\n"
    "- Do they have compatible time references?\n"
    "- Would merging their information create a more complete picture of ONE event?\n\n"
    "Output a JSON object with these exact keys:\n"
    "{\n"
    "  \"same_event\": boolean,           // true if they are the same event\n"
    "  \"has_overlap\": boolean,          // true if they refer to the same situation/topic\n"
    "  \"relation_type\": string | null,  // if has_overlap, suggest relation type\n"
    "  \"reasoning\": string              // brief explanation\n"
    "}\n"
)


def build_similarity_judgment_prompt(node1: EventNode, node2: EventNode) -> str:
    """Build prompt for LLM to judge if two events are the same or overlap."""
    return (
        f"{EVENT_SIMILARITY_PROMPT}\n"
        f"Event 1:\n"
        f"  Session: {node1.session_id}\n"
        f"  Event ID: {node1.event_id}\n"
        f"  Summary: {node1.summary}\n"
        f"  People: {', '.join(node1.people) if node1.people else 'None'}\n"
        f"  Time: {node1.time_explicit if node1.time_explicit else 'Not specified'}\n\n"
        f"Event 2:\n"
        f"  Session: {node2.session_id}\n"
        f"  Event ID: {node2.event_id}\n"
        f"  Summary: {node2.summary}\n"
        f"  People: {', '.join(node2.people) if node2.people else 'None'}\n"
        f"  Time: {node2.time_explicit if node2.time_explicit else 'Not specified'}\n\n"
        f"Output JSON:"
    )


def extract_json_block(text: str) -> Any:
    """Extract the first complete JSON object from text."""
    # Remove any <think>...</think> reasoning blocks
    try:
        text = re.sub(r'<think[\s\S]*?>[\s\S]*?</think>', '', text, flags=re.IGNORECASE)
    except Exception:
        text = re.sub(r'<think>[\s\S]*?</think>', '', text)
    
    # Try to extract JSON from markdown code blocks first
    markdown_pattern = r'```(?:json)?\s*(\{[\s\S]*?\})\s*```'
    markdown_matches = re.findall(markdown_pattern, text)
    if markdown_matches:
        # Try each markdown block
        for json_candidate in markdown_matches:
            try:
                return json.loads(json_candidate)
            except json.JSONDecodeError:
                continue
    
    # If no valid JSON in markdown blocks, remove them and search in plain text
    text = re.sub(r'```[\s\S]*?```', '', text)
    
    # Find the first complete JSON object
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model output")
    
    # Count braces to find the matching closing brace
    brace_count = 0
    end = start
    for i in range(start, len(text)):
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                end = i
                break
    
    if brace_count != 0:
        raise ValueError("Incomplete JSON object found in model output")
    
    json_str = text[start : end + 1]
    return json.loads(json_str)


def judge_similarity_with_llm(
    similar_pairs: List[Tuple[str, str, float]],
    all_nodes: Dict[str, EventNode],
    client: Any,
    model_name: str,
    max_tokens: int,
    temperature: float,
    llm_similarity_threshold: float,
    batch_size: int = 8,
    max_retries: int = 1
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str, str]]]:
    """
    Use LLM API to judge if similar node pairs should be merged or connected with edges.
    
    Args:
        similar_pairs: List of (key1, key2, similarity_score)
        all_nodes: Dict mapping node keys to EventNode objects
        client: OpenAI client instance
        model_name: Model name to use for API calls
        max_tokens: Maximum tokens for LLM response
        temperature: Temperature for LLM sampling
        llm_similarity_threshold: Only judge pairs above this similarity
        batch_size: Number of prompts to process in one batch (sequential API calls)
        max_retries: Maximum number of retries for JSON parse errors
    
    Returns:
        Tuple of:
        - List of (key1, key2) pairs to merge (same_event=True)
        - List of (source_key, target_key, relation_type) for edges to add (has_overlap=True but not same_event)
    """
    if not similar_pairs:
        return [], []
    
    # Filter pairs by LLM similarity threshold
    filtered_pairs = [(k1, k2, sim) for k1, k2, sim in similar_pairs if sim >= llm_similarity_threshold]
    
    if not filtered_pairs:
        print(f"\nNo pairs above LLM similarity threshold {llm_similarity_threshold}")
        return [], []
    
    print(f"\n=== LLM-based Event Similarity Judgment ===")
    print(f"Judging {len(filtered_pairs)} pairs (similarity >= {llm_similarity_threshold})")
    
    pairs_to_merge: List[Tuple[str, str]] = []
    edges_to_add: List[Tuple[str, str, str]] = []
    
    # Track pairs that failed JSON parsing for retry
    failed_pairs: Dict[Tuple[str, str], Tuple[str, str, float]] = {}  # (key1, key2) -> (key1, key2, sim)
    
    # Process pairs (API calls are made sequentially or in small batches)
    for batch_idx in range(0, len(filtered_pairs), batch_size):
        batch_pairs = filtered_pairs[batch_idx : batch_idx + batch_size]
        
        print(f"\nProcessing batch {batch_idx // batch_size + 1}/{(len(filtered_pairs) + batch_size - 1) // batch_size}")
        
        # Process each pair in the batch
        for key1, key2, sim in batch_pairs:
            node1 = all_nodes[key1]
            node2 = all_nodes[key2]
            prompt = build_similarity_judgment_prompt(node1, node2)
            
            # Make API call
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are an expert at analyzing event relationships."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    n=1
                )
                
                # Check if response is valid
                if not hasattr(response, 'choices'):
                    print(f"  WARNING: Invalid response format for pair ({key1}, {key2})")
                    print(f"  Response type: {type(response)}")
                    failed_pairs[(key1, key2)] = (key1, key2, sim)
                    continue
                
                if not response.choices:
                    print(f"  WARNING: Empty output for pair ({key1}, {key2})")
                    failed_pairs[(key1, key2)] = (key1, key2, sim)
                    continue
                
                text = response.choices[0].message.content
                if not text:
                    print(f"  WARNING: Empty content for pair ({key1}, {key2})")
                    failed_pairs[(key1, key2)] = (key1, key2, sim)
                    continue
            except Exception as e:
                print(f"  ERROR calling API for pair ({key1}, {key2}): {e}")
                import traceback
                print(f"  Traceback: {traceback.format_exc()[:500]}")
                failed_pairs[(key1, key2)] = (key1, key2, sim)
                continue
            
            try:
                judgment = extract_json_block(text)
                has_overlap = judgment.get("has_overlap", False)
                same_event = judgment.get("same_event", False)
                relation_type = judgment.get("relation_type")
                reasoning = judgment.get("reasoning", "")
                
                if same_event:
                    # Merge these nodes
                    pairs_to_merge.append((key1, key2))
                    print(f"  ⚡ MERGE: {key1} <==> {key2} (sim={sim:.3f})")
                    print(f"    Reasoning: {reasoning[:100]}...")
                elif has_overlap:
                    # Add edge but don't merge
                    rel_type = relation_type if relation_type else "related_to"
                    edges_to_add.append((key1, key2, rel_type))
                    print(f"  ✓ Adding edge: {key1} --[{rel_type}]--> {key2} (sim={sim:.3f})")
                    print(f"    Reasoning: {reasoning[:100]}...")
                else:
                    print(f"  ✗ No relation: {key1} <-> {key2} (sim={sim:.3f})")
                    
            except Exception as e:
                print(f"  ERROR parsing judgment for pair ({key1}, {key2}): {e}")
                print(f"  Raw output: {text[:200]}...")
                failed_pairs[(key1, key2)] = (key1, key2, sim)
    
    # Retry failed pairs
    if failed_pairs and max_retries > 0:
        print(f"\n=== Retrying {len(failed_pairs)} failed pair judgments ===")
        for retry_attempt in range(max_retries):
            print(f"Retry attempt {retry_attempt + 1}/{max_retries}")
            
            retry_pairs_list = list(failed_pairs.values())
            
            # Process retries
            for batch_idx in range(0, len(retry_pairs_list), batch_size):
                batch_pairs = retry_pairs_list[batch_idx : batch_idx + batch_size]
                
                print(f"Processing retry batch {batch_idx // batch_size + 1}/{(len(retry_pairs_list) + batch_size - 1) // batch_size}")
                
                # Process each pair in the batch
                for key1, key2, sim in batch_pairs:
                    node1 = all_nodes[key1]
                    node2 = all_nodes[key2]
                    prompt = build_similarity_judgment_prompt(node1, node2)
                    
                    # Make API call
                    try:
                        response = client.chat.completions.create(
                            model=model_name,
                            messages=[
                                {"role": "system", "content": "You are an expert at analyzing event relationships."},
                                {"role": "user", "content": prompt}
                            ],
                            max_tokens=max_tokens,
                            temperature=temperature,
                            n=1
                        )
                        
                        # Check if response is valid
                        if not hasattr(response, 'choices'):
                            print(f"  WARNING: Invalid response format on retry for pair ({key1}, {key2})")
                            continue
                        
                        if not response.choices:
                            print(f"  WARNING: Still empty output for pair ({key1}, {key2})")
                            continue
                        
                        text = response.choices[0].message.content
                        if not text:
                            print(f"  WARNING: Empty content on retry for pair ({key1}, {key2})")
                            continue
                    except Exception as e:
                        print(f"  ERROR calling API for pair ({key1}, {key2}) on retry: {e}")
                        continue
                    
                    try:
                        judgment = extract_json_block(text)
                        has_overlap = judgment.get("has_overlap", False)
                        same_event = judgment.get("same_event", False)
                        relation_type = judgment.get("relation_type")
                        reasoning = judgment.get("reasoning", "")
                        
                        if same_event:
                            pairs_to_merge.append((key1, key2))
                            print(f"  ⚡ MERGE (retry): {key1} <==> {key2} (sim={sim:.3f})")
                            print(f"    Reasoning: {reasoning[:100]}...")
                        elif has_overlap:
                            rel_type = relation_type if relation_type else "related_to"
                            edges_to_add.append((key1, key2, rel_type))
                            print(f"  ✓ Adding edge (retry): {key1} --[{rel_type}]--> {key2} (sim={sim:.3f})")
                            print(f"    Reasoning: {reasoning[:100]}...")
                        else:
                            print(f"  ✗ No relation (retry): {key1} <-> {key2} (sim={sim:.3f})")
                        
                        # Remove from failed pairs if successful
                        if (key1, key2) in failed_pairs:
                            del failed_pairs[(key1, key2)]
                            
                    except Exception as e:
                        print(f"  ERROR parsing judgment for pair ({key1}, {key2}) on retry: {e}")
                        print(f"  Raw output: {text[:200]}...")
            
            # If no more failures, break early
            if not failed_pairs:
                print(f"All failed pairs successfully retried!")
                break
    
    print(f"\nLLM judgment complete:")
    print(f"  - {len(pairs_to_merge)} pairs to merge")
    print(f"  - {len(edges_to_add)} edges to add")
    if failed_pairs:
        print(f"  - {len(failed_pairs)} pairs still failed after {max_retries} retries")
    return pairs_to_merge, edges_to_add


def merge_item_graph(
    session_graphs: List[Tuple[nx.MultiDiGraph, Dict[str, EventNode], List[RelationEdge]]],
    embed_method: str,
    threshold: float,
    sbert_model_name: str = "",
    item_id: str | None = None,
    client: Any = None,
    model_name: str = "",
    max_tokens: int = 1024,
    temperature: float = 0.2,
    llm_similarity_threshold: float = 0.85,
    llm_batch_size: int = 8,
    max_retries: int = 1,
) -> nx.MultiDiGraph:
    # Collect all nodes and edges across sessions
    all_nodes: Dict[str, EventNode] = {}
    all_edges: List[RelationEdge] = []
    for G, nodes, edges in session_graphs:
        all_nodes.update(nodes)
        all_edges.extend(edges)

    node_keys = list(all_nodes.keys())
    texts = [all_nodes[k].text for k in node_keys]

    # Compute embeddings for all nodes and store back to nodes
    emb = compute_embeddings(texts, method=embed_method, sbert_model_name=sbert_model_name)
    for i, k in enumerate(node_keys):
        all_nodes[k].embedding = emb[i]

    # Cluster by similarity threshold and get top similar pairs
    use_llm_for_merge = (client is not None and model_name)
    cluster_map, top_similar_pairs = cluster_nodes_by_threshold(emb, node_keys, similarity_threshold=threshold, use_llm=use_llm_for_merge)
    
    # Use LLM to judge similarity and decide merges/edges if configured
    llm_edges: List[Tuple[str, str, str]] = []
    if use_llm_for_merge:
        pairs_to_merge, llm_edges = judge_similarity_with_llm(
            top_similar_pairs,
            all_nodes,
            client,
            model_name,
            max_tokens,
            temperature,
            llm_similarity_threshold,
            llm_batch_size,
            max_retries
        )
        
        # Apply LLM merge decisions to cluster_map using union-find
        if pairs_to_merge:
            print(f"\nApplying {len(pairs_to_merge)} LLM-decided merges...")
            
            # Build union-find structure
            node_to_idx = {key: i for i, key in enumerate(node_keys)}
            n = len(node_keys)
            parent = list(range(n))
            
            def find(x: int) -> int:
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            
            def union(a: int, b: int):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra
            
            # Apply merges
            for key1, key2 in pairs_to_merge:
                idx1 = node_to_idx[key1]
                idx2 = node_to_idx[key2]
                union(idx1, idx2)
            
            # Rebuild cluster_map
            root_to_cluster: Dict[int, int] = {}
            next_cid = 0
            for i, key in enumerate(node_keys):
                r = find(i)
                if r not in root_to_cluster:
                    root_to_cluster[r] = next_cid
                    next_cid += 1
                cluster_map[key] = root_to_cluster[r]
            
            print(f"After LLM merges: {len(set(cluster_map.values()))} clusters from {len(node_keys)} nodes")

    # Prepare merged graph
    MG = nx.MultiDiGraph(item_id=item_id or "", embed_method=embed_method, threshold=float(threshold),
                         created_at=datetime.utcnow().isoformat() + "Z")

    # Build clusters -> representative node attributes
    clusters: Dict[int, Dict[str, Any]] = {}
    cluster_members: Dict[int, List[str]] = {}
    for key, cid in cluster_map.items():
        node = all_nodes[key]
        cluster_members.setdefault(cid, []).append(key)
        info = clusters.setdefault(cid, {
            "node_id": f"N{cid}",
            "session_ids": [],
            "event_ids": [],
            "summaries": [],
            "people": set(),
            "utterance_refs": [],
            "time_explicit": [],
            "texts": [],
            "embeddings": [],
        })
        info["session_ids"].append(node.session_id)
        info["event_ids"].append(node.event_id)
        info["summaries"].append(node.summary)
        info["people"].update(node.people)
        info["utterance_refs"].extend(node.utterance_refs)
        if node.time_explicit:
            info["time_explicit"].append(node.time_explicit)
        info["texts"].append(node.text)
        if node.embedding is not None:
            info["embeddings"].append(node.embedding)

    # Add merged nodes with centroid embedding
    cid_to_nodekey: Dict[int, str] = {}
    for cid, info in clusters.items():
        emb_stack = np.vstack(info["embeddings"]) if info["embeddings"] else None
        centroid = None
        if emb_stack is not None and emb_stack.size > 0:
            centroid = emb_stack.mean(axis=0)
            # normalize centroid to unit length for cosine similarity semantics
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
        node_key = info["node_id"]
        cid_to_nodekey[cid] = node_key
        MG.add_node(
            node_key,
            session_ids=sorted(set(info["session_ids"])),
            event_ids=info["event_ids"],
            summaries=info["summaries"],
            people=sorted(info["people"]),
            utterance_refs=sorted(set(info["utterance_refs"])),
            time_explicit=sorted(set(info["time_explicit"])),
            texts=info["texts"],
            embedding=(centroid.tolist() if centroid is not None else None),
        )

    # Remap edges to merged nodes and add to MG
    for e in all_edges:
        src_c = cluster_map[e.source_key]
        tgt_c = cluster_map[e.target_key]
        src_m = cid_to_nodekey[src_c]
        tgt_m = cid_to_nodekey[tgt_c]
        MG.add_edge(src_m, tgt_m, type=e.type, evidence=e.evidence)
    
    # Add LLM-judged similarity edges (between original nodes before merging)
    for src_key, tgt_key, rel_type in llm_edges:
        src_c = cluster_map[src_key]
        tgt_c = cluster_map[tgt_key]
        src_m = cid_to_nodekey[src_c]
        tgt_m = cid_to_nodekey[tgt_c]
        # Only add edge if nodes were not merged together (different clusters)
        if src_m != tgt_m:
            MG.add_edge(src_m, tgt_m, type=rel_type, evidence=[f"LLM-judged similarity between {src_key} and {tgt_key}"])
            print(f"Added LLM edge: {src_m} --[{rel_type}]--> {tgt_m}")

    return MG


def _graphml_safe_value(v: Any) -> Any:
    # GraphML supports scalar types only. Convert others to JSON strings.
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return v
    try:
        return json.dumps(v, ensure_ascii=False)
    except Exception:
        return str(v)


def to_graphml_compatible(G: nx.Graph) -> nx.Graph:
    # Create a same-kind graph with GraphML-safe attribute values
    if isinstance(G, nx.MultiDiGraph):
        H = nx.MultiDiGraph()
    elif isinstance(G, nx.MultiGraph):
        H = nx.MultiGraph()
    elif G.is_directed():
        H = nx.DiGraph()
    else:
        H = nx.Graph()

    # Graph-level attributes
    H.graph.update({k: _graphml_safe_value(v) for k, v in G.graph.items()})

    # Nodes
    for n, data in G.nodes(data=True):
        H.add_node(n, **{k: _graphml_safe_value(v) for k, v in data.items()})

    # Edges
    if G.is_multigraph():
        for u, v, k, data in G.edges(keys=True, data=True):
            H.add_edge(u, v, key=k, **{k2: _graphml_safe_value(v2) for k2, v2 in data.items()})
    else:
        for u, v, data in G.edges(data=True):
            H.add_edge(u, v, **{k2: _graphml_safe_value(v2) for k2, v2 in data.items()})

    return H


def graph_to_json_dict(G: nx.MultiDiGraph) -> Dict[str, Any]:
    nodes = []
    for nid, data in G.nodes(data=True):
        # Ensure JSON serializable (embedding already list or None)
        node = {"id": nid}
        node.update(data)
        nodes.append(node)

    edges = []
    for u, v, k, data in G.edges(keys=True, data=True):
        edge = {"source": u, "target": v}
        edge.update(data)
        edges.append(edge)

    meta = {k: v for k, v in G.graph.items()}
    return {"meta": meta, "nodes": nodes, "edges": edges}


def build_per_item_graphs(
    jsonl_path: Path,
    outdir: Path,
    embed_method: str,
    threshold: float,
    sbert_model_name: str = "",
) -> Dict[str, nx.MultiDiGraph]:
    outdir.mkdir(parents=True, exist_ok=True)

    grouped = load_sessions_grouped_by_item(jsonl_path)

    item_graphs: Dict[str, nx.MultiDiGraph] = {}

    for item_id, sessions in grouped.items():
        # Build subgraphs per session
        per_session: List[Tuple[nx.MultiDiGraph, Dict[str, EventNode], List[RelationEdge]]] = []
        for extraction in sessions:
            Gs, nodes, edges = build_session_subgraph(extraction)
            per_session.append((Gs, nodes, edges))

        # Merge into per-item graph
        MG = merge_item_graph(
            per_session,
            embed_method=embed_method,
            threshold=threshold,
            sbert_model_name=sbert_model_name,
            item_id=item_id,
        )
        item_graphs[item_id] = MG

        # Write outputs
        gpickle_path = outdir / f"{item_id}.gpickle"
        json_path = outdir / f"{item_id}.json"
        with gpickle_path.open("wb") as f:
            pickle.dump(MG, f, protocol=pickle.HIGHEST_PROTOCOL)
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(graph_to_json_dict(MG), f, ensure_ascii=False, indent=2)
        # Also write GraphML (attributes coerced to GraphML-safe scalars via JSON strings)
        graphml_path = outdir / f"{item_id}.graphml"
        H = to_graphml_compatible(MG)
        nx.write_graphml(H, graphml_path)

        print(f"Wrote {item_id}: nodes={MG.number_of_nodes()} edges={MG.number_of_edges()} -> {gpickle_path} , {json_path} , {graphml_path}")

    return item_graphs


def main():
    parser = argparse.ArgumentParser(description="Build per-item event graphs from two-stage JSONL")
    parser.add_argument("--input", required=True, help="Path to JSONL file (two-stage extraction)")
    parser.add_argument("--outdir", required=True, help="Directory to write output graphs")
    parser.add_argument("--embed", default="e5", choices=["tfidf", "e5"], help="Embedding method")
    parser.add_argument("--sbert_model", default="", help="Model name/path for E5 (HF repo or local dir). Leave blank to fill later")
    parser.add_argument("--threshold", type=float, default=0.88, help="Cosine similarity threshold for merging (higher -> stricter)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for transformer embedding")
    parser.add_argument("--device", type=str, default=None, help="Device for transformer embedding: cuda|cuda:0|cuda:1|cpu, etc. Auto-detect if omitted")
    
    # LLM-based similarity judgment arguments (using API)
    parser.add_argument("--use-llm-judge", action="store_true", help="Enable LLM-based judgment for similar node pairs")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000/v1", help="API base URL (vLLM backend or external API)")
    parser.add_argument("--api-key", type=str, default="", help="API key (if required)")
    parser.add_argument("--llm-model", type=str, default="", help="Model name for API calls (required if --use-llm-judge)")
    parser.add_argument("--llm-similarity-threshold", type=float, default=0.85, help="Similarity threshold for LLM judgment (default: 0.85)")
    parser.add_argument("--llm-batch-size", type=int, default=8, help="Batch size for LLM inference (default: 8)")
    parser.add_argument("--llm-max-tokens", type=int, default=1024, help="Max tokens for LLM response (default: 1024)")
    parser.add_argument("--llm-temperature", type=float, default=0.2, help="Temperature for LLM sampling (default: 0.2)")
    parser.add_argument("--max-retries", type=int, default=1, help="Maximum number of retries for JSON parse errors (default: 1)")

    args = parser.parse_args()

    jsonl_path = Path(args.input)
    outdir = Path(args.outdir)
    embed_method = args.embed
    sbert_model_name = args.sbert_model
    threshold = float(args.threshold)

    if not jsonl_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {jsonl_path}")

    if not (0.0 <= threshold <= 1.0):
        raise ValueError("--threshold must be in [0, 1]")

    device_info = args.device if args.device else "auto"
    print(f"Building graphs from {jsonl_path} -> {outdir}\n"
          f"  embed={embed_method} threshold={threshold} model={sbert_model_name}\n"
          f"  device={device_info}")
    
    # Initialize API client if requested
    client = None
    
    if args.use_llm_judge:
        if not args.llm_model:
            raise ValueError("--llm-model is required when --use-llm-judge is enabled")
        
        if not _OPENAI_AVAILABLE:
            raise RuntimeError(
                "openai is required for --use-llm-judge. Please install with: pip install openai"
            )
        
        print(f"\n=== Initializing LLM API client for similarity judgment ===")
        print(f"API URL: {args.api_url}")
        print(f"Model: {args.llm_model}")
        print(f"Similarity threshold for LLM: {args.llm_similarity_threshold}")
        print(f"Batch size: {args.llm_batch_size}")
        
        try:
            client = OpenAI(
                base_url=args.api_url,
                api_key=args.api_key if args.api_key else "EMPTY"
            )
            print("API client initialized successfully")
        except Exception as e:
            print(f"Error initializing API client: {e}")
            raise
        
        print("=" * 60 + "\n")

    # Monkey-patch compute_embeddings partial for batch/device if transformers path used
    # We pass via global default parameters by re-binding a lambda if needed
    def _compute(texts):
        return compute_embeddings(texts, method=embed_method, sbert_model_name=sbert_model_name, batch_size=args.batch_size, device=args.device)

    # Build using the wrapper
    grouped = load_sessions_grouped_by_item(jsonl_path)
    outdir.mkdir(parents=True, exist_ok=True)

    item_graphs: Dict[str, nx.MultiDiGraph] = {}
    for item_id, sessions in grouped.items():
        per_session: List[Tuple[nx.MultiDiGraph, Dict[str, EventNode], List[RelationEdge]]] = []
        for extraction in sessions:
            Gs, nodes, edges = build_session_subgraph(extraction)
            per_session.append((Gs, nodes, edges))

        # Compute embeddings once for this item
        # Collect texts and fill later
        all_nodes: Dict[str, EventNode] = {}
        all_edges: List[RelationEdge] = []
        for Gs, nodes, edges in per_session:
            all_nodes.update(nodes)
            all_edges.extend(edges)
        node_keys = list(all_nodes.keys())
        texts = [all_nodes[k].text for k in node_keys]
        emb = _compute(texts)
        for i, k in enumerate(node_keys):
            all_nodes[k].embedding = emb[i]
        # Now call merge using precomputed embeddings: we temporarily inject into function by bypassing recomputation
        # To reuse existing merge logic, we pass per_session but immediately overwrite embeddings inside merge
        # Simpler: call cluster_nodes_by_threshold here, then build merged graph
        use_llm_for_merge = (client is not None and args.llm_model)
        cluster_map, top_similar_pairs = cluster_nodes_by_threshold(emb, node_keys, similarity_threshold=threshold, use_llm=use_llm_for_merge)
        
        # Use LLM to judge similarity and decide merges/edges if configured
        llm_edges: List[Tuple[str, str, str]] = []
        if use_llm_for_merge:
            pairs_to_merge, llm_edges = judge_similarity_with_llm(
                top_similar_pairs,
                all_nodes,
                client,
                args.llm_model,
                args.llm_max_tokens,
                args.llm_temperature,
                args.llm_similarity_threshold,
                args.llm_batch_size,
                args.max_retries
            )
            
            # Apply LLM merge decisions to cluster_map
            if pairs_to_merge:
                print(f"\nApplying {len(pairs_to_merge)} LLM-decided merges...")
                
                # Build union-find structure
                node_to_idx = {key: i for i, key in enumerate(node_keys)}
                n = len(node_keys)
                parent = list(range(n))
                
                def find(x: int) -> int:
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x
                
                def union(a: int, b: int):
                    ra, rb = find(a), find(b)
                    if ra != rb:
                        parent[rb] = ra
                
                # Apply merges
                for key1, key2 in pairs_to_merge:
                    idx1 = node_to_idx[key1]
                    idx2 = node_to_idx[key2]
                    union(idx1, idx2)
                
                # Rebuild cluster_map
                root_to_cluster: Dict[int, int] = {}
                next_cid = 0
                for i, key in enumerate(node_keys):
                    r = find(i)
                    if r not in root_to_cluster:
                        root_to_cluster[r] = next_cid
                        next_cid += 1
                    cluster_map[key] = root_to_cluster[r]
                
                print(f"After LLM merges: {len(set(cluster_map.values()))} clusters from {len(node_keys)} nodes")

        MG = nx.MultiDiGraph(item_id=item_id or "", embed_method=embed_method, threshold=float(threshold),
                             created_at=datetime.utcnow().isoformat() + "Z")
        # Aggregate clusters
        clusters: Dict[int, Dict[str, Any]] = {}
        for key, cid in cluster_map.items():
            node = all_nodes[key]
            info = clusters.setdefault(cid, {
                "node_id": f"N{cid}",
                "session_ids": [],
                "event_ids": [],
                "summaries": [],
                "people": set(),
                "utterance_refs": [],
                "time_explicit": [],
                "texts": [],
                "embeddings": [],
            })
            info["session_ids"].append(node.session_id)
            info["event_ids"].append(node.event_id)
            info["summaries"].append(node.summary)
            info["people"].update(node.people)
            info["utterance_refs"].extend(node.utterance_refs)
            if node.time_explicit:
                info["time_explicit"].append(node.time_explicit)
            info["texts"].append(node.text)
            if node.embedding is not None:
                info["embeddings"].append(node.embedding)
        # Add merged nodes with centroid
        cid_to_nodekey: Dict[int, str] = {}
        for cid, info in clusters.items():
            emb_stack = np.vstack(info["embeddings"]) if info["embeddings"] else None
            centroid = None
            if emb_stack is not None and emb_stack.size > 0:
                centroid = emb_stack.mean(axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    centroid = centroid / norm
            node_key = info["node_id"]
            cid_to_nodekey[cid] = node_key
            MG.add_node(
                node_key,
                session_ids=sorted(set(info["session_ids"])),
                event_ids=info["event_ids"],
                summaries=info["summaries"],
                people=sorted(info["people"]),
                utterance_refs=sorted(set(info["utterance_refs"])),
                time_explicit=sorted(set(info["time_explicit"])),
                texts=info["texts"],
                embedding=(centroid.tolist() if centroid is not None else None),
            )
        # Remap edges from per-session to merged
        for Gs, nodes, edges in per_session:
            for e in edges:
                src_m = cid_to_nodekey[cluster_map[e.source_key]]
                tgt_m = cid_to_nodekey[cluster_map[e.target_key]]
                MG.add_edge(src_m, tgt_m, type=e.type, evidence=e.evidence)
        
        # Add LLM-judged similarity edges
        for src_key, tgt_key, rel_type in llm_edges:
            src_c = cluster_map[src_key]
            tgt_c = cluster_map[tgt_key]
            src_m = cid_to_nodekey[src_c]
            tgt_m = cid_to_nodekey[tgt_c]
            # Only add edge if nodes were not merged together (different clusters)
            if src_m != tgt_m:
                MG.add_edge(src_m, tgt_m, type=rel_type, evidence=[f"LLM-judged similarity between {src_key} and {tgt_key}"])
                print(f"Added LLM edge: {src_m} --[{rel_type}]--> {tgt_m}")

        # Save
        item_graphs[item_id] = MG
        gpickle_path = outdir / f"{item_id}.gpickle"
        json_path = outdir / f"{item_id}.json"
        with gpickle_path.open("wb") as f:
            pickle.dump(MG, f, protocol=pickle.HIGHEST_PROTOCOL)
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(graph_to_json_dict(MG), f, ensure_ascii=False, indent=2)
        print(f"Wrote {item_id}: nodes={MG.number_of_nodes()} edges={MG.number_of_edges()} -> {gpickle_path} , {json_path}")

    # Return purely for programmatic use
    return item_graphs


if __name__ == "__main__":
    main()
