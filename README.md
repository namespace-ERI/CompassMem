# CompassMem


- `build/` - Graph building and clustering modules
  - `build.py` - Build per-item event graphs from extraction results
  - `cluster_and_create_hierarchy.py` - Perform K-means clustering and create hierarchical nodes
  - `extract.py` - Two-stage event extraction from dialog sessions

- `QA_system/` - Question answering system modules  
  - `hierarchical_main_qa_system_v3.py` - Main QA system with two-stage retrieval
  - `hierarchical_embedding_manager_v3.py` - Embedding computation and similarity management
  - `hierarchical_graph_loader.py` - Hierarchical graph data loader
  - `multi_path_explorer_v2_queue.py` - Multi-path exploration with global queue
  - `llm_handler_v2_queue.py` - LLM interaction handler (OpenAI API)
  - `context_formatter_v2.py` - Context information formatter
  - `conversation_manager.py` - Dialog data manager
  - `cluster_explorer.py` - Cluster node exploration
  - `debug_logger.py` - LLM interaction logging
  - `evaluate.py` - Evaluation script for QA results

