#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-stage event-centric information extraction from dialog sessions using API.

Stage 1: Extract events from dialog
Stage 2: Extract relations between the extracted events

- Supports two input formats:
  1) jsonl: one utterance per line with metadata {sample_id, session_num, speaker, text, dia_id, date_time, has_image}
  2) locomo10.json style: array of items, each with a "conversation" containing session_1, session_2, ...
- Stage 1: For each session, sends one prompt to extract events (E1..En) in order
- Stage 2: For each session with events, sends another prompt to extract pairwise relations
- Writes structured JSONL, one line per session

Usage examples:

# 1) jsonl input (utterance-per-line)
python Memory/build_graph/extract.py \
  --input /home/u2021201791/workspace/Memory/data/locomo/dialog/dialog.jsonl \
  --output /home/u2021201791/workspace/Memory/data/locomo/dialog/event_extractions.jsonl \
  --api-url http://localhost:8000/v1 \
  --model Meta-Llama-3.1-8B-Instruct \
  --max-new-tokens 2048 --temperature 0.2

# 2) locomo10.json input (array with "conversation")
python extract.py \
  --input /share/project/zyt/hyy/Memory/data/locomo/locomo10.json \
  --output /share/project/zyt/hyy/Memory/build_graph/locomo10_events.jsonl \
  --api-url http://localhost:8000/v1 \
  --model Qwen3-8B \
  --max-new-tokens 32768 --temperature 0.2 \
  --max-sessions 1

Requirements:
- openai (pip install openai)
- API server (vLLM backend or external API)
"""

import argparse
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Optional

try:
    from openai import OpenAI
except Exception as e:  # noqa: E722
    OpenAI = None  # type: ignore
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


# Stage 1: Event extraction prompt
EVENT_PROMPT_INSTRUCTION = (
    "You are an expert information extraction system. Given a multi-turn dialog, extract meaningful events and output ONE strict JSON object.\n"
    "Goals:\n"
    "- Extract logically coherent events (E1, E2, ...) in chronological order. Each event should represent a complete logical unit or process.\n"
    "- AGGRESSIVELY COMBINE related micro-events into comprehensive event summaries to avoid fragmentation. Merge events that:\n"
    "  * Involve the same participants discussing the same topic (combine Q&A exchanges into one conversation event)\n"
    "  * Form a logical sequence or workflow (combine decision + action + completion into one comprehensive event)\n"
    "  * Are temporally close and thematically related (within 3-5 utterances)\n"
    "  * Represent different aspects of the same situation or problem\n"
    "  * Include follow-up questions, clarifications, or elaborations on the same topic\n"
    "- PRESERVE ALL important details within each merged event summary. Include comprehensive information:\n"
    "  * Complete context of what was discussed, decided, or accomplished\n"
    "  * All key outcomes, results, conclusions, and decisions made\n"
    "  * Specific facts, numbers, dates, locations, and concrete details\n"
    "  * Emotional states, reactions, and interpersonal dynamics\n"
    "  * Technical details, requirements, and specifications when mentioned\n"
    "  * Any conditions, constraints, or limitations discussed\n"
    "  * IMPORTANT: Include visual content descriptions when images are shared\n"
    "- For each event, ONLY list people involved as an array `people` (max 3). Do not output other entity types or attributes/states.\n"
    "- Extract 6-10 comprehensive events that capture the complete narrative flow without redundancy. Prioritize fewer, more detailed events over many fragmented ones.\n"
    "- Output JSON only, no additional commentary.\n"
)

EVENT_SCHEMA_BLOCK = (
    "Return JSON with exactly these keys:\n"
    "{\n"
    "  \"session_id\": string,\n"
    "  \"time\": string | null,\n"
    "  \"events\": [\n"
    "    {\n"
    "      \"id\": string,              // E1, E2, ... in order\n"
    "      \"summary\": string,         // comprehensive description including key details and outcomes\n"
    "      \"utterance_refs\": [string],// list of utterance ids (e.g., D1:3) that support the event\n"
    "      \"time\": {\n"
    "        \"explicit\": string | null,  // direct quote time if present\n"
    "        \"relative_order\": integer   // event order index (1-based)\n"
    "      },\n"
    "      \"people\": [string]        // ONLY the people involved in this event (max 3)\n"
    "    }\n"
    "  ]\n"
    "}\n"
    "\n"
    "IMPORTANT GUIDELINES FOR EVENT SUMMARIES:\n"
    "- Make summaries comprehensive but focused - include what happened, why it matters, and key outcomes\n"
    "- Use specific details rather than vague descriptions (e.g., 'discussed budget allocation of $50,000' not 'discussed money')\n"
    "- Include emotional context when relevant (e.g., 'expressed concern about', 'agreed enthusiastically')\n"
    "- Capture decisions, agreements, disagreements, and their implications\n"
    "- Preserve important facts, numbers, dates, and specific information mentioned\n"
    "- CRITICAL: When images are shared, include the visual content description in the event summary (e.g., 'Melanie shared a painting of a sunset over a lake')\n"
)

# Stage 2: Relation extraction prompt
RELATION_PROMPT_INSTRUCTION = (
    "You are an expert information extraction system. Given a list of extracted events from a dialog, identify meaningful pairwise relations between them and output ONE strict JSON object.\n"
    "Goals:\n"
    "- Consider ALL unordered pairs of events within the same session (not only adjacent events).\n"
    "- Extract pairwise event relations with a SHORT, free-form label in `type` that best characterizes the link.\n"
    "- Relation types can include: causal, motivation, enablement, follow_up, temporal_before, temporal_after, contrast, part_of, parallel, elaboration. These are examples, not a closed set.\n"
    "- Add relations only when meaningful. Prefer specific semantic links (causal/motivation/enablement/part_of) over trivial temporal ordering.\n"
    "- It is acceptable to have no temporal edges if they add no insight.\n"
    "- IMPORTANT: For temporal relations (follow_up, temporal_before, temporal_after), base them on the ACTUAL TIME when events occurred in the real world, NOT on when they are described or mentioned in the dialog. Focus on the chronological sequence of what actually happened.\n"
    "- For each relation, cite minimal `evidence` utterance ids that support the linkage between the two events.\n"
    "- Output JSON only, no additional commentary.\n"
)

RELATION_SCHEMA_BLOCK = (
    "Return JSON with exactly these keys:\n"
    "{\n"
    "  \"session_id\": string,\n"
    "  \"relations\": [\n"
    "    {\n"
    "      \"source\": string,           // event id, e.g., E1\n"
    "      \"target\": string,           // event id, e.g., E2\n"
    "      \"type\": string,             // e.g.: causal, motivation, enablement, follow_up, temporal_before, temporal_after, contrast, part_of, parallel, elaboration. These are examples, not a closed set\n"
    "      \"evidence\": [string]        // utterance ids that support the relation\n"
    "    }\n"
    "  ]\n"
    "}\n"
)

# Helpers for debug saving

def _sanitize_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    return safe[:255]


def _write_debug(debug_dir: Optional[str], session_id: str, stage: str, text: str) -> None:
    if not debug_dir:
        return
    try:
        os.makedirs(debug_dir, exist_ok=True)
        fname = f"{_sanitize_filename(session_id)}.raw.txt"
        path = os.path.join(debug_dir, fname)
        mode = "w" if stage == "stage1" else "a"
        with open(path, mode, encoding="utf-8") as f:
            f.write(f"===== {session_id} | {stage.upper()} RAW OUTPUT =====\n")
            f.write(text if isinstance(text, str) else str(text))
            f.write("\n\n")
    except Exception as _e:
        print(f"[DEBUG SAVE ERROR] {session_id} {stage}: {_e}")


def _write_result_to_file(output_file: str, session_id: str, result: Dict[str, Any]) -> None:
    """Incrementally write a single result to the output file."""
    try:
        # Format output according to original schema
        if result["status"] == "ok":
            extraction = {
                "session_id": result["session_id"],
                "time": result.get("time"),
                "events": result.get("events", []),
                "relations": result.get("relations", [])
            }
            payload = {
                "session_id": session_id,
                "status": "ok",
                "extraction": extraction,
            }
        else:
            payload = {
                "session_id": session_id,
                "status": result["status"],
                "extraction": None,
                "error": result.get("error"),
                "raw": result.get("raw"),
            }
        
        # Append to file
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        
        print(f"  💾 Saved result for {session_id}")
    except Exception as e:
        print(f"[SAVE ERROR] Failed to write {session_id}: {e}")


def build_session_text(turns: List[Dict[str, Any]]) -> str:
    """Format dialog turns into a compact block for prompting."""
    lines = []
    for t in turns:
        meta = t.get("metadata", {})
        ts = meta.get("date_time") or ""
        spk = meta.get("speaker") or "Unknown"
        dia_id = meta.get("dia_id") or t.get("id")
        text = meta.get("text") or t.get("contents") or ""
        has_img = meta.get("has_image", False)
        blip_caption = meta.get("blip_caption", "")
        
        # Include image description if available
        if has_img and blip_caption:
            suffix = f" [shared image: {blip_caption}]"
        elif has_img:
            suffix = " [shared image]"
        else:
            suffix = ""
            
        lines.append(f"- [{dia_id}] [{ts}] {spk}: {text}{suffix}")
    return "\n".join(lines)


def build_event_extraction_prompt(session_id: str, session_text: str) -> str:
    """Build prompt for Stage 1: Event extraction."""
    return (
        f"{EVENT_PROMPT_INSTRUCTION}\n"
        f"{EVENT_SCHEMA_BLOCK}\n"
        f"Input:\n"
        f"session_id: {session_id}\n"
        f"dialog:\n{session_text}\n\n"
        f"Output JSON:"
    )


def build_relation_extraction_prompt(session_id: str, session_text: str, events: List[Dict[str, Any]]) -> str:
    """Build prompt for Stage 2: Relation extraction."""
    # Format events for the prompt
    events_text = "Extracted Events:\n"
    for event in events:
        events_text += f"- {event['id']}: {event['summary']}\n"
        if event.get('utterance_refs'):
            events_text += f"  References: {', '.join(event['utterance_refs'])}\n"
    
    return (
        f"{RELATION_PROMPT_INSTRUCTION}\n"
        f"{RELATION_SCHEMA_BLOCK}\n"
        f"Input:\n"
        f"session_id: {session_id}\n"
        f"Original dialog:\n{session_text}\n\n"
        f"{events_text}\n"
        f"Output JSON:"
    )


def read_sessions(input_path: str) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """Read sessions from either jsonl (utterance-per-line) or locomo10.json (array with conversation).

    Returns: list of (session_id, turns), where each turn is a dict with metadata fields
    compatible with build_session_text(): {metadata: {speaker, text, dia_id, date_time, has_image}}
    """
    with open(input_path, "r", encoding="utf-8") as f:
        head = f.read(2048)
        f.seek(0)
        stripped = head.lstrip()
        is_json_container = stripped.startswith("[") or stripped.startswith("{")
        
        if not is_json_container:
            # Assume jsonl format
            sessions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                meta = obj.get("metadata", {})
                sample_id = meta.get("sample_id", "unknown")
                session_num = meta.get("session_num", "unknown")
                key = f"{sample_id}_session{session_num}"
                sessions[key].append(obj)
            return list(sessions.items())
        
        # Parse as JSON container (e.g., locomo10.json)
        data = json.load(f)
        # If it's a dict with a top-level "conversation"
        if isinstance(data, dict) and "conversation" in data:
            data = [data]
        
        if isinstance(data, list) and data and isinstance(data[0], dict) and "conversation" in data[0]:
            sessions: List[Tuple[str, List[Dict[str, Any]]]] = []
            for idx, item in enumerate(data):
                conv = item.get("conversation", {})
                # Collect session keys like session_1, session_2, ...
                session_keys = []
                for k in conv.keys():
                    m = re.match(r"^session_(\d+)$", k)
                    if m:
                        session_keys.append((int(m.group(1)), k))
                session_keys.sort()
                # speakers (optional)
                speaker_a = conv.get("speaker_a")
                speaker_b = conv.get("speaker_b")
                for snum, skey in session_keys:
                    turns_raw = conv.get(skey, [])
                    # Find date_time for this session
                    dt = conv.get(f"session_{snum}_date_time", None)
                    turns: List[Dict[str, Any]] = []
                    for t in turns_raw:
                        speaker = t.get("speaker") or "Unknown"
                        dia_id = t.get("dia_id")
                        text = t.get("text") or ""
                        has_img = bool(t.get("img_url"))
                        blip_caption = t.get("blip_caption", "")
                        # Build a uniform turn object
                        turns.append({
                            "metadata": {
                                "speaker": speaker,
                                "text": text,
                                "dia_id": dia_id,
                                "date_time": dt,
                                "has_image": has_img,
                                "blip_caption": blip_caption,
                            }
                        })
                    session_id = f"locomo_item{idx+1}_session{snum}"
                    sessions.append((session_id, turns))
            return sessions
        else:
            # Fallback: try to interpret as jsonl-like array
            sessions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for obj in data if isinstance(data, list) else []:
                meta = obj.get("metadata", {})
                sample_id = meta.get("sample_id", "unknown")
                session_num = meta.get("session_num", "unknown")
                key = f"{sample_id}_session{session_num}"
                sessions[key].append(obj)
            return list(sessions.items())


def extract_json_block(text: str) -> Any:
    """Try to parse the first complete JSON object from text."""
    # Remove any <think>...</think> reasoning blocks then strip markdown code blocks and extra text
    try:
        text = re.sub(r'<think[\s\S]*?>[\s\S]*?</think>', '', text, flags=re.IGNORECASE)
    except Exception:
        # Fallback simple pattern if flags not supported in environment
        text = re.sub(r'<think>[\s\S]*?</think>', '', text)
    
    # Remove known noise patterns
    # text = re.sub(r'自然语言处理', '', text)
    
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


# def extract_json_block(text: str) -> Any:
#     """Extract the last complete JSON object from text."""
#     # Remove markdown code blocks
#     text = re.sub(r'```[^`]*```', '', text)
#     text = re.sub(r'自然语言处理', '', text)
    
#     # Find all opening braces, start from the last one
#     brace_positions = [i for i, char in enumerate(text) if char == '{']
#     if not brace_positions:
#         raise ValueError("No JSON object found in model output")
    
#     # Try extracting from last to first position
#     for start in reversed(brace_positions):
#         brace_count = 0
#         for i in range(start, len(text)):
#             if text[i] == '{':
#                 brace_count += 1
#             elif text[i] == '}':
#                 brace_count -= 1
#                 if brace_count == 0:
#                     try:
#                         return json.loads(text[start:i + 1])
#                     except json.JSONDecodeError:
#                         break  # Try next position
    
#     raise ValueError("No valid JSON object found in model output")

def process_stage1_events(client: Any, model_name: str, max_tokens: int, temperature: float, top_p: float,
                         sessions: List[Tuple[str, List[Dict[str, Any]]]], 
                         batch_size: int, output_file: str, debug_dir: Optional[str] = None, max_retries: int = 1) -> Dict[str, Dict[str, Any]]:
    """Stage 1: Extract events from all sessions with retry on JSON parse error. 
    Writes results incrementally to output file."""
    print("=== Stage 1: Extracting Events ===")
    
    # Prepare prompts for event extraction
    event_prompts: List[str] = []
    session_ids: List[str] = []
    session_texts: Dict[str, str] = {}
    
    for sid, turns in sessions:
        session_text = build_session_text(turns)
        session_texts[sid] = session_text
        prompt = build_event_extraction_prompt(sid, session_text)
        event_prompts.append(prompt)
        session_ids.append(sid)
    
    # Extract events (API calls)
    event_results: Dict[str, Dict[str, Any]] = {}
    
    for i in range(0, len(event_prompts), batch_size):
        batch_prompts = event_prompts[i : i + batch_size]
        batch_ids = session_ids[i : i + batch_size]
        
        print(f"Processing event extraction batch {i//batch_size + 1}/{(len(event_prompts) + batch_size - 1)//batch_size}")
        
        # Process each session in the batch
        for sid, prompt in zip(batch_ids, batch_prompts):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are an expert information extraction system."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    n=1
                )
                
                # Check if response is valid
                if not hasattr(response, 'choices') or not response.choices:
                    print(f"  WARNING: Invalid response format for {sid}")
                    event_results[sid] = {
                        "session_id": sid,
                        "status": "empty_output",
                        "events": [],
                        "session_text": session_texts[sid]
                    }
                    continue
                
                text = response.choices[0].message.content
                if not text:
                    print(f"  WARNING: Empty content for {sid}")
                    event_results[sid] = {
                        "session_id": sid,
                        "status": "empty_output",
                        "events": [],
                        "session_text": session_texts[sid]
                    }
                    continue
            except Exception as e:
                print(f"Error calling API for session {sid}: {e}")
                event_results[sid] = {
                    "session_id": sid,
                    "status": "api_error",
                    "error": str(e),
                    "events": [],
                    "session_text": session_texts[sid]
                }
                continue

        # Process each session in the batch
        for sid, prompt in zip(batch_ids, batch_prompts):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are an expert information extraction system."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    n=1
                )
                
                # Check if response is valid
                if not hasattr(response, 'choices') or not response.choices:
                    print(f"  WARNING: Invalid response format for {sid}")
                    print(f"  Response type: {type(response)}, Response: {str(response)[:200]}")
                    event_results[sid] = {
                        "session_id": sid,
                        "status": "empty_output",
                        "events": [],
                        "session_text": session_texts[sid]
                    }
                    continue
                
                text = response.choices[0].message.content
                if not text:
                    print(f"  WARNING: Empty content for {sid}")
                    event_results[sid] = {
                        "session_id": sid,
                        "status": "empty_output",
                        "events": [],
                        "session_text": session_texts[sid]
                    }
                    continue
            except Exception as e:
                print(f"Error calling API for session {sid}: {e}")
                import traceback
                print(f"  Traceback: {traceback.format_exc()}")
                event_results[sid] = {
                    "session_id": sid,
                    "status": "api_error",
                    "error": str(e),
                    "events": [],
                    "session_text": session_texts[sid]
                }
                continue
            _write_debug(debug_dir, sid, "stage1", text)
            print(f"Event extraction result for {sid}")
            
            try:
                extraction = extract_json_block(text)
                # Ensure session_id matches and extract events
                if isinstance(extraction, dict):
                    extraction.setdefault("session_id", sid)
                    events = extraction.get("events", [])
                    event_results[sid] = {
                        "session_id": sid,
                        "status": "ok",
                        "events": events,
                        "session_text": session_texts[sid],
                        "time": extraction.get("time")
                    }
                    print(f"  ✓ Extracted {len(events)} events for {sid}")
                else:
                    event_results[sid] = {
                        "session_id": sid,
                        "status": "invalid_format",
                        "events": [],
                        "session_text": session_texts[sid]
                    }
            except Exception as e:
                print(f"Error parsing events for {sid}: {e}")
                event_results[sid] = {
                    "session_id": sid,
                    "status": "json_parse_error",
                    "error": str(e),
                    "events": [],
                    "session_text": session_texts[sid]
                }
    
    # Retry for sessions with JSON parse errors
    failed_sessions = [(sid, session_texts[sid]) for sid, result in event_results.items() 
                      if result["status"] == "json_parse_error"]
    
    if failed_sessions and max_retries > 0:
        print(f"\n=== Retrying {len(failed_sessions)} failed sessions ===")
        for retry_attempt in range(max_retries):
            print(f"Retry attempt {retry_attempt + 1}/{max_retries}")
            
            retry_prompts: List[str] = []
            retry_ids: List[str] = []
            
            for sid, session_text in failed_sessions:
                prompt = build_event_extraction_prompt(sid, session_text)
                retry_prompts.append(prompt)
                retry_ids.append(sid)
            
            # Process retries
            for i in range(0, len(retry_prompts), batch_size):
                batch_prompts = retry_prompts[i : i + batch_size]
                batch_ids = retry_ids[i : i + batch_size]
                
                print(f"Processing retry batch {i//batch_size + 1}/{(len(retry_prompts) + batch_size - 1)//batch_size}")
                
                for sid, prompt in zip(batch_ids, batch_prompts):
                    try:
                        response = client.chat.completions.create(
                            model=model_name,
                            messages=[
                                {"role": "system", "content": "You are an expert information extraction system."},
                                {"role": "user", "content": prompt}
                            ],
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            n=1
                        )
                        
                        if not response.choices:
                            continue
                        
                        text = response.choices[0].message.content
                    except Exception as e:
                        print(f"Error calling API for session {sid} on retry: {e}")
                        continue
                    
                    _write_debug(debug_dir, sid, f"stage1_retry{retry_attempt+1}", text)
                    print(f"Retry result for {sid}")
                    
                    try:
                        extraction = extract_json_block(text)
                        if isinstance(extraction, dict):
                            extraction.setdefault("session_id", sid)
                            events = extraction.get("events", [])
                            event_results[sid] = {
                                "session_id": sid,
                                "status": "ok",
                                "events": events,
                                "session_text": session_texts[sid],
                                "time": extraction.get("time")
                            }
                            print(f"Successfully extracted events for {sid} on retry")
                        else:
                            print(f"Still invalid format for {sid} on retry")
                    except Exception as e:
                        print(f"Still failed to parse events for {sid} on retry: {e}")
            
            # Update failed_sessions for next retry
            failed_sessions = [(sid, session_texts[sid]) for sid, result in event_results.items() 
                             if result["status"] == "json_parse_error"]
            if not failed_sessions:
                break
    
    return event_results


def process_stage2_relations(client: Any, model_name: str, max_tokens: int, temperature: float, top_p: float,
                            event_results: Dict[str, Dict[str, Any]], 
                            batch_size: int, output_file: str, debug_dir: Optional[str] = None, max_retries: int = 1) -> Dict[str, Dict[str, Any]]:
    """Stage 2: Extract relations based on events with retry on JSON parse error.
    Writes results incrementally to output file."""
    print("=== Stage 2: Extracting Relations ===")
    
    # Prepare prompts for relation extraction (only for sessions with events)
    relation_prompts: List[str] = []
    session_ids: List[str] = []
    
    for sid, event_data in event_results.items():
        if event_data["status"] == "ok" and event_data.get("events"):
            prompt = build_relation_extraction_prompt(
                sid, 
                event_data["session_text"], 
                event_data["events"]
            )
            relation_prompts.append(prompt)
            session_ids.append(sid)
        else:
            # No events to relate, set empty relations
            event_results[sid]["relations"] = []
    
    # Track which sessions failed to parse relations
    relation_parse_errors: Dict[str, bool] = {}
    
    # Extract relations (API calls)
    for i in range(0, len(relation_prompts), batch_size):
        batch_prompts = relation_prompts[i : i + batch_size]
        batch_ids = session_ids[i : i + batch_size]
        
        print(f"Processing relation extraction batch {i//batch_size + 1}/{(len(relation_prompts) + batch_size - 1)//batch_size}")

        # Process each session in the batch
        for sid, prompt in zip(batch_ids, batch_prompts):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are an expert information extraction system."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    n=1
                )
                
                if not response.choices:
                    print(f"WARNING: Empty output for {sid} in relation extraction")
                    event_results[sid]["relations"] = []
                    continue
                
                text = response.choices[0].message.content
            except Exception as e:
                print(f"Error calling API for session {sid} in relation extraction: {e}")
                event_results[sid]["relations"] = []
                relation_parse_errors[sid] = True
                continue
            
            # Parse the API response
            _write_debug(debug_dir, sid, "stage2", text)
            
            try:
                extraction = extract_json_block(text)
                if isinstance(extraction, dict):
                    relations = extraction.get("relations", [])
                    event_results[sid]["relations"] = relations
                    print(f"  ✓ Extracted {len(relations)} relations for {sid}")
                    
                    # Write incrementally after successful extraction
                    _write_result_to_file(output_file, sid, event_results[sid])
                else:
                    print(f"WARNING: Non-dict extraction for {sid}")
                    event_results[sid]["relations"] = []
                    relation_parse_errors[sid] = True
            except Exception as e:
                print(f"Error parsing relations for {sid}: {e}")
                event_results[sid]["relations"] = []
                relation_parse_errors[sid] = True
    
    # Retry for sessions with relation parse errors
    failed_sessions = [sid for sid in relation_parse_errors.keys()]
    
    if failed_sessions and max_retries > 0:
        print(f"\n=== Retrying {len(failed_sessions)} failed relation extractions ===")
        for retry_attempt in range(max_retries):
            print(f"Retry attempt {retry_attempt + 1}/{max_retries}")
            
            retry_prompts: List[str] = []
            retry_ids: List[str] = []
            
            for sid in failed_sessions:
                event_data = event_results[sid]
                prompt = build_relation_extraction_prompt(
                    sid,
                    event_data["session_text"],
                    event_data["events"]
                )
                retry_prompts.append(prompt)
                retry_ids.append(sid)
            
            # Process retries
            for i in range(0, len(retry_prompts), batch_size):
                batch_prompts = retry_prompts[i : i + batch_size]
                batch_ids = retry_ids[i : i + batch_size]
                
                print(f"Processing retry batch {i//batch_size + 1}/{(len(retry_prompts) + batch_size - 1)//batch_size}")
                
                for sid, prompt in zip(batch_ids, batch_prompts):
                    try:
                        response = client.chat.completions.create(
                            model=model_name,
                            messages=[
                                {"role": "system", "content": "You are an expert information extraction system."},
                                {"role": "user", "content": prompt}
                            ],
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            n=1
                        )
                        
                        if not response.choices:
                            continue
                        
                        text = response.choices[0].message.content
                    except Exception as e:
                        print(f"Error calling API for session {sid} on retry: {e}")
                        continue
                    _write_debug(debug_dir, sid, f"stage2_retry{retry_attempt+1}", text)
                    print(f"Retry result for {sid}")
                    
                    try:
                        extraction = extract_json_block(text)
                        if isinstance(extraction, dict):
                            relations = extraction.get("relations", [])
                            event_results[sid]["relations"] = relations
                            print(f"  ✓ Extracted {len(relations)} relations for {sid} on retry")
                            
                            # Write incrementally after successful retry
                            _write_result_to_file(output_file, sid, event_results[sid])
                            
                            # Remove from failed list
                            if sid in relation_parse_errors:
                                del relation_parse_errors[sid]
                        else:
                            print(f"Still invalid format for {sid} on retry")
                    except Exception as e:
                        print(f"Still failed to parse relations for {sid} on retry: {e}")
            
            # Update failed_sessions for next retry
            failed_sessions = list(relation_parse_errors.keys())
            if not failed_sessions:
                break
        
    return event_results


def main():
    parser = argparse.ArgumentParser(description="Two-stage event-centric extraction using API")
    parser.add_argument("--input", required=True, help="Path to input dialog.jsonl")
    parser.add_argument("--output", required=True, help="Path to write extractions.jsonl")
    parser.add_argument("--api-url", required=True, help="API base URL (vLLM backend or external API)")
    parser.add_argument("--api-key", type=str, default="", help="API key (if required)")
    parser.add_argument("--model", required=True, help="Model name for API calls")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-sessions", type=int, default=None, help="Limit number of sessions for a quick run")
    parser.add_argument("--batch-size", type=int, default=4, help="Number of prompts to process")
    parser.add_argument("--debug-dir", type=str, default=None, help="Directory to save raw LLM outputs per session (not in JSONL)")
    parser.add_argument("--max-retries", type=int, default=1, help="Maximum number of retries for JSON parse errors (default: 1)")
    args = parser.parse_args()

    if _IMPORT_ERROR is not None:
        raise RuntimeError(f"Failed to import openai. Please install openai. Original error: {_IMPORT_ERROR}")

    sessions = read_sessions(args.input)
    # stable order by session_id for reproducibility
    sessions.sort(key=lambda x: x[0])
    if args.max_sessions is not None:
        sessions = sessions[: args.max_sessions]

    print(f"Initializing API client")
    print(f"API URL: {args.api_url}")
    print(f"Model: {args.model}")
    print(f"Max tokens: {args.max_new_tokens}")
    print(f"Temperature: {args.temperature}")

    try:
        # Initialize OpenAI client
        api_key = args.api_key if args.api_key else None
        client = OpenAI(
            base_url=args.api_url,
            api_key=api_key
        )
        print("API client initialized successfully")
    except Exception as e:
        print(f"Error initializing API client: {e}")
        raise

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Clear output file if it exists
    if os.path.exists(args.output):
        print(f"Clearing existing output file: {args.output}")
        with open(args.output, "w", encoding="utf-8") as f:
            pass
    
    # Stage 1: Extract events with retry
    print(f"\n💾 Results will be saved incrementally to: {args.output}")
    event_results = process_stage1_events(client, args.model, args.max_new_tokens, args.temperature, args.top_p,
                                         sessions, args.batch_size, args.output,
                                         debug_dir=args.debug_dir, max_retries=args.max_retries)
    
    # Stage 2: Extract relations with retry (incrementally writes to file)
    final_results = process_stage2_relations(client, args.model, args.max_new_tokens, args.temperature, args.top_p,
                                            event_results, args.batch_size, args.output,
                                            debug_dir=args.debug_dir, max_retries=args.max_retries)
    
    # Write any remaining results that don't have relations yet (failed sessions)
    print(f"\n💾 Writing any remaining incomplete results...")
    for sid, result in final_results.items():
        if "relations" not in result:
            result["relations"] = []
            _write_result_to_file(args.output, sid, result)

    print(f"Wrote extractions to {args.output}")
    print(f"Processed {len(sessions)} sessions with two-stage extraction")


if __name__ == "__main__":
    main()
