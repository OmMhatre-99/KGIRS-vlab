"""
Virtual Laboratory: Knowledge Graphs and Information Retrieval Systems (KGIRS)
Experiment 7: Relationship Extraction from Unstructured Text
Department of Computer Engineering / Information Technology - Virtual Laboratory Portal

Extracts semantic relationships (Subject - Predicate - Object triples) between
entities identified in text, producing entity-relationship triples suitable for
graph construction (feeds directly into Experiment 8: Create and Manage a Graph
Database, and Experiment 9: Knowledge Graph Schema & Data Import).

Features:
- Dual Light/Dark mode contrast-calibrated interactive knowledge graph visualization.
- Interactive inline pop-up disclosure notes on technical terms for student guidance.
- 6 comprehensive analytical tabs: Triples, Annotated Text, Entities, Graph, Flow (Sankey), and Evaluation.
- Ground truth reference sets for empirical Precision, Recall, and F1 evaluation.
- Institutional formal academic UI with trial logger, conceptual quiz, and PDF report generator.
"""

from __future__ import annotations

import os
import re
import io
import csv
import json
import math
import html
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from fpdf import FPDF


# ======================================================================================
# 1. THEME ENGINE: CONTRAST-CALIBRATED PALETTES FOR LIGHT & DARK MODES
# ======================================================================================

class Theme:
    """Provides high-contrast, WCAG-compliant styling tokens for Light and Dark modes."""
    def __init__(self, dark: bool = False):
        self.dark = dark
        if dark:
            self.ink = "#F8FAFC"             # Crisp off-white text
            self.muted = "#94A3B8"           # Medium slate for subtitles / cues
            self.edge = "#94A3B8"            # Clean slate line
            self.grid = "#334155"            # Subtle grid line
            self.halo = "rgba(15, 23, 42, 0.95)"  # Dark slate boundary for markers
            self.paper_bg = "rgba(0,0,0,0)"
            self.plot_bg = "rgba(0,0,0,0)"
            self.card_bg = "rgba(30, 41, 59, 0.65)"
            self.card_border = "rgba(148, 163, 184, 0.3)"
            self.card_accent = "#38BDF8"
            self.badge_bg = "rgba(56, 189, 248, 0.18)"
            self.badge_text = "#38BDF8"
            self.node_colors = {
                "PERSON": "#38BDF8",     # Bright sky blue
                "ORG": "#FB923C",        # Bright coral orange
                "LOCATION": "#34D399",   # Vibrant emerald
                "DATE": "#C084FC",       # Vibrant violet
                "TERM": "#F87171",       # Salmon rose
                "MISC": "#A3A3A3",       # Neutral silver
            }
        else:
            self.ink = "#0F172A"             # Deep slate / navy text
            self.muted = "#475569"           # Slate for subtitles / cues
            self.edge = "#64748B"            # Rich slate edge
            self.grid = "#E2E8F0"            # Light grey grid line
            self.halo = "rgba(255, 255, 255, 0.95)" # White crisp boundary
            self.paper_bg = "rgba(0,0,0,0)"
            self.plot_bg = "rgba(0,0,0,0)"
            self.card_bg = "rgba(248, 250, 252, 0.85)"
            self.card_border = "rgba(100, 116, 139, 0.25)"
            self.card_accent = "#2563EB"
            self.badge_bg = "rgba(37, 99, 235, 0.12)"
            self.badge_text = "#1D4ED8"
            self.node_colors = {
                "PERSON": "#2563EB",     # Classic royal blue
                "ORG": "#D97706",        # Amber orange
                "LOCATION": "#059669",   # Deep emerald
                "DATE": "#7C3AED",       # Deep violet
                "TERM": "#DC2626",       # Deep crimson
                "MISC": "#4B5563",       # Charcoal
            }

    def for_type(self, entity_type: str) -> str:
        return self.node_colors.get(entity_type, self.node_colors["MISC"])


def get_current_theme() -> Theme:
    """Detects active theme or reads user manual toggle from session state."""
    choice = st.session_state.get("theme_mode_choice", "Auto")
    if choice == "Dark":
        return Theme(dark=True)
    elif choice == "Light":
        return Theme(dark=False)

    is_dark = False
    try:
        ctx_theme = getattr(st.context, "theme", None)
        if ctx_theme is not None and getattr(ctx_theme, "type", None):
            is_dark = (ctx_theme.type == "dark")
        else:
            base = str(st.get_option("theme.base")).lower()
            is_dark = (base == "dark")
    except Exception:
        is_dark = False
    return Theme(dark=is_dark)


# Fixed marker shapes per entity type for double encoding (color + shape)
TYPE_SYMBOL = {
    "PERSON": "circle",
    "ORG": "square",
    "LOCATION": "diamond",
    "DATE": "triangle-up",
    "TERM": "hexagon",
    "MISC": "cross",
}

TYPE_SLOT = {
    "PERSON": 0,
    "ORG": 1,
    "LOCATION": 2,
    "DATE": 3,
    "TERM": 4,
    "MISC": 5,
}

# Streamlit Markdown text-highlight colors for annotated text view
TYPE_HIGHLIGHT = {
    "PERSON": "blue",
    "ORG": "orange",
    "LOCATION": "green",
    "DATE": "violet",
    "TERM": "red",
    "MISC": "gray",
}

ENTITY_TYPE_ORDER = ["PERSON", "ORG", "LOCATION", "DATE", "TERM", "MISC"]


# ======================================================================================
# 2. TECHNICAL TERMINOLOGY & INTERACTIVE POP-UP DISCLOSURE SYSTEM
# ======================================================================================

_TERM_STYLE = """
<style>
details.kgterm {
    display: inline-block;
    position: relative;
}
details.kgterm > summary {
    display: inline;
    list-style: none;
    cursor: pointer;
    border-bottom: 1.5px dotted currentColor;
    font-weight: 500;
    color: inherit;
    transition: all 0.2s ease-in-out;
    padding: 0 2px;
    border-radius: 2px;
}
details.kgterm > summary::-webkit-details-marker {
    display: none;
}
details.kgterm > summary::marker {
    content: "";
}
details.kgterm > summary:hover {
    background-color: rgba(59, 130, 246, 0.12);
    border-bottom: 1.5px solid currentColor;
}
details.kgterm[open] > summary {
    font-weight: 600;
    background-color: rgba(59, 130, 246, 0.18);
    border-bottom: 2px solid currentColor;
}
/* Positioned absolutely (a floating popover) rather than left in normal text flow, so
   opening a note overlays the page instead of pushing every following line down/sideways -
   which is what caused text after a closed note to land on a stray broken line. */
div.kgnote {
    display: block;
    position: absolute;
    top: 100%;
    left: 0;
    z-index: 40;
    width: max-content;
    max-width: min(26rem, 88vw);
    margin: 0.35rem 0 0 0;
    padding: 0.85rem 1.15rem;
    border: 1px solid rgba(128, 128, 128, 0.28);
    border-left: 4px solid #3b82f6;
    border-radius: 6px;
    font-size: 0.92rem;
    line-height: 1.55;
    font-weight: 400;
    white-space: normal;
    background-color: rgba(255, 255, 255, 0.97);
    backdrop-filter: blur(4px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.16);
}
div.kgnote .kgnote-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.35rem;
}
div.kgnote .kgnote-term {
    font-weight: 700;
    font-size: 0.96rem;
    letter-spacing: -0.01em;
}
div.kgnote .kgnote-badge {
    font-size: 0.70rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 2px 7px;
    border-radius: 4px;
    background-color: rgba(59, 130, 246, 0.15);
    color: #3b82f6;
    font-weight: 600;
}
div.kgnote .kgnote-def {
    margin-bottom: 0.45rem;
}
div.kgnote .kgnote-example {
    display: block;
    margin-top: 0.45rem;
    padding: 0.5rem 0.75rem;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.84rem;
    background: rgba(0, 0, 0, 0.05);
    border-radius: 4px;
    border-left: 3px solid rgba(128, 128, 128, 0.35);
    white-space: pre-wrap;
}
</style>
"""

TERMS: Dict[str, Dict[str, str]] = {
    "information_retrieval": {
        "name": "Information Retrieval (IR)",
        "definition": "The science and engineering discipline of searching for, indexing, extracting, and representing relevant facts or knowledge from large collections of unstructured data.",
        "example": "Turning raw narrative news reports into queryable knowledge graph assertions.",
    },
    "relationship_extraction": {
        "name": "Relationship Extraction (RE)",
        "definition": "The NLP and IR task of detecting semantic relationships holding between two identified entities mentioned in text and mapping them into structured assertions.",
        "example": '"Larry Page co-founded Google in California" -> (Larry Page, founded, Google)',
    },
    "knowledge_graph": {
        "name": "Knowledge Graph (KG)",
        "definition": "A network structured representation of real-world knowledge where entities form nodes and semantic relationships form directed, typed edges. Enables multi-hop relational queries.",
        "example": "(Larry Page)-[:FOUNDED]->(Google)-[:LOCATED_IN]->(California)",
    },
    "triple": {
        "name": "Subject-Predicate-Object (SPO) Triple",
        "definition": "The atomic data unit in Knowledge Graphs and RDF consisting of exactly three components: (Subject Entity, Predicate / Relationship, Object Entity).",
        "example": "(Google, acquired, YouTube)\nSubject = Google, Predicate = acquired, Object = YouTube",
    },
    "subject": {
        "name": "Subject",
        "definition": "The source entity initiating or possessing the semantic relationship. In active voice English, the subject typically precedes the verbal predicate.",
        "example": 'In "Marie Curie discovered radium", the subject is Marie Curie (PERSON).',
    },
    "predicate": {
        "name": "Predicate / Relation",
        "definition": "The semantic link connecting subject and object. Normalised to snake_case without spaces to form clean edge types in graph databases like Neo4j.",
        "example": 'The verbs "co-founded", "founded", and "established" all map to the canonical predicate "founded".',
    },
    "object": {
        "name": "Object",
        "definition": "The target or destination entity that receives or completes the relation assertion.",
        "example": 'In "Microsoft is located in Redmond", the object is Redmond (LOCATION).',
    },
    "named_entity_recognition": {
        "name": "Named Entity Recognition (NER)",
        "definition": "The preceding IR stage that locates mentions of real-world objects in unstructured text and classifies them into predefined types (Experiment 6).",
        "example": "Recognizing [Sundar Pichai] as PERSON, [Google] as ORG, and [California] as LOCATION.",
    },
    "entity": {
        "name": "Entity",
        "definition": "A specific real-world object referred to in the document (person, organisation, place, or domain concept). Distinct from generic common nouns.",
        "example": '"Google" is an entity; "a software company" is a generic noun phrase.',
    },
    "entity_type": {
        "name": "Entity Type",
        "definition": "The category assigned to an entity (PERSON, ORG, LOCATION, DATE, TERM, MISC). Restricts which relational predicates can meaningfully link them.",
        "example": 'born_in requires subject: PERSON and object: LOCATION or DATE.',
    },
    "relation_cue": {
        "name": "Relation Cue / Trigger Lexicon",
        "definition": "The specific lexical token or verbal phrase between two entities that provides textual evidence for the relationship.",
        "example": 'In "Google acquired YouTube in 2006", the word "acquired" is the lexical cue.',
    },
    "pattern_based": {
        "name": "Pattern-Based Extraction",
        "definition": "Extracts triples only when an explicit relation cue matches a predefined lexical rule or regex. Achieves high precision, but lower recall due to natural language paraphrasing.",
        "example": 'Matches "X founded Y" and "X launched Y", but may miss "X was the architect behind Y".',
    },
    "cooccurrence": {
        "name": "Co-occurrence Extraction",
        "definition": "Links every pair of entities occurring in the same sentence or window with a generic related_to predicate. Achieves high recall but low precision.",
        "example": '(Einstein, related_to, Niels Bohr) with low baseline confidence 0.35.',
    },
    "hybrid": {
        "name": "Hybrid Extraction",
        "definition": "Applies pattern matching first with high confidence, and falls back to co-occurrence with lower confidence when no cue matches. Balances precision and recall.",
        "example": 'Yields (Google, acquired, YouTube) at 0.95 and (Google, related_to, Android) at 0.35.',
    },
    "confidence": {
        "name": "Confidence Score",
        "definition": "A heuristic score in [0, 1] indicating extraction reliability. Allows downstream graph ingestion to filter noise.",
        "example": "Score 0.95 for unambiguous cue 'acquired'; score 0.35 for unguided co-occurrence.",
    },
    "threshold": {
        "name": "Minimum Confidence Threshold",
        "definition": "The cutoff below which candidate triples are discarded. Higher thresholds favour precision; lower thresholds favour recall.",
        "example": "Threshold 0.70 filters out weak verbal ties and generic co-occurrence edges.",
    },
    "word_distance": {
        "name": "Maximum Word Distance",
        "definition": "The maximum permitted token distance between two entities in a sentence. Prevents linking entities in unrelated clauses.",
        "example": "At limit = 10 words, entities 22 words apart in different clauses are rejected.",
    },
    "candidate_pair": {
        "name": "Candidate Pair",
        "definition": "A pair of entities found in the same sentence evaluated by the extraction engine for a valid relation.",
        "example": '"Page and Brin founded Google" generates pairs (Page, Brin) and (Brin, Google).',
    },
    "precision": {
        "name": "Precision (P)",
        "definition": "The fraction of extracted triples that are factually correct: P = Correct Extractions / Total Extracted.",
        "example": "10 triples extracted, 8 correct -> Precision = 0.80 (80%).",
    },
    "recall": {
        "name": "Recall (R)",
        "definition": "The fraction of actual relationships present in the text that were extracted: R = Correct Extractions / Total Present in Ground Truth.",
        "example": "12 relations exist in text, 8 extracted -> Recall = 0.67 (66.7%).",
    },
    "f1": {
        "name": "F1 Score",
        "definition": "The harmonic mean of Precision and Recall: F1 = 2 * (P * R) / (P + R). Penalises extreme imbalances between precision and recall.",
        "example": "P = 0.80, R = 0.67 -> F1 = 0.73.",
    },
    "reference_set": {
        "name": "Ground Truth Reference Set",
        "definition": "A human-curated set of gold standard triples known to be asserted by the text, used to evaluate extraction accuracy.",
        "example": "The Tech & Business corpus carries 14 gold triples against which systems are benchmarked.",
    },
    "false_positive": {
        "name": "False Positive (FP)",
        "definition": "A triple extracted by the system that is unsupported or incorrect based on the text. Lowers precision.",
        "example": 'Extracting (California, founded, Google) due to word proximity.',
    },
    "false_negative": {
        "name": "False Negative (FN)",
        "definition": "A valid relationship in the text that the extractor failed to extract (an omission). Lowers recall.",
        "example": 'Failing to extract a passive-voice fact: "YouTube was bought by Google".',
    },
    "cypher": {
        "name": "Cypher Query Language",
        "definition": "The declarative graph query language for Neo4j. Represents graph patterns visually using ASCII-art syntax: (node)-[:EDGE]->(node).",
        "example": "MATCH (s:Entity)-[:FOUNDED]->(o:Entity) RETURN s.name, o.name;",
    },
    "merge": {
        "name": "MERGE Clause (Neo4j)",
        "definition": "Cypher clause that ensures a node or relationship is created only if it does not already exist, preventing graph duplication upon re-import.",
        "example": "MERGE (s:Entity {name: 'Google'})",
    },
    "provenance": {
        "name": "Provenance",
        "definition": "The audit trail tracking the exact source sentence and text span from which a knowledge triple was derived.",
        "example": 'Triple (Google, acquired, YouTube) stores provenance: "Google acquired YouTube in 2006."',
    },
    "coreference": {
        "name": "Co-reference Resolution",
        "definition": "Resolving pronouns ('he', 'she', 'they', 'it') and descriptive noun phrases back to their antecedent entity mention.",
        "example": '"Marie Curie won the prize. She moved to Paris." -> "She" refers to Marie Curie.',
    },
    "dependency_parsing": {
        "name": "Dependency Parsing",
        "definition": "Syntactic parsing that maps grammatical relationships between words. The shortest path between entities identifies the governing verb.",
        "example": "In complex or inverted sentences, dependency parse paths bridge long entity separations.",
    },
    "open_ie": {
        "name": "Open Information Extraction (OpenIE)",
        "definition": "Relation extraction that extracts arbitrary relation phrases directly from text without adhering to a fixed, predefined schema.",
        "example": 'Extracts ("Sundar Pichai", "served as senior executive at", "Google") directly.',
    },
    "brittleness": {
        "name": "Brittleness of Rules",
        "definition": "The susceptibility of hand-crafted pattern rules to fail when encountering varied linguistic paraphrasing, metaphors, or complex syntax.",
        "example": 'A rule matching "acquired" will miss colloquial paraphrases like "swallowed up" or "brought under its umbrella".',
    },
    "error_propagation": {
        "name": "Error Propagation",
        "definition": "A cascade in a multi-stage pipeline where errors in early stages (e.g. misclassified NER entity) irrevocably corrupt later stages (RE).",
        "example": 'If "Paris" is misclassified as PERSON, RE might output (Paris, studied_at, University).',
    }
}

_MARKER_REGEX = re.compile(r"\[\[([a-z0-9_]+)(?:\|([^\]]+))?\]\]")
_PROPER_TERMS = {"cypher", "merge", "f1", "open_ie", "information_retrieval"}


def inject_custom_styles():
    """Injects custom CSS for interactive pop-up notes into the Streamlit app."""
    st.markdown(_TERM_STYLE, unsafe_allow_html=True)


# `.streamlit/config.toml` fixes a single (light, navy) native theme - Streamlit itself has
# no way to swap the *page chrome* (background, text, widget colours) at runtime from Python.
# The in-app "Light/Dark" control therefore used to only recolour the Plotly charts, leaving
# a dark-styled graph sitting on a white page. This CSS override makes the picker actually
# repaint the page itself, so it genuinely does something.
_DARK_MODE_CSS = """
<style>
:root, .stApp { color-scheme: dark; }
.stApp {
    background-color: #0B1220 !important;
    color: #F1F5F9 !important;
}
.stApp, .stApp p, .stApp span, .stApp label, .stApp li, .stApp small,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
    color: #F1F5F9 !important;
}
[data-testid="stHeader"] { background-color: rgba(0,0,0,0) !important; }
[data-testid="stSidebar"] { background-color: #0F1A2E !important; border-right: 1px solid rgba(148,163,184,0.18); }
[data-testid="stSidebar"] * { color: #F1F5F9 !important; }
[data-testid="stMetricValue"], [data-testid="stMetricLabel"] { color: #F1F5F9 !important; }
.stMarkdown code, code, pre { background-color: #1E293B !important; color: #E2E8F0 !important; }
[data-testid="stExpander"] { background-color: #10192B !important; border: 1px solid rgba(148,163,184,0.25) !important; }
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div, textarea, input {
    background-color: #1E293B !important;
    color: #F1F5F9 !important;
    border-color: rgba(148,163,184,0.35) !important;
}
[data-testid="stDataFrame"] { background-color: #10192B !important; }
hr { border-color: rgba(148,163,184,0.3) !important; }
div.kgnote { background-color: rgba(30,41,59,0.85) !important; border-color: rgba(148,163,184,0.3) !important; }
</style>
"""

_LIGHT_MODE_CSS = "<style>:root, .stApp { color-scheme: light; }</style>"


def inject_mode_css(dark: bool):
    """Applies (or resets) the CSS override above so the chosen mode visibly changes the
    whole page, not just the Plotly chart colours."""
    st.markdown(_DARK_MODE_CSS if dark else _LIGHT_MODE_CSS, unsafe_allow_html=True)


def _format_inline_name(key: str, name: str) -> str:
    if key in _PROPER_TERMS or not name:
        return name
    return name[0].lower() + name[1:] if len(name) > 1 else name.lower()


def _term_note_html(key: str, display: str = "") -> str:
    """`display`, if given, is already-escaped safe HTML (see `_inline`) - it is used as-is."""
    entry = TERMS.get(key)
    if not entry:
        return display or html.escape(key.replace("_", " "))
    name = html.escape(entry["name"])
    shown = display or html.escape(_format_inline_name(key, entry["name"]))
    definition = html.escape(entry["definition"])
    example = html.escape(entry.get("example", "")).replace("\n", "<br>")
    example_html = f'<span class="kgnote-example"><b>Example:</b><br>{example}</span>' if example else ""
    return (
        f'<details class="kgterm"><summary>{shown}</summary>'
        f'<div class="kgnote">'
        f'<div class="kgnote-header">'
        f'<span class="kgnote-term">{name}</span>'
        f'<span class="kgnote-badge">IR Concept Note</span>'
        f'</div>'
        f'<div class="kgnote-def">{definition}</div>'
        f'{example_html}'
        f'</div></details>'
    )


def _markdown_inline_to_html(fragment: str) -> str:
    """Escapes raw text, then applies inline **bold**, *italic* and `code` markdown."""
    escaped = html.escape(fragment)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", escaped)
    escaped = re.sub(r"`([^`]+)`", r"<code style='background:rgba(128,128,128,0.15); padding:1px 4px; border-radius:3px;'>\1</code>", escaped)
    return escaped


def _inline(text: str) -> str:
    """Inline markdown (bold/italic/code) is applied first, so a `**...**` span may safely
    wrap a `[[term]]` marker; the (now HTML-safe) marker text is substituted in afterwards,
    so it is not escaped a second time."""
    formatted = _markdown_inline_to_html(text)
    parts = []
    cursor = 0
    for match in _MARKER_REGEX.finditer(formatted):
        parts.append(formatted[cursor:match.start()])
        parts.append(_term_note_html(match.group(1), match.group(2) or ""))
        cursor = match.end()
    parts.append(formatted[cursor:])
    return "".join(parts)


_HEADER_LINE = re.compile(r"^(#{1,4})\s+(.*)$")
_BULLET_LINE = re.compile(r"^[-*]\s+(.*)$")
_NUMBERED_LINE = re.compile(r"^\d+\.\s+(.*)$")
_HEADER_TAG_BY_LEVEL = {1: "h3", 2: "h4", 3: "h4", 4: "h5"}


def annotate_terms(text: str) -> str:
    """Renders prose that mixes ordinary block Markdown (headers, lists, paragraphs) with
    [[key]] / [[key|Display]] markers for clickable inline term-disclosure notes."""
    lines = text.strip("\n").split("\n")
    blocks: List[str] = []
    paragraph: List[str] = []

    def flush_paragraph():
        if not paragraph:
            return
        joined = " ".join(l.strip() for l in paragraph if l.strip())
        if joined:
            blocks.append(f"<p style='margin-bottom:0.75rem; line-height:1.6;'>{_inline(joined)}</p>")
        paragraph.clear()

    # An indented block of `- ` bullets directly beneath a numbered item (a common pattern
    # in this file's theory prose) is that item's nested detail, not a new top-level list -
    # so it must not reset the outer <ol> back to "1.".
    def consume_nested_bullets(start: int) -> Tuple[str, int]:
        j = start
        nested_items = []
        while j < len(lines) and lines[j][:1].isspace() and _BULLET_LINE.match(lines[j].strip()):
            nested_items.append(_BULLET_LINE.match(lines[j].strip()).group(1))
            j += 1
        if not nested_items:
            return "", start
        html = (
            "<ul style='margin:0.2rem 0 0.4rem 0; padding-left:1.3rem; line-height:1.55;'>"
            + "".join(f"<li>{_inline(item)}</li>" for item in nested_items) + "</ul>"
        )
        return html, j

    i, n = 0, len(lines)
    while i < n:
        stripped = lines[i].strip()
        if not stripped:
            flush_paragraph()
            i += 1
            continue

        header_match = _HEADER_LINE.match(stripped)
        if header_match:
            flush_paragraph()
            tag = _HEADER_TAG_BY_LEVEL.get(len(header_match.group(1)), "h4")
            blocks.append(f"<{tag} style='margin:1.1rem 0 0.5rem 0;'>{_inline(header_match.group(2))}</{tag}>")
            i += 1
            continue

        if _BULLET_LINE.match(stripped) and not lines[i][:1].isspace():
            flush_paragraph()
            items = []
            while i < n and lines[i][:1] != " " and _BULLET_LINE.match(lines[i].strip()):
                items.append(_BULLET_LINE.match(lines[i].strip()).group(1))
                i += 1
            blocks.append(
                "<ul style='margin:0 0 0.75rem 0; padding-left:1.4rem; line-height:1.6;'>"
                + "".join(f"<li>{_inline(item)}</li>" for item in items) + "</ul>"
            )
            continue

        if _NUMBERED_LINE.match(stripped) and not lines[i][:1].isspace():
            flush_paragraph()
            items = []
            while i < n and lines[i][:1] != " " and _NUMBERED_LINE.match(lines[i].strip()):
                content = _NUMBERED_LINE.match(lines[i].strip()).group(1)
                i += 1
                nested_html, i = consume_nested_bullets(i)
                # A single blank line between numbered items is layout only, not a list
                # break, as long as another numbered item follows it.
                if i < n and not lines[i].strip():
                    lookahead = i + 1
                    if lookahead < n and lines[lookahead][:1] != " " and _NUMBERED_LINE.match(lines[lookahead].strip()):
                        i = lookahead
                items.append(f"<li>{_inline(content)}{nested_html}</li>")
            blocks.append(
                "<ol style='margin:0 0 0.75rem 0; padding-left:1.4rem; line-height:1.6;'>"
                + "".join(items) + "</ol>"
            )
            continue

        paragraph.append(lines[i])
        i += 1

    flush_paragraph()
    return "".join(blocks)


def write_annotated(text: str):
    """Convenience helper to render prose with interactive term notes."""
    st.markdown(annotate_terms(text), unsafe_allow_html=True)


def render_glossary_expander():
    """Renders a full searchable reference table of all technical terms."""
    with st.expander("Explore Full Technical Terms Glossary (25+ IR & KG Concepts)", expanded=False):
        st.write("Browse formal definitions, IR roles, and concrete examples for terms used throughout the virtual laboratory.")
        search_query = st.text_input("Filter Glossary by keyword", placeholder="e.g. predicate, recall, confidence", key="glossary_search")
        
        rows = []
        for k, v in sorted(TERMS.items(), key=lambda item: item[1]["name"]):
            if search_query:
                q = search_query.lower()
                if q not in v["name"].lower() and q not in v["definition"].lower() and q not in v.get("example", "").lower():
                    continue
            rows.append({
                "Technical Term": v["name"],
                "Formal Definition": v["definition"],
                "Concrete Lab Example": v.get("example", "—")
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        else:
            st.info("No matching terms found in the glossary.")


# ======================================================================================
# 3. EXPERIMENT CONFIGURATION & EDUCATIONAL CONTENT
# ======================================================================================

EXPERIMENT_CONFIG = {
    "title": "Experiment 7: Relationship Extraction from Text",
    "course": "Knowledge Graphs and Information Retrieval Systems (KGIRS) · Course Code: D17A/B/C",
    "department": "Department of Computer Engineering / Information Technology",
    "portal": "Virtual Laboratory Learning Portal",
    "objectives": [
        "Understand how semantic relationships between entities are identified in unstructured text.",
        "Implement rule-based and pattern-based relation extraction over sentence boundaries.",
        "Represent extracted knowledge as Subject-Predicate-Object (SPO) triples with confidence scoring.",
        "Evaluate extraction precision, recall, and F1 score against human-curated gold standard reference sets.",
        "Prepare extracted knowledge graph triples for downstream graph database ingestion (Experiments 8-9)."
    ]
}

THEORY_CONTENT = {
    "background": """
### Overview & Principles
[[relationship_extraction|Relationship Extraction (RE)]] is the central task in [[information_retrieval|Information Retrieval]] and natural language processing that identifies and classifies semantic relationships holding between pairs of [[entity|entities]] mentioned in unstructured prose.

Given an input sentence such as *"Larry Page co-founded Google in California"*, an RE system automatically recognizes the entities and extracts the structured [[triple|Subject-Predicate-Object (SPO) triple]]:

`(Larry Page) --[founded]--> (Google)`

These triples represent the atomic building blocks of a [[knowledge_graph|Knowledge Graph]]:
- **Nodes** correspond to real-world entities (people, organizations, locations).
- **Directed Edges** correspond to relational predicates (`founded`, `acquired`, `located_in`).

### Extraction Approaches Compared in this Laboratory
1. **[[pattern_based|Pattern-Based Extraction]]**:
   - Matches hand-crafted lexico-syntactic patterns and verbal cues (e.g. *acquired*, *is the CEO of*, *located in*) occurring between entities.
   - **Characteristics**: High [[precision]], transparent and interpretable, but subject to linguistic [[brittleness]] when sentences contain complex clauses or unseen paraphrases.

2. **[[cooccurrence|Co-occurrence Extraction]]**:
   - Assumes that any two entities appearing within the same sentence boundary are related, assigning a generic `related_to` predicate.
   - **Characteristics**: High [[recall]], but low [[precision]] and generates unnamed, weakly-specified edges.

3. **[[hybrid|Hybrid Extraction]]**:
   - Executes pattern matching first. If an explicit [[relation_cue|relation cue]] is matched, it assigns the specific predicate with high [[confidence]]; otherwise, it defaults to a low-confidence co-occurrence link.
   - **Characteristics**: Maximizes coverage while equipping downstream graph ingestion stages with a quantitative confidence filter.

### Systematic Pipeline Workflow
1. **[[named_entity_recognition|Entity Recognition]]**: Identify mentions of people, organizations, places, dates, and domain terms.
2. **[[candidate_pair|Candidate Pair Generation]]**: Identify all pairs of entities occurring within the [[word_distance|maximum word distance]] window in each sentence.
3. **Cue Matching & Lexicon Lookup**: Inspect the intervening text span against the [[relation_cue|relation lexicon]].
4. **[[triple|Triple Construction]]**: Emit structured facts with attached [[provenance]] (source sentence) and [[confidence|confidence score]].
5. **[[knowledge_graph|Graph Construction & Verification]]**: Preview the resulting graph, evaluate accuracy against [[reference_set|ground truth]], and export to [[cypher|Cypher]] scripts.
""",
    "procedure": [
        "Step 1: Study the theoretical principles of relation extraction and examine key IR terms using the interactive notes.",
        "Step 2: Navigate to the Simulation section and choose a sample corpus (or upload/paste custom text).",
        "Step 3: Choose an extraction strategy (Pattern-Based, Co-occurrence, or Hybrid) and adjust the confidence threshold.",
        "Step 4: Execute extraction and explore the 6 analysis tabs: Triples, Annotated Text, Entities, Graph, Flow, and Evaluation.",
        "Step 5: Inspect the interactive Knowledge Graph; toggle predicate labels and test legibility in Light and Dark modes.",
        "Step 6: Click 'Record Current Trial' to log the experiment parameters and results into your session logbook.",
        "Step 7: Complete at least 3 trials comparing different extraction methods and confidence cutoffs.",
        "Step 8: Export the extracted triples as CSV, JSON, or a Neo4j Cypher (.cql) script.",
        "Step 9: Complete the self-grading Conceptual Assessment Quiz to test your understanding.",
        "Step 10: Open Report Generation, enter your student credentials, and download your official PDF Laboratory Report."
    ]
}


# ======================================================================================
# 4. RELATION-EXTRACTION ENGINE (PURE CORE FUNCTIONS)
# ======================================================================================

_STOPWORD_CAPS = {
    "The", "A", "An", "In", "On", "At", "This", "That", "It", "He", "She",
    "They", "We", "I", "His", "Her", "Its", "Their", "From", "To", "With", "By"
}

_ORG_SUFFIXES = (
    "Inc", "Inc.", "Ltd", "Ltd.", "Corp", "Corp.", "LLC", "University",
    "Institute", "Company", "Corporation", "Labs", "Laboratories",
    "Foundation", "Group", "Technologies", "Systems", "Council", "Hospital"
)

_LOC_HINTS = (
    "City", "State", "Street", "River", "Mountain", "Island", "Ocean",
    "Valley", "County", "Province", "Republic", "Kingdom", "Nation"
)

# Relation lexicon: (regex pattern over connecting span) -> (canonical predicate, confidence)
_RELATION_PATTERNS = [
    (r"\b(co-?founded|founded|established|started|launched|set up)\b", "founded", 0.95),
    (r"\b(acquired|bought|purchased|took over|absorbed)\b", "acquired", 0.95),
    (r"\b(is|was|are|were)\s+(the\s+)?(ceo|founder|president|director|chairman|chairwoman|chairperson|leader|chief executive)\s+of\b", "leads", 0.92),
    (r"\b(leads|led|heads|directed|governs)\b", "leads", 0.90),
    (r"\b(works?\s+at|worked\s+at|employed\s+by|employee\s+of|serves?\s+at)\b", "works_at", 0.90),
    (r"\b(located\s+in|based\s+in|headquartered\s+in|situated\s+in|operates?\s+in)\b", "located_in", 0.90),
    (r"\b(born\s+in|native\s+of)\b", "born_in", 0.92),
    (r"\b(married\s+to|husband\s+of|wife\s+of|spouse\s+of)\b", "married_to", 0.92),
    (r"\b(developed|created|invented|designed|built|engineered)\b", "created", 0.90),
    (r"\b(discovered|isolated|identified)\b", "discovered", 0.92),
    (r"\b(published|authored|wrote|formulated)\b", "authored", 0.90),
    (r"\b(subsidiary\s+of|part\s+of|division\s+of|owned\s+by|unit\s+of)\b", "subsidiary_of", 0.90),
    (r"\b(collaborated\s+with|partnered\s+with|partnership\s+with|allied\s+with)\b", "collaborates_with", 0.88),
    (r"\b(merged\s+with|combined\s+with)\b", "merged_with", 0.90),
    (r"\b(studied\s+at|graduated\s+from|attended|alumnus\s+of)\b", "studied_at", 0.90),
    (r"\b(reports\s+to|managed\s+by|subordinate\s+to)\b", "reports_to", 0.85),
    (r"\b(competes\s+with|rival\s+of|competing\s+against)\b", "competes_with", 0.85),
    (r"\b(treats|alleviates|cures|prescribed\s+for)\b", "treats", 0.90),
    (r"\b(causes|leads\s+to|triggers|induces)\b", "causes", 0.88),
    (r"\b(is|was|are|were)\b", "is_a", 0.50),
]

_MAX_WORDS_BETWEEN = 12


def _split_sentences(text: str) -> List[str]:
    """Splits document text into clean sentences while respecting common honorifics and abbreviations."""
    clean = text.strip().replace("\r\n", " ").replace("\n", " ")
    clean = re.sub(r"\b(Dr|Prof|Mr|Mrs|Ms|vs|Inc|Ltd|Corp|e\.g|i\.e)\.", r"\1<DOT>", clean)
    parts = re.split(r"(?<=[.!?])\s+", clean)
    sentences = [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]
    return sentences


def extract_entities(text: str) -> List[Dict[str, Any]]:
    """Lightweight deterministic NER fallback identifying PERSON, ORG, LOCATION, DATE, and TERM."""
    sentences = _split_sentences(text)
    entities = []
    entity_id = 1

    known_locations = {
        "california", "mountain view", "redmond", "warsaw", "paris", "london",
        "mumbai", "bengaluru", "chennai", "delhi", "tokyo", "new york", "san francisco",
        "india", "france", "poland", "usa", "switzerland", "cambridge", "oxford"
    }
    known_orgs = {
        "google", "alphabet", "microsoft", "youtube", "linkedin", "openai",
        "infosys", "tata group", "tata consultancy services", "curie institute",
        "university of paris", "indian space research organisation", "isro",
        "stanford university", "harvard", "mit", "who", "pfizer", "apple", "amazon"
    }

    for sent_idx, sentence in enumerate(sentences):
        # Match capitalized phrases
        # [A-Za-z]* (not just [a-z]*) so brand names with an internal capital, such as
        # "YouTube", "LinkedIn" or "OpenAI", are recognized as a single entity instead of
        # being silently skipped.
        for match in re.finditer(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b", sentence):
            span_text = match.group(1).strip()
            if span_text in _STOPWORD_CAPS:
                continue

            lowered = span_text.lower()
            if lowered in known_locations or any(hint in span_text for hint in _LOC_HINTS):
                label = "LOCATION"
            elif lowered in known_orgs or any(span_text.endswith(sfx) for sfx in _ORG_SUFFIXES):
                label = "ORG"
            elif re.search(r"\b(University|Institute|College|Academy|Hospital|Board)\b", span_text):
                label = "ORG"
            else:
                label = "PERSON"

            entities.append({
                "id": f"e{entity_id}",
                "text": span_text,
                "label": label,
                "start": match.start(),
                "end": match.end(),
                "sentence_idx": sent_idx,
                "sentence": sentence
            })
            entity_id += 1

        # Match 4-digit years as DATE
        for match in re.finditer(r"\b(19\d\d|20\d\d)\b", sentence):
            year_text = match.group(1)
            entities.append({
                "id": f"e{entity_id}",
                "text": year_text,
                "label": "DATE",
                "start": match.start(),
                "end": match.end(),
                "sentence_idx": sent_idx,
                "sentence": sentence
            })
            entity_id += 1

    # Remove duplicates within identical character spans
    seen_spans = set()
    unique_entities = []
    for ent in entities:
        key = (ent["sentence_idx"], ent["start"], ent["end"])
        if key not in seen_spans:
            seen_spans.add(key)
            unique_entities.append(ent)

    return unique_entities


def extract_relationships(
    text: str,
    entities: Optional[List[Dict[str, Any]]] = None,
    method: str = "Hybrid (Pattern + Co-occurrence)",
    max_word_distance: int = _MAX_WORDS_BETWEEN
) -> List[Dict[str, Any]]:
    """Extracts Subject-Predicate-Object triples over sentence entity pairs."""
    if entities is None:
        entities = extract_entities(text)

    # Group entities by sentence
    by_sentence: Dict[int, List[Dict[str, Any]]] = {}
    for ent in entities:
        by_sentence.setdefault(ent["sentence_idx"], []).append(ent)

    triples = []
    triple_id = 1

    for sent_idx, sent_ents in by_sentence.items():
        if len(sent_ents) < 2:
            continue
        # Sort by occurrence order in the sentence
        sorted_ents = sorted(sent_ents, key=lambda e: e["start"])

        # Only pair adjacent entity mentions (no other entity in between). Pairing every
        # combination in the sentence lets a relation cue between two OTHER entities get
        # wrongly attributed across an intervening entity - e.g. in "Larry Page and Sergey
        # Brin co-founded Google in California in 1998", pairing Larry Page with California
        # or 1998 directly would (wrongly) match the "co-founded" cue in between them and
        # emit a spurious high-confidence "Larry Page -> founded -> California" triple.
        # Restricting pattern/co-occurrence pairing to adjacent mentions keeps every emitted
        # triple tied to the actual local relation cue.
        for i in range(len(sorted_ents) - 1):
                subj = sorted_ents[i]
                obj = sorted_ents[i + 1]

                # If an entity is identical in string, skip self-links
                if subj["text"].lower() == obj["text"].lower():
                    continue

                sentence = subj["sentence"]
                span_between = sentence[subj["end"]:obj["start"]].strip()
                words_between = len(span_between.split())

                if words_between > max_word_distance:
                    continue

                matched_predicate = None
                confidence = 0.0
                matched_cue = None

                # Pattern matching over span
                for pattern, pred, conf in _RELATION_PATTERNS:
                    m = re.search(pattern, span_between, re.IGNORECASE)
                    if m:
                        matched_predicate = pred
                        confidence = conf
                        matched_cue = m.group(0)
                        break

                is_hybrid = "Hybrid" in method
                is_pattern = "Pattern" in method
                is_cooc = "Co-occurrence" in method and not is_hybrid

                if matched_predicate and (is_pattern or is_hybrid):
                    # Type constraint heuristic validation
                    if matched_predicate == "born_in" and subj["label"] != "PERSON":
                        confidence *= 0.6
                    if matched_predicate == "leads" and obj["label"] not in ("ORG", "LOCATION"):
                        confidence *= 0.75

                    triples.append({
                        "id": f"t{triple_id}",
                        "subject": subj["text"],
                        "subject_label": subj["label"],
                        "predicate": matched_predicate,
                        "object": obj["text"],
                        "object_label": obj["label"],
                        "confidence": round(confidence, 2),
                        "cue": matched_cue,
                        "method": "Pattern-Based",
                        "sentence": sentence,
                        "sentence_idx": sent_idx
                    })
                    triple_id += 1

                elif (is_cooc or is_hybrid) and not matched_predicate:
                    # Fallback co-occurrence relation
                    cooc_conf = max(0.20, round(0.45 - 0.02 * words_between, 2))
                    triples.append({
                        "id": f"t{triple_id}",
                        "subject": subj["text"],
                        "subject_label": subj["label"],
                        "predicate": "related_to",
                        "object": obj["text"],
                        "object_label": obj["label"],
                        "confidence": cooc_conf,
                        "cue": None,
                        "method": "Co-occurrence",
                        "sentence": sentence,
                        "sentence_idx": sent_idx
                    })
                    triple_id += 1

    return triples


def compute_extraction_metrics(entities: List[Dict[str, Any]], triples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summary statistics for trial logging, display metrics, and reporting."""
    n_entities = len({e["text"].lower() for e in entities})
    n_triples = len(triples)
    high_conf = sum(1 for t in triples if t["confidence"] >= 0.80)
    avg_conf = round(float(np.mean([t["confidence"] for t in triples])), 2) if triples else 0.0
    density = round(n_triples / n_entities, 2) if n_entities else 0.0
    n_predicates = len({t["predicate"] for t in triples})
    return {
        "entities": n_entities,
        "triples": n_triples,
        "high_confidence_triples": high_conf,
        "avg_confidence": avg_conf,
        "relation_density": density,
        "unique_predicates": n_predicates
    }


def evaluate_against_reference_set(
    extracted_triples: List[Dict[str, Any]],
    gold_triples: List[Tuple[str, str, str]]
) -> Dict[str, Any]:
    """Evaluates extracted triples against a human ground truth reference set."""
    if not gold_triples:
        return {
            "has_gold": False,
            "gold_count": 0,
            "extracted_count": len(extracted_triples),
            "tp_count": 0,
            "fp_count": len(extracted_triples),
            "fn_count": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "tp_list": [],
            "fp_list": extracted_triples,
            "fn_list": []
        }

    # Normalize tuples for robust matching
    def norm_triple(s: str, p: str, o: str) -> Tuple[str, str, str]:
        p_norm = p.lower().replace(" ", "_")
        if p_norm in ("co-founded", "started", "established"):
            p_norm = "founded"
        elif p_norm in ("bought", "purchased", "took_over"):
            p_norm = "acquired"
        elif p_norm in ("headquartered_in", "based_in"):
            p_norm = "located_in"
        return (s.lower().strip(), p_norm, o.lower().strip())

    gold_set = {norm_triple(g[0], g[1], g[2]): g for g in gold_triples}
    matched_gold = set()

    tp_list = []
    fp_list = []

    for ext in extracted_triples:
        key = norm_triple(ext["subject"], ext["predicate"], ext["object"])
        if key in gold_set:
            tp_list.append(ext)
            matched_gold.add(key)
        else:
            fp_list.append(ext)

    fn_list = [orig for key, orig in gold_set.items() if key not in matched_gold]

    tp = len(tp_list)
    fp = len(fp_list)
    fn = len(fn_list)

    precision = round(tp / (tp + fp), 3) if (tp + fp) > 0 else 0.0
    recall = round(tp / (tp + fn), 3) if (tp + fn) > 0 else 0.0
    f1 = round(2 * (precision * recall) / (precision + recall), 3) if (precision + recall) > 0 else 0.0

    return {
        "has_gold": True,
        "gold_count": len(gold_triples),
        "extracted_count": len(extracted_triples),
        "tp_count": tp,
        "fp_count": fp,
        "fn_count": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp_list": tp_list,
        "fp_list": fp_list,
        "fn_list": fn_list
    }


# ======================================================================================
# 5. SAMPLE CORPORA & GROUND TRUTH REFERENCE SETS
# ======================================================================================

CORPORA: Dict[str, Dict[str, Any]] = {
    "Tech & Business": {
        "name": "Technology and Enterprise",
        "level": "Beginner",
        "blurb": "Concise factual statements on technology companies in straightforward active voice.",
        "purpose": (
            "Every sentence is in the active voice, consists of clear subject-verb-object syntax, "
            "and places the relationship verb directly between the two entities. This represents "
            "the baseline scenario where rule-based extraction performs with high precision."
        ),
        "observe": [
            "Observe that 'Larry Page' and 'Google' are linked by 'founded' with confidence >= 0.90.",
            "Notice how 'Google acquired YouTube' generates a high-confidence 'acquired' predicate.",
            "Compare how 'Sundar Pichai is the CEO of Google' matches the 'leads' regex rule."
        ],
        "text": (
            "Larry Page and Sergey Brin co-founded Google in California in 1998. "
            "Google acquired YouTube in 2006 and developed the Android operating system. "
            "Sundar Pichai is the CEO of Google and reports to the Alphabet board. "
            "Alphabet is headquartered in Mountain View. "
            "Microsoft, based in Redmond, competes with Google in cloud computing. "
            "Satya Nadella leads Microsoft as its chief executive. "
            "Microsoft acquired LinkedIn and partnered with OpenAI."
        ),
        "gold": [
            ("Larry Page", "founded", "Google"),
            ("Sergey Brin", "founded", "Google"),
            ("Google", "located_in", "California"),
            ("Google", "acquired", "YouTube"),
            ("Google", "created", "Android"),
            ("Sundar Pichai", "leads", "Google"),
            ("Sundar Pichai", "reports_to", "Alphabet"),
            ("Alphabet", "located_in", "Mountain View"),
            ("Microsoft", "located_in", "Redmond"),
            ("Microsoft", "competes_with", "Google"),
            ("Satya Nadella", "leads", "Microsoft"),
            ("Microsoft", "acquired", "LinkedIn"),
            ("Microsoft", "collaborates_with", "OpenAI"),
        ]
    },
    "Science & Academia": {
        "name": "Science and Academia",
        "level": "Intermediate",
        "blurb": "Biographical assertions concerning scientists, universities, and discoveries.",
        "purpose": (
            "Combines PERSON, ORG, LOCATION, and DATE entities. Demonstrates the need for entity type "
            "constraints (e.g. 'born_in' accepts a PERSON subject and LOCATION or DATE object)."
        ),
        "observe": [
            "In 'Pierre Curie, her husband', the relationship is expressed via an appositive noun phrase.",
            "Notice that Marie Curie is born in Warsaw (LOCATION) and studied at University of Paris (ORG).",
            "Observe how Albert Einstein is linked to 'theory of relativity' and collaborators."
        ],
        "text": (
            "Marie Curie was born in Warsaw and studied at the University of Paris. "
            "Curie discovered radium and polonium alongside Pierre Curie, her husband. "
            "The Curie Institute was established in Paris to continue scientific research. "
            "Albert Einstein published his theory of relativity in 1905. "
            "Einstein collaborated with Niels Bohr on quantum mechanics debates."
        ),
        "gold": [
            ("Marie Curie", "born_in", "Warsaw"),
            ("Marie Curie", "studied_at", "University of Paris"),
            ("Curie", "discovered", "radium"),
            ("Curie", "married_to", "Pierre Curie"),
            ("Curie Institute", "located_in", "Paris"),
            ("Albert Einstein", "authored", "theory of relativity"),
            ("Einstein", "collaborates_with", "Niels Bohr"),
        ]
    },
    "Indian Industry & Space": {
        "name": "Indian Industry and Space Technology",
        "level": "Intermediate",
        "blurb": "Institutional and executive statements from Indian industry and space exploration.",
        "purpose": (
            "Demonstrates extraction over domain-specific organizations and regional geographical entities."
        ),
        "observe": [
            "Look at the acquisition of Jaguar Land Rover by Tata Group.",
            "Examine how 'headquartered in Bengaluru' is detected as a located_in relationship.",
            "Observe the CEO and birthplace assertions regarding Sundar Pichai."
        ],
        "text": (
            "Ratan Tata was the chairman of the Tata Group for more than two decades. "
            "The Tata Group is headquartered in Mumbai. "
            "Tata Group acquired Jaguar Land Rover in 2008. "
            "Narayana Murthy and Nandan Nilekani founded Infosys. "
            "Infosys is headquartered in Bengaluru and competes with Tata Consultancy Services. "
            "The Indian Space Research Organisation is based in Bengaluru. "
            "Sundar Pichai was born in Chennai. "
            "Sundar Pichai is the CEO of Google."
        ),
        "gold": [
            ("Ratan Tata", "leads", "Tata Group"),
            ("Tata Group", "located_in", "Mumbai"),
            ("Tata Group", "acquired", "Jaguar Land Rover"),
            ("Narayana Murthy", "founded", "Infosys"),
            ("Nandan Nilekani", "founded", "Infosys"),
            ("Infosys", "located_in", "Bengaluru"),
            ("Infosys", "competes_with", "Tata Consultancy Services"),
            ("Indian Space Research Organisation", "located_in", "Bengaluru"),
            ("Sundar Pichai", "born_in", "Chennai"),
            ("Sundar Pichai", "leads", "Google"),
        ]
    },
    "Healthcare & Medicine": {
        "name": "Healthcare and Biomedical Research",
        "level": "Advanced",
        "blurb": "Clinical and pharmaceutical relations between institutions, therapies, and indications.",
        "purpose": (
            "Introduces specialized clinical relations (treats, causes). Demonstrates how expanding the "
            "lexicon allows rule-based extractors to adapt across technical domains."
        ),
        "observe": [
            "Observe the extraction of 'treats' between pharmacological agents and clinical conditions.",
            "Notice institutional affiliations connecting researchers to research hospitals."
        ],
        "text": (
            "Alexander Fleming discovered penicillin in London in 1928. "
            "Penicillin treats bacterial infections effectively. "
            "The World Health Organisation is based in Geneva. "
            "Pfizer partnered with BioNTech to produce vaccines. "
            "Chronic smoking causes cardiovascular disease and respiratory failure."
        ),
        "gold": [
            ("Alexander Fleming", "discovered", "penicillin"),
            ("Alexander Fleming", "located_in", "London"),
            ("Penicillin", "treats", "bacterial infections"),
            ("World Health Organisation", "located_in", "Geneva"),
            ("Pfizer", "collaborates_with", "BioNTech"),
            ("smoking", "causes", "cardiovascular disease"),
        ]
    }
}


# ======================================================================================
# 6. DOCUMENT UPLOAD LOADER
# ======================================================================================

MAX_UPLOAD_CHARACTERS = 25_000


@dataclass
class LoadedDocument:
    filename: str
    extension: str
    text: str
    raw_character_count: int
    truncated: bool = False
    error: Optional[str] = None

    def summary(self) -> str:
        if self.error:
            return f"Failed to load `{self.filename}`: {self.error}"
        status = " (truncated to limit)" if self.truncated else ""
        return f"`{self.filename}` ({self.raw_character_count:,} characters{status})"


def load_uploaded_document(uploaded: Any) -> LoadedDocument:
    """Reads uploaded files (.txt, .md, .csv, .tsv, .pdf, .docx)."""
    name = getattr(uploaded, "name", "uploaded_file")
    ext = os.path.splitext(name)[1].lower().lstrip(".")
    try:
        data = uploaded.read() if hasattr(uploaded, "read") else bytes(uploaded)
    except Exception as exc:
        return LoadedDocument(name, ext, "", 0, error=f"Could not read upload stream: {exc}")

    text = ""
    error = None
    try:
        if ext in ("txt", "text", "log"):
            text = data.decode("utf-8", errors="replace")
        elif ext in ("md", "markdown"):
            raw = data.decode("utf-8", errors="replace")
            text = re.sub(r"#+\s*", "", raw)
            text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        elif ext in ("csv", "tsv"):
            delim = "\t" if ext == "tsv" else ","
            reader = csv.reader(io.StringIO(data.decode("utf-8", errors="replace")), delimiter=delim)
            rows = [" ".join(cell.strip() for cell in row if cell.strip()) for row in reader]
            text = ". ".join(r for r in rows if r)
        elif ext == "pdf":
            try:
                import pypdf
                reader = pypdf.PdfReader(io.BytesIO(data))
                pages = [page.extract_text() or "" for page in reader.pages]
                text = " ".join(pages)
            except ImportError:
                error = "pypdf package is required for PDF parsing. Please run `pip install pypdf`."
        elif ext in ("docx", "doc"):
            try:
                import docx
                doc = docx.Document(io.BytesIO(data))
                text = " ".join(p.text for p in doc.paragraphs if p.text.strip())
            except ImportError:
                error = "python-docx package is required for Word documents. Please run `pip install python-docx`."
        else:
            text = data.decode("utf-8", errors="replace")
    except Exception as e:
        error = f"Error decoding file: {e}"

    clean_text = " ".join(text.split())
    raw_len = len(clean_text)
    truncated = False
    if raw_len > MAX_UPLOAD_CHARACTERS:
        clean_text = clean_text[:MAX_UPLOAD_CHARACTERS]
        truncated = True

    return LoadedDocument(
        filename=name,
        extension=ext,
        text=clean_text,
        raw_character_count=raw_len,
        truncated=truncated,
        error=error
    )


# ======================================================================================
# 7. HIGH-PERFORMANCE GRAPH LAYOUT ALGORITHM
# ======================================================================================

MAX_GRAPH_NODES = 28


def _components(count: int, edges: Sequence[Tuple[int, int]]) -> List[List[int]]:
    """Partitions node indices into connected components, sorted largest first."""
    parent = list(range(count))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for start, end in edges:
        if 0 <= start < count and 0 <= end < count:
            ra, rb = find(start), find(end)
            if ra != rb:
                parent[rb] = ra

    groups: Dict[int, List[int]] = {}
    for node in range(count):
        groups.setdefault(find(node), []).append(node)
    return sorted(groups.values(), key=len, reverse=True)


def _force_layout(count: int, edges: Sequence[Tuple[int, int]], iterations: int = 400, seed: int = 11) -> np.ndarray:
    """Deterministic Fruchterman-Reingold spring embedder with boundary damping."""
    if count == 0:
        return np.zeros((0, 2))
    if count == 1:
        return np.zeros((1, 2))
    if count == 2:
        return np.array([[-0.8, 0.0], [0.8, 0.0]])

    rng = np.random.default_rng(seed)
    angles = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
    pos = np.column_stack([np.cos(angles), np.sin(angles)]) + rng.normal(0.0, 0.03, (count, 2))

    ideal = 2.6 * math.sqrt(1.0 / count)
    temperature = 0.28
    cooling = temperature / (iterations + 1)
    edge_array = np.array(edges, dtype=int) if len(edges) else np.zeros((0, 2), dtype=int)

    for _ in range(iterations):
        delta = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(delta, axis=-1)
        np.fill_diagonal(dist, np.inf)
        dist = np.clip(dist, 0.01, None)
        displacement = np.einsum("ijk,ij->ik", delta / dist[:, :, None], (ideal * ideal) / dist)

        if len(edge_array):
            s, e = edge_array[:, 0], edge_array[:, 1]
            diff = pos[s] - pos[e]
            l = np.clip(np.linalg.norm(diff, axis=1), 0.01, None)
            pull = (diff / l[:, None]) * ((l * l) / ideal)[:, None]
            np.add.at(displacement, s, -pull)
            np.add.at(displacement, e, pull)

        displacement -= pos * 0.012
        l = np.clip(np.linalg.norm(displacement, axis=1), 1e-9, None)
        pos += (displacement / l[:, None]) * np.minimum(l, temperature)[:, None]
        temperature = max(temperature - cooling, 1e-4)

    return _normalise_layout(pos)


def _normalise_layout(pos: np.ndarray) -> np.ndarray:
    if len(pos) <= 1:
        return np.zeros_like(pos)
    low, high = pos.min(axis=0), pos.max(axis=0)
    span = high - low
    scale = max(span[0], span[1]) or 1.0
    centred = pos - (low + high) / 2.0
    return centred / (scale / 2.0)


def _label_boxes(names: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    half_width = np.array([0.022 + 0.006 * len(n[:18]) for n in names], dtype=float)
    half_height = np.full(len(names), 0.060, dtype=float)
    return half_width, half_height


def _separate_labels(pos: np.ndarray, half_width: np.ndarray, half_height: np.ndarray, rounds: int = 140) -> np.ndarray:
    if len(pos) < 2:
        return pos
    pos = pos.copy()
    for _ in range(rounds):
        dx = pos[:, 0][:, None] - pos[:, 0][None, :]
        dy = pos[:, 1][:, None] - pos[:, 1][None, :]
        need_x = half_width[:, None] + half_width[None, :]
        need_y = half_height[:, None] + half_height[None, :]
        ox = need_x - np.abs(dx)
        oy = need_y - np.abs(dy)
        colliding = (ox > 0) & (oy > 0)
        np.fill_diagonal(colliding, False)
        rows, cols = np.where(np.triu(colliding))
        if len(rows) == 0:
            break
        for f, s in zip(rows, cols):
            if ox[f, s] <= oy[f, s]:
                dir_v = np.array([1.0 if dx[f, s] >= 0 else -1.0, 0.0])
                amt = ox[f, s] / 2.0 + 1e-3
            else:
                dir_v = np.array([0.0, 1.0 if dy[f, s] >= 0 else -1.0])
                amt = oy[f, s] / 2.0 + 1e-3
            pos[f] += dir_v * amt
            pos[s] -= dir_v * amt
    return pos


def _graph_layout(names: Sequence[str], edges: Sequence[Tuple[int, int]], seed: int = 11) -> np.ndarray:
    count = len(names)
    if count <= 1:
        return np.zeros((count, 2))

    groups = _components(count, edges)
    positions = np.zeros((count, 2))

    local_layouts = []
    for grp in groups:
        idx_map = {n: i for i, n in enumerate(grp)}
        inner_edges = [(idx_map[s], idx_map[e]) for s, e in edges if s in idx_map and e in idx_map]
        local_layouts.append(_force_layout(len(grp), inner_edges, seed=seed))

    cols = max(1, int(math.ceil(math.sqrt(len(groups)))))
    rows = max(1, int(math.ceil(len(groups) / cols)))
    weights = [max(1.0, math.sqrt(len(g))) for g in groups]
    largest = max(weights)

    for pos_idx, (grp, lay) in enumerate(zip(groups, local_layouts)):
        r, c = divmod(pos_idx, cols)
        cx = (c + 0.5) / cols * 2.0 - 1.0
        cy = 1.0 - (r + 0.5) / rows * 2.0
        share = weights[pos_idx] / largest
        sx = (0.94 / cols) * (0.45 + 0.55 * share)
        sy = (0.94 / rows) * (0.45 + 0.55 * share)
        for node, pt in zip(grp, lay):
            positions[node] = [cx + pt[0] * sx, cy + pt[1] * sy]

    if len(groups) == 1:
        positions = _normalise_layout(positions) * 0.92

    hw, hh = _label_boxes(names)
    positions = _separate_labels(positions, hw, hh)
    return positions


def _label_positions(pos: np.ndarray, edges: Sequence[Tuple[int, int]]) -> List[str]:
    count = len(pos)
    neighbors: Dict[int, List[int]] = {i: [] for i in range(count)}
    for s, e in edges:
        neighbors[s].append(e)
        neighbors[e].append(s)

    options = [
        ("bottom center", np.array([0.0, -1.0])),
        ("top center", np.array([0.0, 1.0])),
        ("middle right", np.array([1.0, 0.0])),
        ("middle left", np.array([-1.0, 0.0])),
    ]
    chosen = []
    for i in range(count):
        if not neighbors[i]:
            chosen.append("bottom center")
            continue
        direction = np.zeros(2)
        for nbr in neighbors[i]:
            offset = pos[nbr] - pos[i]
            l = float(np.linalg.norm(offset)) or 1.0
            direction += offset / l
        best = min(options, key=lambda opt: float(np.dot(direction, opt[1])) - (0.25 if "center" in opt[0] else 0.0))
        chosen.append(best[0])
    return chosen


# ======================================================================================
# 8. POLISHED VISUALIZATION BUILDERS (PLOTLY & GRAPH FIGURES)
# ======================================================================================

def _build_graph_figure(
    triples: List[Dict[str, Any]],
    theme: Theme,
    show_predicates: bool = True,
    seed: int = 11,
    type_filter: Optional[List[str]] = None
) -> go.Figure:
    """Constructs a high-contrast, publication-grade interactive knowledge graph."""
    if not triples:
        fig = go.Figure()
        fig.add_annotation(
            text="No relationships to display.<br>Lower the confidence threshold or switch extraction method.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(color=theme.muted, size=13)
        )
        fig.update_layout(
            height=360, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(visible=False), yaxis=dict(visible=False)
        )
        return fig

    # Apply optional entity type filtering
    filtered_triples = triples
    if type_filter and len(type_filter) > 0:
        filtered_triples = [
            t for t in triples
            if t["subject_label"] in type_filter or t["object_label"] in type_filter
        ]
        if not filtered_triples:
            filtered_triples = triples  # Fallback to prevent blank graph

    # Thinned subset if graph exceeds max legibility bounds
    if len(filtered_triples) > MAX_GRAPH_NODES * 2:
        sorted_t = sorted(filtered_triples, key=lambda t: -t["confidence"])
        seen_nodes = set()
        kept = []
        for t in sorted_t:
            new_nodes = {t["subject"], t["object"]} - seen_nodes
            if len(seen_nodes) + len(new_nodes) > MAX_GRAPH_NODES and kept:
                continue
            seen_nodes |= new_nodes
            kept.append(t)
        filtered_triples = kept

    names: List[str] = []
    labels: Dict[str, str] = {}
    for t in filtered_triples:
        for name, label in ((t["subject"], t["subject_label"]), (t["object"], t["object_label"])):
            if name not in labels:
                labels[name] = label
                names.append(name)

    index_of = {name: i for i, name in enumerate(names)}
    edges = [(index_of[t["subject"]], index_of[t["object"]]) for t in filtered_triples]

    pos = _graph_layout(names, edges, seed=seed)
    label_at = _label_positions(pos, edges)

    degree: Dict[str, int] = {}
    for t in filtered_triples:
        degree[t["subject"]] = degree.get(t["subject"], 0) + 1
        degree[t["object"]] = degree.get(t["object"], 0) + 1

    fig = go.Figure()

    # Draw curved edges with directional arrows
    drawn_pairs: Dict[Tuple[int, int], int] = {}
    lbl_x, lbl_y, lbl_text, lbl_hover = [], [], [], []

    for t, (start, end) in zip(filtered_triples, edges):
        if start == end:
            continue
        x0, y0 = pos[start]
        x1, y1 = pos[end]
        pair = (min(start, end), max(start, end))
        rank = drawn_pairs.get(pair, 0)
        drawn_pairs[pair] = rank + 1

        length = math.hypot(x1 - x0, y1 - y0) or 1.0
        ux, uy = (x1 - x0) / length, (y1 - y0) / length
        bow = 0.0 if rank == 0 else 0.058 * ((rank + 1) // 2) * (1 if rank % 2 else -1)
        mid_x = (x0 + x1) / 2 - uy * bow
        mid_y = (y0 + y1) / 2 + ux * bow

        gap = min(0.065, length * 0.22)
        sx, sy = x0 + ux * gap, y0 + uy * gap
        ex, ey = x1 - ux * gap, y1 - uy * gap

        # Edge weight communicates confidence at a glance: a confident, cue-backed edge is
        # drawn bolder and more opaque; a weak co-occurrence fallback fades into the
        # background and is dashed, so the eye is drawn to the reliable structure first.
        is_weak = not t.get("cue")
        confidence = float(t.get("confidence", 0.5))
        edge_opacity = max(0.35, min(0.95, 0.35 + confidence * 0.6))
        edge_width = 1.1 + confidence * 2.2
        edge_colour = _translucent(theme.edge, edge_opacity) if theme.edge.startswith("#") else theme.edge

        # Spline curved edge line
        fig.add_trace(go.Scatter(
            x=[sx, mid_x, ex], y=[sy, mid_y, ey], mode="lines",
            hoverinfo="skip", showlegend=False,
            line=dict(
                width=edge_width, color=edge_colour, shape="spline", smoothing=0.88,
                dash="dot" if is_weak else "solid"
            )
        ))

        # Directional arrowhead pointing toward object node
        fig.add_annotation(
            x=ex, y=ey, ax=mid_x, ay=mid_y, xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1.1 + confidence * 0.3, arrowwidth=max(1.2, edge_width * 0.75),
            arrowcolor=edge_colour, text=""
        )

        lbl_x.append(mid_x)
        lbl_y.append(mid_y)
        lbl_text.append(t["predicate"] if show_predicates else "")
        lbl_hover.append(
            f"<b>Relationship:</b> {t['predicate']}<br>"
            f"<b>Assertion:</b> {t['subject']} &rarr; {t['object']}<br>"
            f"<b>Confidence:</b> {t['confidence']:.2f}<br>"
            f"<b>Lexical Cue:</b> {t['cue'] or 'Co-occurrence fallback'}<br>"
            f"<b>Sentence:</b> <i>{t['sentence']}</i>"
        )

    # Edge predicate labels, each on a soft rounded "pill" background so the text stays
    # legible where it crosses another edge or sits close to a node.
    if lbl_x:
        if show_predicates:
            fig.add_trace(go.Scatter(
                x=lbl_x, y=lbl_y, mode="markers", showlegend=False, hoverinfo="skip",
                marker=dict(
                    size=[min(120, 13 + 5.5 * len(t)) for t in lbl_text],
                    color=theme.card_bg, symbol="square",
                    line=dict(width=1, color=theme.card_border)
                )
            ))
        fig.add_trace(go.Scatter(
            x=lbl_x, y=lbl_y, mode="text" if show_predicates else "markers",
            text=lbl_text,
            textfont=dict(
                size=10,
                color=theme.ink,
                family="ui-sans-serif, system-ui, -apple-system, sans-serif"
            ),
            marker=dict(size=7, color="rgba(0,0,0,0)"),
            customdata=lbl_hover,
            hovertemplate="%{customdata}<extra></extra>",
            showlegend=False
        ))

    # Node markers grouped by entity type (gives clear legend items)
    for ent_type in ENTITY_TYPE_ORDER:
        members = [n for n in names if labels[n] == ent_type]
        if not members:
            continue
        indices = [index_of[n] for n in members]

        fig.add_trace(go.Scatter(
            x=[pos[i][0] for i in indices],
            y=[pos[i][1] for i in indices],
            mode="markers+text",
            name=ent_type,
            text=[(n if len(n) <= 20 else n[:18] + "...") for n in members],
            textposition=[label_at[i] for i in indices],
            textfont=dict(
                size=11,
                color=theme.ink,
                family="ui-sans-serif, system-ui, sans-serif"
            ),
            marker=dict(
                size=[min(34, 16 + 2.5 * degree.get(n, 1)) for n in members],
                color=theme.for_type(ent_type),
                symbol=TYPE_SYMBOL.get(ent_type, "circle"),
                line=dict(width=2.5, color=theme.halo)
            ),
            customdata=[
                f"<b>Entity:</b> {n}<br>"
                f"<b>Type:</b> {ent_type}<br>"
                f"<b>Connected Relationships:</b> {degree.get(n, 0)}"
                for n in members
            ],
            hovertemplate="%{customdata}<extra></extra>"
        ))

    margin_x = float(max(_label_boxes(names)[0])) + 0.08
    margin_y = 0.12
    fig.update_xaxes(visible=False, range=[float(pos[:, 0].min()) - margin_x, float(pos[:, 0].max()) + margin_x])
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1,
                     range=[float(pos[:, 1].min()) - margin_y, float(pos[:, 1].max()) + margin_y])

    height = int(min(760, max(460, 320 + 20 * len(names))))
    n_edges = sum(1 for _, (s, e) in zip(filtered_triples, edges) if s != e)
    fig.update_layout(
        height=height,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=11, color=theme.muted),
            bgcolor="rgba(0,0,0,0)"
        ),
        margin=dict(l=15, r=15, t=40, b=15),
        paper_bgcolor=theme.paper_bg,
        plot_bgcolor=theme.card_bg,
        font=dict(color=theme.ink, size=12),
        hoverlabel=dict(font_size=12, bgcolor=theme.card_bg, font_color=theme.ink)
    )
    fig.add_annotation(
        text=f"{len(names)} entities  &middot;  {n_edges} relationships",
        xref="paper", yref="paper", x=1, y=1.02, xanchor="right", yanchor="bottom",
        showarrow=False, font=dict(size=10.5, color=theme.muted)
    )
    return fig


def _build_flow_figure(triples: List[Dict[str, Any]], theme: Theme, limit: int = 12) -> go.Figure:
    """Sankey flow diagram: Subject Entity Type -> Relational Predicate -> Object Entity Type."""
    if not triples:
        fig = go.Figure()
        fig.add_annotation(text="No relationships available for flow visualization.",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                           font=dict(color=theme.muted, size=13))
        fig.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        return fig

    counts: Dict[Tuple[str, str, str], int] = {}
    for t in triples:
        key = (t["subject_label"], t["predicate"], t["object_label"])
        counts[key] = counts.get(key, 0) + 1

    ranked = sorted(counts.items(), key=lambda item: -item[1])[:limit]

    left = sorted({k[0] for k, _ in ranked}, key=lambda l: TYPE_SLOT.get(l, 9))
    middle = sorted({k[1] for k, _ in ranked})
    right = sorted({k[2] for k, _ in ranked}, key=lambda l: TYPE_SLOT.get(l, 9))

    node_labels = (
        [f"{l} (subj)" for l in left] +
        middle +
        [f"{l} (obj)" for l in right]
    )
    node_colors = (
        [theme.for_type(l) for l in left] +
        [theme.muted for _ in middle] +
        [theme.for_type(l) for l in right]
    )

    left_idx = {l: i for i, l in enumerate(left)}
    mid_idx = {l: len(left) + i for i, l in enumerate(middle)}
    right_idx = {l: len(left) + len(middle) + i for i, l in enumerate(right)}

    sources, targets, values, link_colors = [], [], [], []

    def hex_to_rgba(hex_str: str, alpha: float = 0.35) -> str:
        hex_str = hex_str.lstrip("#")
        if len(hex_str) == 6:
            r, g, b = int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16)
            return f"rgba({r},{g},{b},{alpha})"
        return f"rgba(100,116,139,{alpha})"

    for (s_type, pred, o_type), c in ranked:
        # Subject -> Predicate
        sources.append(left_idx[s_type])
        targets.append(mid_idx[pred])
        values.append(c)
        link_colors.append(hex_to_rgba(theme.for_type(s_type), 0.38))

        # Predicate -> Object
        sources.append(mid_idx[pred])
        targets.append(right_idx[o_type])
        values.append(c)
        link_colors.append(hex_to_rgba(theme.for_type(o_type), 0.38))

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=16,
            thickness=20,
            line=dict(color=theme.halo, width=1.5),
            label=node_labels,
            color=node_colors
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors
        )
    )])
    fig.update_layout(
        height=380,
        margin=dict(l=15, r=15, t=25, b=15),
        paper_bgcolor=theme.paper_bg,
        plot_bgcolor=theme.card_bg,
        font=dict(color=theme.ink, size=11),
        hoverlabel=dict(bgcolor=theme.card_bg, font_color=theme.ink, font_size=11)
    )
    return fig


def _build_entity_type_chart(entities: List[Dict[str, Any]], theme: Theme) -> go.Figure:
    """Distribution of detected entities across types."""
    counts: Dict[str, int] = {}
    for e in entities:
        counts[e["label"]] = counts.get(e["label"], 0) + 1

    fig = go.Figure(data=[go.Bar(
        x=list(counts.keys()),
        y=list(counts.values()),
        marker=dict(
            color=[theme.for_type(t) for t in counts.keys()],
            line=dict(color=theme.halo, width=1.5)
        ),
        text=list(counts.values()),
        textposition="auto",
        hovertemplate="<b>%{x}</b>: %{y} mentions<extra></extra>"
    )])
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=25, b=25),
        paper_bgcolor=theme.paper_bg,
        plot_bgcolor=theme.card_bg,
        font=dict(color=theme.ink, size=11),
        xaxis=dict(gridcolor=theme.grid),
        yaxis=dict(gridcolor=theme.grid, title="Mentions"),
        bargap=0.35,
        hoverlabel=dict(bgcolor=theme.card_bg, font_color=theme.ink)
    )
    return fig


def _translucent(hex_colour: str, alpha: float = 0.18) -> str:
    """A hex colour converted to a low-opacity rgba() string, for shaded diagram boxes."""
    hex_colour = hex_colour.lstrip("#")
    r, g, b = (int(hex_colour[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha:.2f})"


def _build_pipeline_diagram(theme: Theme) -> go.Figure:
    """A left-to-right flowchart of the extraction pipeline's six processing stages."""
    stages = [
        ("Source Text", "Raw document or sentence"),
        ("Sentence Splitting", "Split on . ! ?"),
        ("Entity Recognition", "Capitalised-span NER"),
        ("Candidate Pairs", "Adjacent entities per sentence"),
        ("Cue Matching", "Ordered regex lexicon lookup"),
        ("SPO Triple", "Subject, predicate, object, confidence"),
    ]
    colour_cycle = ["PERSON", "ORG", "LOCATION", "DATE", "TERM", "MISC"]
    box_w, gap = 1.0, 0.32
    n = len(stages)

    fig = go.Figure()
    for i, (title, sub) in enumerate(stages):
        x0 = i * (box_w + gap)
        x1 = x0 + box_w
        colour = theme.for_type(colour_cycle[i % len(colour_cycle)])
        fig.add_shape(
            type="rect", x0=x0, y0=0, x1=x1, y1=1,
            line=dict(color=colour, width=1.75),
            fillcolor=_translucent(colour, 0.16)
        )
        fig.add_annotation(x=(x0 + x1) / 2, y=0.64, text=f"<b>{title}</b>", showarrow=False,
                           font=dict(size=12, color=theme.ink), align="center")
        fig.add_annotation(x=(x0 + x1) / 2, y=0.28, text=sub, showarrow=False,
                           font=dict(size=9.5, color=theme.muted), align="center")
        if i < n - 1:
            fig.add_annotation(
                x=x0 + box_w + gap, y=0.5, ax=x1, ay=0.5,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=2, arrowsize=1.1, arrowwidth=1.6, arrowcolor=theme.edge
            )

    fig.update_xaxes(visible=False, range=[-0.15, n * (box_w + gap)], fixedrange=True)
    fig.update_yaxes(visible=False, range=[-0.1, 1.1], fixedrange=True)
    fig.update_layout(
        height=180, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor=theme.paper_bg, plot_bgcolor=theme.card_bg
    )
    return fig


# Methods compared side-by-side in the Theory section's "Method Comparison" chart.
_COMPARISON_METHODS = [
    "Pattern-Based (Lexicon Match)",
    "Co-occurrence Only",
    "Hybrid (Pattern + Co-occurrence)"
]


def _compute_method_comparison(corpus_name: str = "Tech & Business") -> List[Dict[str, Any]]:
    """Runs all three extraction methods on one benchmark corpus and scores each against
    its ground-truth reference set, for a live, authentic method comparison (not
    illustrative/fabricated numbers)."""
    corpus = CORPORA[corpus_name]
    entities = extract_entities(corpus["text"])
    rows = []
    for method in _COMPARISON_METHODS:
        triples = extract_relationships(text=corpus["text"], entities=entities, method=method)
        metrics = evaluate_against_reference_set(triples, corpus["gold"])
        rows.append({
            "Method": method,
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1": metrics["f1"],
            "Triples Extracted": len(triples)
        })
    return rows


def _build_method_comparison_figure(theme: Theme, corpus_name: str = "Tech & Business") -> go.Figure:
    """Grouped bar chart: Precision / Recall / F1 for each extraction method, computed
    live against the named benchmark corpus's gold reference set."""
    rows = _compute_method_comparison(corpus_name)
    names = [r["Method"].split(" (")[0] for r in rows]

    fig = go.Figure()
    for metric, colour_key in (("Precision", "PERSON"), ("Recall", "ORG"), ("F1", "LOCATION")):
        values = [r[metric] for r in rows]
        fig.add_trace(go.Bar(
            name=metric, x=names, y=values,
            marker=dict(color=theme.for_type(colour_key)),
            text=[f"{v:.2f}" for v in values], textposition="outside",
            hovertemplate=f"<b>%{{x}}</b><br>{metric}: %{{y:.2f}}<extra></extra>"
        ))
    fig.update_layout(
        barmode="group", height=340,
        margin=dict(l=20, r=20, t=30, b=25),
        paper_bgcolor=theme.paper_bg, plot_bgcolor=theme.card_bg,
        font=dict(color=theme.ink, size=11),
        yaxis=dict(gridcolor=theme.grid, title="Score", range=[0, 1.08]),
        xaxis=dict(gridcolor=theme.grid),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        bargap=0.28, bargroupgap=0.08,
        hoverlabel=dict(bgcolor=theme.card_bg, font_color=theme.ink)
    )
    return fig


def _annotate_sentence(sentence: str, entities_in_sentence: List[Dict[str, Any]], cues: Sequence[str]) -> str:
    """Formats sentence prose with color-coded entity badges and bold relation cues."""
    sorted_ents = sorted(entities_in_sentence, key=lambda e: e["start"])
    fragments = []
    cursor = 0
    for e in sorted_ents:
        if e["start"] > cursor:
            span = sentence[cursor:e["start"]]
            fragments.append(_mark_cues(span, cues))
        highlight = TYPE_HIGHLIGHT.get(e["label"], "gray")
        fragments.append(f":{highlight}-background[**{e['text']}** ({e['label']})]")
        cursor = e["end"]
    if cursor < len(sentence):
        fragments.append(_mark_cues(sentence[cursor:], cues))
    return "".join(fragments)


def _mark_cues(fragment: str, cues: Sequence[str]) -> str:
    for c in cues:
        if not c:
            continue
        fragment = re.sub(
            re.escape(c),
            f"<b style='text-decoration:underline; font-weight:700;'>{c}</b>",
            fragment,
            flags=re.IGNORECASE
        )
    return fragment


# ======================================================================================
# 9. PDF LABORATORY REPORT GENERATOR (FPDF)
# ======================================================================================

class LabReportPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(100, 116, 139)
        self.cell(0, 5, "VIRTUAL LABORATORY PORTAL | KNOWLEDGE GRAPHS & INFORMATION RETRIEVAL SYSTEMS", align="R")
        self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(148, 163, 184)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}} | Academic Virtual Lab Report | Exp-07 Relationship Extraction", align="C")


# A condensed, plain-text (no markdown/[[term]] markers) version of the Theory section's
# background, for the PDF report's own "Theoretical Background" section.
PDF_THEORY_SUMMARY = (
    "Relationship Extraction (RE) identifies and classifies semantic relationships holding "
    "between pairs of entities mentioned in unstructured text, expressing each as a "
    "Subject-Predicate-Object (SPO) triple - the atomic unit from which a Knowledge Graph is "
    "built: entities become nodes, and predicates become directed, labelled edges between them. "
    "This laboratory implements and compares three rule-based extraction strategies, none of "
    "which depend on an external NLP model. Pattern-Based extraction emits a triple only when "
    "the text span between two adjacent entities matches an ordered lexicon of relation-cue "
    "regular expressions (e.g. 'acquired', 'is the CEO of'), giving high precision at the cost "
    "of recall. Co-occurrence extraction links every adjacent entity pair with a generic "
    "'related_to' predicate whose confidence decays with word distance, giving high recall at "
    "the cost of precision. Hybrid extraction applies the Pattern-Based check first and falls "
    "back to a Co-occurrence edge only when no cue matches, so no candidate pair is ever "
    "dropped outright. Every emitted triple carries a numeric confidence score and its source "
    "sentence as provenance, so downstream stages (Experiments 8-9: Neo4j graph construction "
    "and schema import) can filter or audit it before ingestion."
)

# (Method, Precision, Recall, Characteristic) - the same qualitative comparison shown in the
# Theory tab's method-comparison table, condensed for the PDF.
PDF_METHOD_COMPARISON_ROWS = [
    ("Pattern-Based", "High", "Lower", "Every edge traces to an explicit cue word in the text"),
    ("Co-occurrence", "Low", "Highest", "Confidence decays with word distance; no cue required"),
    ("Hybrid", "Medium-High", "High", "Pattern-Based first, Co-occurrence fallback if no cue matches"),
]


def generate_pdf_report(
    student_name: str,
    student_id: str,
    student_batch: str,
    date_str: str,
    trials_df: pd.DataFrame,
    quiz_score: int,
    quiz_total: int,
    student_notes: str,
    triples_df: pd.DataFrame,
    eval_metrics: Optional[Dict[str, Any]] = None
) -> bytes:
    """Generates an official institutional laboratory report PDF."""
    pdf = LabReportPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    # Title Banner
    pdf.set_text_color(15, 23, 42)
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, EXPERIMENT_CONFIG["title"], align="L", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(0, 5, f"{EXPERIMENT_CONFIG['course']} | {EXPERIMENT_CONFIG['department']}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # Student Credentials Card
    pdf.set_fill_color(248, 250, 252)
    pdf.set_draw_color(203, 213, 225)
    pdf.rect(10, 28, 190, 24, "FD")

    pdf.set_xy(14, 30)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(30, 5, "Student Name:", 0)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(60, 5, student_name or "N/A", 0)

    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(35, 5, "Roll No / Student ID:", 0)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(55, 5, student_id or "N/A", 1)

    pdf.set_xy(14, 38)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(30, 5, "Batch / Class:", 0)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(60, 5, student_batch or "D17", 0)

    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(35, 5, "Quiz Assessment:", 0)
    pdf.set_font("Helvetica", "B", 8)
    pct = int((quiz_score / quiz_total) * 100) if quiz_total else 0
    if pct >= 50:
        pdf.set_text_color(16, 185, 129)
    else:
        pdf.set_text_color(239, 68, 68)
    pdf.cell(55, 5, f"{quiz_score} / {quiz_total} ({pct}%)", 1)

    pdf.set_xy(14, 45)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(30, 5, "Experiment Date:", 0)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(60, 5, date_str or datetime.now().strftime("%Y-%m-%d"), 0)

    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(35, 5, "Lab Status:", 0)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(37, 99, 235)
    pdf.cell(55, 5, "Verified & Evaluated", 1)

    pdf.set_xy(10, 56)

    # 1. Objectives
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 6, "1. Educational Aim & Learning Objectives", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(51, 65, 85)
    for obj in EXPERIMENT_CONFIG["objectives"]:
        pdf.cell(4, 4, "-", 0)
        pdf.cell(0, 4, f" {obj}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # 2. Theoretical Background
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 6, "2. Theoretical Background", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(51, 65, 85)
    pdf.multi_cell(0, 4.3, PDF_THEORY_SUMMARY)
    pdf.ln(1.5)

    pdf.set_font("Helvetica", "B", 7.5)
    pdf.set_text_color(255, 255, 255)
    pdf.set_fill_color(30, 41, 59)
    pdf.cell(45, 5, "Method", 1, 0, "L", True)
    pdf.cell(35, 5, "Precision", 1, 0, "C", True)
    pdf.cell(35, 5, "Recall", 1, 0, "C", True)
    pdf.cell(75, 5, "Characteristic", 1, 1, "L", True)
    pdf.set_font("Helvetica", "", 7.5)
    pdf.set_text_color(30, 41, 59)
    fill = False
    for method_name, precision, recall, characteristic in PDF_METHOD_COMPARISON_ROWS:
        pdf.set_fill_color(248, 250, 252) if fill else pdf.set_fill_color(255, 255, 255)
        pdf.cell(45, 4.5, method_name, 1, 0, "L", fill)
        pdf.cell(35, 4.5, precision, 1, 0, "C", fill)
        pdf.cell(35, 4.5, recall, 1, 0, "C", fill)
        pdf.cell(75, 4.5, characteristic, 1, 1, "L", fill)
        fill = not fill
    pdf.ln(3)

    # 3. Evaluation Summary (if available)
    if eval_metrics and eval_metrics.get("has_gold"):
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(30, 58, 138)
        pdf.cell(0, 6, "3. Ground Truth Benchmark Evaluation", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(51, 65, 85)
        pdf.cell(45, 5, f"Precision: {eval_metrics['precision']:.3f}", 1, 0, "C")
        pdf.cell(45, 5, f"Recall: {eval_metrics['recall']:.3f}", 1, 0, "C")
        pdf.cell(45, 5, f"F1 Score: {eval_metrics['f1']:.3f}", 1, 0, "C")
        pdf.cell(55, 5, f"True Positives: {eval_metrics['tp_count']} / {eval_metrics['gold_count']}", 1, 1, "C")
        pdf.ln(3)

    # 4. Recorded Experimental Trials
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 6, "4. Recorded Experimental Trials (Session Logbook)", new_x="LMARGIN", new_y="NEXT")

    if trials_df.empty:
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(0, 5, "No experimental trials recorded during this session.", new_x="LMARGIN", new_y="NEXT")
    else:
        pdf.set_fill_color(37, 99, 235)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 7)
        cols = list(trials_df.columns)
        col_w = max(18, int(190 / max(1, len(cols))))
        for c in cols:
            pdf.cell(col_w, 5, str(c)[:14], 1, 0, "C", True)
        pdf.ln()

        pdf.set_fill_color(248, 250, 252)
        pdf.set_text_color(30, 41, 59)
        pdf.set_font("Helvetica", "", 7)
        fill = False
        for _, row in trials_df.iterrows():
            for c in cols:
                val = row[c]
                val_str = f"{val:.2f}" if isinstance(val, float) else str(val)
                pdf.cell(col_w, 4.5, val_str[:15], 1, 0, "C", fill)
            pdf.ln()
            fill = not fill
    pdf.ln(3)

    # 4. Extracted Triples Sample
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 6, "5. Extracted Subject-Predicate-Object Knowledge Triples", new_x="LMARGIN", new_y="NEXT")

    if triples_df.empty:
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(0, 5, "No triples extracted in the active session.", new_x="LMARGIN", new_y="NEXT")
    else:
        pdf.set_fill_color(30, 41, 59)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 7)
        pdf.cell(50, 5, "Subject", 1, 0, "L", True)
        pdf.cell(45, 5, "Predicate", 1, 0, "L", True)
        pdf.cell(50, 5, "Object", 1, 0, "L", True)
        pdf.cell(20, 5, "Conf.", 1, 0, "C", True)
        pdf.cell(25, 5, "Method", 1, 1, "C", True)

        pdf.set_font("Helvetica", "", 7)
        pdf.set_text_color(30, 41, 59)
        fill = False
        for _, row in triples_df.head(15).iterrows():
            # A bare ternary-as-statement here (not assigned to anything) is picked up by
            # Streamlit's "magic" auto-display and rendered as a stray `None` in the app,
            # since streamlit run auto-wraps unassigned top-level expressions in st.write().
            if fill:
                pdf.set_fill_color(248, 250, 252)
            else:
                pdf.set_fill_color(255, 255, 255)
            pdf.cell(50, 4.5, str(row.get("subject", ""))[:26], 1, 0, "L", fill)
            pdf.cell(45, 4.5, str(row.get("predicate", ""))[:24], 1, 0, "L", fill)
            pdf.cell(50, 4.5, str(row.get("object", ""))[:26], 1, 0, "L", fill)
            pdf.cell(20, 4.5, f"{float(row.get('confidence', 0)):.2f}", 1, 0, "C", fill)
            pdf.cell(25, 4.5, str(row.get("method", "Pattern"))[:12], 1, 1, "C", fill)
            fill = not fill
    pdf.ln(3)

    # 5. Student Conclusions & Observations
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 6, "6. Student Analytical Discussion & Conclusions", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(51, 65, 85)
    pdf.multi_cell(0, 4.5, student_notes or "The student did not record analytical notes.")
    pdf.ln(4)

    # Signature Block
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(95, 6, "Student Signature: _______________________", 0, 0, "L")
    pdf.cell(95, 6, "Faculty Evaluator Signature: _______________________", 0, 1, "R")

    return bytes(pdf.output())


# ======================================================================================
# 10. ASSESSMENT QUIZ BANK (10 PEDAGOGICAL QUESTIONS)
# ======================================================================================

QUIZ_QUESTIONS = [
    {
        "id": 1,
        "question": "What are the three canonical components of a knowledge triple extracted from unstructured text?",
        "options": [
            "A) Noun, Adjective, Verb",
            "B) Subject, Predicate, Object",
            "C) Keyword, Frequency, Weight",
            "D) Document, Term, Score"
        ],
        "answer_index": 1,
        "explanation": "Subject-Predicate-Object (SPO) triples form the atomic data assertions in Knowledge Graphs and Semantic Web models."
    },
    {
        "id": 2,
        "question": "In a Knowledge Graph construction pipeline, what do the extracted entities and predicates represent?",
        "options": [
            "A) Entities become nodes, and predicates become directed labeled edges",
            "B) Entities become edge labels, and predicates become nodes",
            "C) Both entities and predicates become database tables without relationships",
            "D) Entities become vector indexes and predicates are discarded"
        ],
        "answer_index": 0,
        "explanation": "Entities (e.g. Google, California) represent network nodes, while predicates (e.g. founded, located_in) form directed edges connecting them."
    },
    {
        "id": 3,
        "question": "Why does co-occurrence based extraction typically exhibit high recall but low precision?",
        "options": [
            "A) It only considers capitalized words",
            "B) It assumes any two entities co-occurring in a sentence are related, over-generating generic links without lexical proof",
            "C) It strictly enforces grammatical dependency rules",
            "D) It drops all pairs with distance greater than 2 words"
        ],
        "answer_index": 1,
        "explanation": "Co-occurrence captures almost all true pairs (high recall) but generates many spurious or meaningless relationships (low precision)."
    },
    {
        "id": 4,
        "question": "What is the primary vulnerability (limitation) of hand-crafted pattern-based relation extraction?",
        "options": [
            "A) Linguistic brittleness when encountering paraphrases, metaphors, or complex syntactic inversions",
            "B) High computational overhead compared to neural language models",
            "C) Inability to run on CPU without GPUs",
            "D) Extreme over-generation of false positive edges"
        ],
        "answer_index": 0,
        "explanation": "Pattern rules are brittle: they reliably catch anticipated phrasings ('acquired'), but fail on unexpected paraphrases ('swallowed up')."
    },
    {
        "id": 5,
        "question": "What does Open Information Extraction (OpenIE) NOT require, unlike traditional supervised relation extraction?",
        "options": [
            "A) Input text",
            "B) A fixed, predefined schema of relation types",
            "C) Sentences",
            "D) Words"
        ],
        "answer_index": 1,
        "explanation": "OpenIE extracts arbitrary relation phrases directly from text without needing a rigid predefined ontology or relation taxonomy."
    },
    {
        "id": 6,
        "question": "What NLP pre-processing stage resolves pronouns like 'she' or 'the company' back to their antecedent entity mention?",
        "options": [
            "A) Lemmatization",
            "B) Co-reference resolution",
            "C) Stop-word pruning",
            "D) Part-of-speech tagging"
        ],
        "answer_index": 1,
        "explanation": "Co-reference resolution links pronouns ('she') to their true referent (e.g. 'Marie Curie'), enabling facts spread across sentences to be extracted."
    },
    {
        "id": 7,
        "question": "In the triple (Google) --[acquired]--> (YouTube), which element is the predicate?",
        "options": [
            "A) Google",
            "B) YouTube",
            "C) acquired",
            "D) Confidence score"
        ],
        "answer_index": 2,
        "explanation": "'acquired' is the relational predicate characterizing the semantic connection between the subject and object."
    },
    {
        "id": 8,
        "question": "Why is attaching a quantitative confidence score to each extracted triple beneficial for downstream graph ingestion?",
        "options": [
            "A) It allows downstream graph databases (e.g. Neo4j) to filter out low-confidence noise and prioritize verified facts",
            "B) It eliminates the need for entity normalization",
            "C) It automatically translates English into Cypher syntax",
            "D) It guarantees 100% precision regardless of input text quality"
        ],
        "answer_index": 0,
        "explanation": "Confidence scores empower downstream ingestion pipelines to discard unreliable inferences and weight graph algorithms."
    },
    {
        "id": 9,
        "question": "Which evaluation metric represents the harmonic mean of Precision and Recall?",
        "options": [
            "A) Accuracy",
            "B) F1 Score",
            "C) Mean Reciprocal Rank (MRR)",
            "D) Discounted Cumulative Gain (DCG)"
        ],
        "answer_index": 1,
        "explanation": "The F1 score is the harmonic mean of precision and recall: 2 * (P * R) / (P + R)."
    },
    {
        "id": 10,
        "question": "In Neo4j graph construction, why is the MERGE clause preferred over CREATE when importing extracted triples?",
        "options": [
            "A) MERGE is idempotent: it avoids creating duplicate nodes when an entity appears in multiple triples",
            "B) CREATE is deprecated in modern Cypher",
            "C) MERGE automatically executes relationship extraction in memory",
            "D) MERGE deletes existing nodes before adding new ones"
        ],
        "answer_index": 0,
        "explanation": "MERGE creates a node or relationship only if it does not already exist, ensuring identical entities consolidate into a connected network."
    }
]


# ======================================================================================
# 11. SECTION RENDERERS: THEORY, SIMULATION, QUIZ, REPORT
# ======================================================================================

def render_theory_section():
    """Renders Section 1: Theory, Background, Objectives, Procedure, and Glossary."""
    theme = get_current_theme()

    st.header("Theoretical Framework & Background")
    write_annotated(THEORY_CONTENT["background"])

    st.subheader("Visual Overview: The Extraction Pipeline")
    st.caption(
        "Every extraction run, regardless of method, passes through these six stages in order. "
        "Colours match the entity-type legend used throughout the Simulation section."
    )
    st.plotly_chart(_build_pipeline_diagram(theme), use_container_width=True, config={"displayModeBar": False})

    st.divider()
    st.subheader("Algorithmic Detail: How Each Extraction Method Works")
    st.write(
        "All three methods share the same first three pipeline stages (splitting, entity "
        "recognition, candidate-pair generation) and differ only in how a candidate pair is "
        "turned into a predicate. Each tab below shows the exact procedure as pseudocode, "
        "followed by that method's *live* output on the same worked example sentence."
    )

    _example_text = "Larry Page co-founded Google in California in 1998."
    _example_entities = extract_entities(_example_text)

    tab_pattern, tab_cooc, tab_hybrid = st.tabs([
        "Pattern-Based (Lexicon Match)", "Co-occurrence Only", "Hybrid (Pattern + Co-occurrence)"
    ])

    with tab_pattern:
        write_annotated(
            "Emits a triple **only** when the text span between two adjacent entities matches "
            "one of the ordered [[relation_cue|relation-cue]] regular expressions "
            "(`_RELATION_PATTERNS`). Patterns are tried most-specific-first and the **first "
            "match wins** - so `co-founded` is caught by the `founded` family before any looser "
            "pattern gets a chance. If nothing matches, the pair is silently dropped: this is "
            "why the method is high-[[precision]] but limited in [[recall]]."
        )
        st.code(
            "for sentence in split_sentences(document):\n"
            "    entities = extract_entities(sentence)                 # NER\n"
            "    for (subject, object) in adjacent_pairs(entities):    # skip if entity in between\n"
            "        span = text_between(subject, object)\n"
            "        if word_count(span) > max_word_distance:\n"
            "            continue                                      # too far apart, skip pair\n"
            "        for (regex, predicate, base_confidence) in RELATION_PATTERNS:  # ordered\n"
            "            if regex.search(span):\n"
            "                confidence = apply_type_constraints(predicate, subject, object, base_confidence)\n"
            "                emit Triple(subject, predicate, object, confidence, cue=regex.match_text)\n"
            "                break                                     # first match wins, stop looking\n"
            "        # else: no cue found -> pair produces NO triple in Pattern-Based mode",
            language="python"
        )
        st.caption(
            "`apply_type_constraints` softens confidence when the entity types look wrong for "
            "the predicate - e.g. `born_in` is discounted (x0.6) unless the subject is a PERSON, "
            "and `leads` is discounted (x0.75) unless the object is an ORG or LOCATION."
        )
        pattern_triples = extract_relationships(text=_example_text, entities=_example_entities,
                                                method="Pattern-Based (Lexicon Match)")
        st.write(f"**Live trace** for: *\"{_example_text}\"*")
        if pattern_triples:
            st.dataframe(pd.DataFrame(pattern_triples)[["subject", "predicate", "object", "confidence", "cue"]],
                        hide_index=True, use_container_width=True)
        else:
            st.info("No cue matched in this example, so Pattern-Based mode emits no triple here.")

    with tab_cooc:
        write_annotated(
            "Makes no attempt to read the span at all: **every** adjacent entity pair in a "
            "sentence is linked with the generic predicate `related_to`. This maximises "
            "[[recall]] (nothing is ever silently dropped) at the cost of [[precision]] (most "
            "edges carry no real semantic content). Confidence is not fixed - it **decays with "
            "distance**, so two entities separated by many words score lower than two sitting "
            "right next to each other."
        )
        st.code(
            "for sentence in split_sentences(document):\n"
            "    entities = extract_entities(sentence)\n"
            "    for (subject, object) in adjacent_pairs(entities):\n"
            "        if word_count(span) > max_word_distance:\n"
            "            continue\n"
            "        confidence = max(0.20, 0.45 - 0.02 * word_count(span))   # decays with distance\n"
            "        emit Triple(subject, 'related_to', object, confidence, cue=None)",
            language="python"
        )
        cooc_triples = extract_relationships(text=_example_text, entities=_example_entities,
                                             method="Co-occurrence Only")
        st.write(f"**Live trace** for: *\"{_example_text}\"*")
        if cooc_triples:
            st.dataframe(pd.DataFrame(cooc_triples)[["subject", "predicate", "object", "confidence"]],
                        hide_index=True, use_container_width=True)
        else:
            st.info("No adjacent entity pairs were found in this example.")

    with tab_hybrid:
        write_annotated(
            "Runs the **Pattern-Based** check first on every candidate pair; only when no cue "
            "matches does it fall back to a **Co-occurrence** edge instead of dropping the pair. "
            "The result is a superset of what either method produces alone: the confident, "
            "correctly-labelled edges from Pattern-Based, topped up with low-confidence "
            "`related_to` edges so no relationship is lost entirely. This is the [[hybrid]] "
            "default used throughout the Simulation section."
        )
        st.code(
            "for (subject, object) in adjacent_pairs(entities):\n"
            "    if word_count(span) > max_word_distance:\n"
            "        continue\n"
            "    match = first_matching_pattern(span)          # same ordered lexicon as Pattern-Based\n"
            "    if match:\n"
            "        emit Triple(subject, match.predicate, object, match.confidence, cue=match.text)\n"
            "    else:\n"
            "        confidence = max(0.20, 0.45 - 0.02 * word_count(span))\n"
            "        emit Triple(subject, 'related_to', object, confidence, cue=None)  # fallback",
            language="python"
        )
        hybrid_triples = extract_relationships(text=_example_text, entities=_example_entities,
                                               method="Hybrid (Pattern + Co-occurrence)")
        st.write(f"**Live trace** for: *\"{_example_text}\"*")
        if hybrid_triples:
            st.dataframe(pd.DataFrame(hybrid_triples)[["subject", "predicate", "object", "confidence", "cue"]],
                        hide_index=True, use_container_width=True)
        else:
            st.info("No adjacent entity pairs were found in this example.")

    st.divider()
    st.subheader("Illustrated Example: From Raw Sentences to a Knowledge Graph")
    _illustration_text = (
        "Larry Page and Sergey Brin co-founded Google in California in 1998. "
        "Google acquired YouTube in 2006."
    )
    _illustration_entities = extract_entities(_illustration_text)
    _illustration_triples = extract_relationships(
        text=_illustration_text, entities=_illustration_entities,
        method="Hybrid (Pattern + Co-occurrence)"
    )
    st.write("**Step 1 - Annotated source text** (entities highlighted by type, relation cues underlined):")
    _illustration_sentences = _split_sentences(_illustration_text)
    _illustration_cues = [t["cue"] for t in _illustration_triples if t.get("cue")]
    for s_idx, s_text in enumerate(_illustration_sentences):
        s_ents = [e for e in _illustration_entities if e["sentence_idx"] == s_idx]
        st.markdown(_annotate_sentence(s_text, s_ents, _illustration_cues), unsafe_allow_html=True)
    st.write("**Step 2 - Resulting knowledge graph** (same layout engine used in the Simulation section):")
    st.plotly_chart(_build_graph_figure(_illustration_triples, theme, show_predicates=True),
                    use_container_width=True)

    st.divider()
    st.subheader("Comparing the Three Extraction Methods")
    write_annotated(
        "The qualitative trade-offs above are confirmed empirically below: all three methods "
        "run against the **Tech & Business** benchmark corpus and scored against its "
        "human-curated [[reference_set|gold reference set]]."
    )
    st.markdown(
        "| Method | Precision | Recall | Interpretability | Typical Confidence | Best suited to |\n"
        "|---|---|---|---|---|---|\n"
        "| **Pattern-Based** | High | Lower | Every edge traces to an explicit cue word | 0.80-0.95 "
        "(fixed per pattern) | Clean, well-worded documents where missing a few facts is "
        "acceptable but wrong facts are not |\n"
        "| **Co-occurrence** | Low | Highest | Trivial (proximity only) | 0.20-0.45 (decays with "
        "distance) | Early-stage exploration, or as a recall safety-net alongside a stricter "
        "method |\n"
        "| **Hybrid** | Medium-High | High | Mostly cue-based, with visibly weaker fallback edges | "
        "Mixed (0.20-0.95) | General-purpose default; lets a confidence threshold slider do the "
        "precision/recall trade-off instead of the extraction method itself |\n"
    )
    st.plotly_chart(_build_method_comparison_figure(theme), use_container_width=True)
    st.caption(
        "Read across the three bars per group: Precision and Recall consistently move in "
        "opposite directions between Pattern-Based and Co-occurrence - this is the classic "
        "precision/recall trade-off, and Hybrid's F1 (the harmonic mean of the two) typically "
        "lands between them or above both, since it never drops a candidate pair entirely."
    )

    st.divider()
    st.subheader("Educational Aim & Learning Objectives")
    for i, obj in enumerate(EXPERIMENT_CONFIG["objectives"]):
        st.markdown(f"- **Goal {i+1}**: {obj}")

    st.divider()
    st.subheader("Standard Experimental Procedure")
    for step in THEORY_CONTENT["procedure"]:
        st.markdown(f"- {step}")

    st.divider()
    render_glossary_expander()

    st.divider()
    with st.expander("Pipeline Integration: Connection to the Complete KGIRS Architecture", expanded=False):
        st.markdown("""
This laboratory shares an standardized `st.session_state` data contract to integrate seamlessly
across the entire KGIRS curriculum:

| Session State Key | Producer Module | Consumer Modules |
|---|---|---|
| `kg_source_text` | Experiment 1-2 (Preprocessing) or this page | Experiment 6, this page |
| `kg_entities` | Experiment 6 (Named Entity Recognition) | **This page (Exp 7)**, Exp 8, Exp 9 |
| `kg_triples` | **This page (Experiment 7: Relationship Extraction)** | Experiment 8 (Graph DB Ingestion), Exp 9 (Schema Design), Exp 10-11 (Cypher Querying) |

Downstream, Experiment 8 and 9 consume the extracted triples directly using idempotent Cypher:
```cypher
MERGE (s:Entity {name: $subject, type: $subject_label})
MERGE (o:Entity {name: $object, type: $object_label})
MERGE (s)-[r:RELATION {type: $predicate, confidence: $confidence}]->(o);
```
        """)


def render_simulation_section():
    """Renders Section 2: Interactive Sandbox, 6 Analytical Tabs, Evaluation, and Exports."""
    theme = get_current_theme()

    st.header("Interactive Relationship Extraction Sandbox")
    write_annotated(
        "Configure extraction parameters, run [[pattern_based|Pattern-Based]], "
        "[[cooccurrence|Co-occurrence]], or [[hybrid|Hybrid]] extraction, and evaluate the "
        "resulting [[triple|triples]] against a [[reference_set|ground truth]] where available."
    )

    # 1. Document Selection Card
    with st.container():
        doc_names = list(CORPORA.keys()) + ["Custom Input", "Upload Document File (.txt, .pdf, .docx, .csv)"]
        doc_choice = st.selectbox(
            "Select Source Document / Corpus",
            options=doc_names,
            help="Choose from benchmark pedagogical corpora with ground truth reference sets, or supply custom text."
        )

        selected_corpus = CORPORA.get(doc_choice)
        default_text = ""
        gold_triples = []

        if selected_corpus:
            default_text = selected_corpus["text"]
            gold_triples = selected_corpus.get("gold", [])
            col_b1, col_b2 = st.columns([1, 4])
            with col_b1:
                st.info(f"**Level:** {selected_corpus['level']}")
            with col_b2:
                st.caption(f"**Pedagogical Purpose:** {selected_corpus['purpose']}")
            with st.expander("Things to Observe in this Document", expanded=False):
                for obs in selected_corpus.get("observe", []):
                    st.markdown(f"- {obs}")
        elif doc_choice == "Upload Document File (.txt, .pdf, .docx, .csv)":
            uploaded_file = st.file_uploader(
                "Upload document",
                type=["txt", "md", "csv", "tsv", "pdf", "docx"],
                help="Upload a file up to 25,000 characters."
            )
            if uploaded_file:
                loaded = load_uploaded_document(uploaded_file)
                if loaded.error:
                    st.error(loaded.error)
                else:
                    default_text = loaded.text
                    st.success(loaded.summary())
        else:
            # Custom input
            upstream = st.session_state.get("kg_source_text", "")
            default_text = upstream if upstream else "Larry Page co-founded Google in California in 1998."

        text_input = st.text_area(
            "Document Text (editable for experimentation)",
            value=default_text,
            height=140,
            key="sandbox_text_area"
        )

    # 2. Parameters Card
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        method = st.radio(
            "Extraction Strategy",
            options=[
                "Hybrid (Pattern + Co-occurrence)",
                "Pattern-Based (Lexicon Match)",
                "Co-occurrence Only"
            ],
            help="Pattern-Based prioritizes precision; Co-occurrence maximizes recall; Hybrid balances both."
        )
    with col_p2:
        min_confidence = st.slider(
            "Minimum Confidence Cutoff",
            min_value=0.10,
            max_value=1.0,
            value=0.30,
            step=0.05,
            help="Filters out triples whose confidence falls below this threshold."
        )
    with col_p3:
        max_dist = st.slider(
            "Max Word Separation",
            min_value=4,
            max_value=25,
            value=12,
            step=1,
            help="Maximum words allowed between two entities in a sentence to form a candidate pair."
        )
        use_upstream_ner = st.checkbox(
            "Reuse upstream entities (Exp 6)",
            value=True,
            help="If Experiment 6 already executed in this session, reuse its recognized entities."
        )

    run_btn = st.button("Run Relationship Extraction Pipeline", type="primary", use_container_width=True)

    # Execution or cached state retrieval
    if run_btn or "exp7_last_triples" in st.session_state:
        if run_btn:
            upstream_ents = st.session_state.get("kg_entities")
            if use_upstream_ner and upstream_ents and st.session_state.get("kg_source_text") == text_input:
                entities = upstream_ents
            else:
                entities = extract_entities(text_input)

            triples = extract_relationships(
                text=text_input,
                entities=entities,
                method=method,
                max_word_distance=max_dist
            )
            filtered_triples = [t for t in triples if t["confidence"] >= min_confidence]

            st.session_state["exp7_last_entities"] = entities
            st.session_state["exp7_last_triples"] = filtered_triples
            st.session_state["exp7_last_text"] = text_input
            st.session_state["exp7_last_method"] = method
            st.session_state["exp7_last_corpus"] = doc_choice

            # Downstream pipeline session state contract
            st.session_state["kg_entities"] = entities
            st.session_state["kg_triples"] = filtered_triples
            st.session_state["kg_source_text"] = text_input

        entities = st.session_state.get("exp7_last_entities", [])
        triples = st.session_state.get("exp7_last_triples", [])
        metrics = compute_extraction_metrics(entities, triples)
        eval_result = evaluate_against_reference_set(triples, gold_triples)
        st.session_state["exp7_last_eval"] = eval_result

        st.divider()

        # Metrics Banner
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Entities Identified", metrics["entities"])
        m2.metric("Triples Extracted", metrics["triples"])
        m3.metric("High-Confidence", metrics["high_confidence_triples"])
        m4.metric("Avg. Confidence", f"{metrics['avg_confidence']:.2f}")
        m5.metric("Rel. Density", metrics["relation_density"])
        if eval_result.get("has_gold"):
            m6.metric("F1 Score", f"{eval_result['f1']:.2f}", delta=f"P:{eval_result['precision']:.2f} R:{eval_result['recall']:.2f}")
        else:
            m6.metric("Unique Preds", metrics["unique_predicates"])

        # 6 Comprehensive Analytical Tabs
        tab_triples, tab_annotated, tab_entities, tab_graph, tab_flow, tab_eval = st.tabs([
            "Extracted Triples (SPO)",
            "Annotated Text (Evidence)",
            "Detected Entities",
            "Knowledge Graph Preview",
            "Entity-Relationship Flow",
            "Evaluation against Ground Truth"
        ])

        # TAB 1: TRIPLES
        with tab_triples:
            if triples:
                df_triples = pd.DataFrame(triples)[[
                    "subject", "predicate", "object", "confidence",
                    "subject_label", "object_label", "method", "cue", "sentence"
                ]]
                df_triples.columns = [
                    "Subject", "Predicate", "Object", "Confidence",
                    "Subj Type", "Obj Type", "Method", "Matched Cue", "Source Sentence"
                ]
                st.dataframe(df_triples, hide_index=True, use_container_width=True)
            else:
                st.warning("No triples met the confidence threshold. Try lowering the threshold or switching to Hybrid mode.")

        # TAB 2: ANNOTATED TEXT
        with tab_annotated:
            write_annotated(
                "**Sentence Evidence Inspection:** Each sentence displays identified "
                "[[entity|entities]] with type tags and matched [[relation_cue|relation cues]] "
                "highlighted - the direct textual [[provenance]] for every extracted triple."
            )
            sentences = _split_sentences(st.session_state.get("exp7_last_text", text_input))
            cues = [t["cue"] for t in triples if t.get("cue")]

            for s_idx, s_text in enumerate(sentences):
                s_ents = [e for e in entities if e["sentence_idx"] == s_idx]
                annotated_html = _annotate_sentence(s_text, s_ents, cues)
                st.markdown(f"**Sentence {s_idx + 1}:** {annotated_html}", unsafe_allow_html=True)

        # TAB 3: DETECTED ENTITIES
        with tab_entities:
            col_e1, col_e2 = st.columns([3, 2])
            with col_e1:
                if entities:
                    ent_df = pd.DataFrame(entities).drop_duplicates(subset=["text"])[["text", "label", "sentence_idx"]]
                    ent_df.columns = ["Entity Mention", "Classified Type", "First Sentence #"]
                    st.dataframe(ent_df, hide_index=True, use_container_width=True)
                else:
                    st.info("No entities detected.")
            with col_e2:
                if entities:
                    st.write("**Entity Type Distribution**")
                    st.plotly_chart(_build_entity_type_chart(entities, theme), use_container_width=True)

        # TAB 4: KNOWLEDGE GRAPH PREVIEW
        with tab_graph:
            col_gc1, col_gc2, col_gc3 = st.columns([2, 3, 2])
            with col_gc1:
                show_labels = st.checkbox("Show predicate labels on edges", value=True, key="graph_show_preds")
            with col_gc2:
                selected_types = st.multiselect(
                    "Filter by Entity Type",
                    options=ENTITY_TYPE_ORDER,
                    default=ENTITY_TYPE_ORDER,
                    key="graph_type_filter"
                )
            with col_gc3:
                if st.button("Re-layout / Jiggle Graph", use_container_width=True):
                    st.session_state["graph_seed"] = st.session_state.get("graph_seed", 11) + 7
                    st.rerun()

            curr_seed = st.session_state.get("graph_seed", 11)
            graph_fig = _build_graph_figure(
                triples=triples,
                theme=theme,
                show_predicates=show_labels,
                seed=curr_seed,
                type_filter=selected_types
            )
            st.plotly_chart(graph_fig, use_container_width=True)
            st.caption("Tip: Hover over any node or edge to inspect full entity attributes, source sentences, and extraction cues.")

        # TAB 5: FLOW (SANKEY)
        with tab_flow:
            st.write("**Entity Type to Predicate Transition Patterns:**")
            st.plotly_chart(_build_flow_figure(triples, theme), use_container_width=True)
            write_annotated(
                "Traces how [[entity_type|entity types]] flow from subjects into "
                "[[predicate|predicates]] and target object types."
            )

        # TAB 6: EVALUATION
        with tab_eval:
            if eval_result.get("has_gold"):
                st.subheader(f"Ground Truth Evaluation Benchmark ({selected_corpus['name'] if selected_corpus else 'Corpus'})")
                write_annotated(
                    "Every extracted [[triple]] is checked against this corpus's "
                    "[[reference_set|hand-curated gold set]]: a match is a True Positive, an "
                    "extracted triple with no match is a [[false_positive|False Positive]], and "
                    "a gold triple nothing matched is a [[false_negative|False Negative]]."
                )
                col_ev1, col_ev2, col_ev3 = st.columns(3)
                col_ev1.metric("Precision (P)", f"{eval_result['precision'] * 100:.1f}%", help="Fraction of extracted triples that are factually correct.")
                col_ev2.metric("Recall (R)", f"{eval_result['recall'] * 100:.1f}%", help="Fraction of ground truth relations successfully recovered.")
                col_ev3.metric("F1 Score", f"{eval_result['f1'] * 100:.1f}%", help="Harmonic mean of precision and recall.")

                st.write(f"**Summary:** True Positives: `{eval_result['tp_count']}` | False Positives: `{eval_result['fp_count']}` | False Negatives (Omissions): `{eval_result['fn_count']}`")

                col_tp, col_fp, col_fn = st.columns(3)
                with col_tp:
                    st.success(f"**True Positives ({eval_result['tp_count']})**")
                    if eval_result["tp_list"]:
                        st.dataframe(pd.DataFrame(eval_result["tp_list"])[["subject", "predicate", "object"]], hide_index=True)
                    else:
                        st.caption("None.")
                with col_fp:
                    st.warning(f"**False Positives ({eval_result['fp_count']})**")
                    if eval_result["fp_list"]:
                        st.dataframe(pd.DataFrame(eval_result["fp_list"])[["subject", "predicate", "object"]], hide_index=True)
                    else:
                        st.caption("None.")
                with col_fn:
                    st.error(f"**False Negatives ({eval_result['fn_count']})**")
                    if eval_result["fn_list"]:
                        st.dataframe(pd.DataFrame(eval_result["fn_list"], columns=["Subject", "Predicate", "Object"]), hide_index=True)
                    else:
                        st.caption("None.")
            else:
                st.info("Evaluation against ground truth is available for built-in benchmark corpora (e.g. Tech & Business, Science & Academia, Indian Industry, Healthcare). Custom and uploaded texts do not carry pre-annotated reference sets.")

        # 3. Export Card
        st.divider()
        st.subheader("Export for Downstream Graph Ingestion (Experiments 8-9)")
        col_x1, col_x2, col_x3 = st.columns(3)

        exp_df = pd.DataFrame(triples) if triples else pd.DataFrame(columns=["subject", "predicate", "object", "confidence"])
        with col_x1:
            st.download_button(
                "Download Triples (CSV)",
                data=exp_df.to_csv(index=False).encode("utf-8"),
                file_name="exp7_extracted_triples.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col_x2:
            st.download_button(
                "Download Triples (JSON)",
                data=json.dumps(triples, indent=2).encode("utf-8"),
                file_name="exp7_extracted_triples.json",
                mime="application/json",
                use_container_width=True
            )
        with col_x3:
            cypher_statements = [
                f'MERGE (s:Entity {{name: "{t["subject"]}", type: "{t["subject_label"]}"}})\n'
                f'MERGE (o:Entity {{name: "{t["object"]}", type: "{t["object_label"]}"}})\n'
                f'MERGE (s)-[:{t["predicate"].upper()} {{confidence: {t["confidence"]}}}]->(o);'
                for t in triples
            ]
            st.download_button(
                "Download Neo4j Cypher (.cql)",
                data="\n".join(cypher_statements).encode("utf-8"),
                file_name="exp7_graph_import.cql",
                mime="text/plain",
                use_container_width=True
            )

        # 4. Trial Logging
        st.divider()
        st.subheader("Record Current Trial to Session Logbook")
        if st.button("Record Trial to Logbook", use_container_width=True):
            trial_record = {
                "Trial #": len(st.session_state["trials"]) + 1,
                "Document": doc_choice,
                "Method": method,
                "Min Conf": min_confidence,
                "Max Dist": max_dist,
                "Entities": metrics["entities"],
                "Triples": metrics["triples"],
                "High Conf": metrics["high_confidence_triples"],
                "Avg Conf": metrics["avg_confidence"],
                "F1 Score": eval_result.get("f1", "N/A") if eval_result.get("has_gold") else "N/A"
            }
            st.session_state["trials"].append(trial_record)
            st.success(f"Trial #{trial_record['Trial #']} successfully recorded into session logbook.")

    # Display Trials Table if any recorded
    if st.session_state.get("trials"):
        st.divider()
        st.subheader("Recorded Session Trials Logbook")
        st.dataframe(pd.DataFrame(st.session_state["trials"]), hide_index=True, use_container_width=True)
        if st.button("Clear Session Trials Logbook"):
            st.session_state["trials"] = []
            st.rerun()


def render_quiz_section():
    """Renders Section 3: Self-grading Conceptual Assessment Quiz."""
    st.header("Conceptual Assessment Quiz")
    st.write("Complete all questions to assess your understanding of Information Retrieval and Relationship Extraction principles.")

    user_responses = {}
    with st.form("quiz_form"):
        for q in QUIZ_QUESTIONS:
            st.markdown(f"**Question {q['id']}:** {q['question']}")
            key = f"quiz_q_{q['id']}"
            choice = st.radio(
                f"Options for Question {q['id']}:",
                options=q["options"],
                index=None,
                key=key,
                label_visibility="collapsed"
            )
            user_responses[q["id"]] = choice
            st.markdown("---")

        submit_btn = st.form_submit_button("Submit Assessment for Evaluation", type="primary", use_container_width=True)

    if submit_btn:
        score = 0
        unanswered = 0
        feedback = []

        for q in QUIZ_QUESTIONS:
            selected = user_responses.get(q["id"])
            if selected is None:
                unanswered += 1
                feedback.append((q, False, "Question not answered."))
            else:
                sel_idx = q["options"].index(selected)
                is_correct = (sel_idx == q["answer_index"])
                if is_correct:
                    score += 1
                feedback.append((q, is_correct, selected))

        st.session_state["quiz_submitted"] = True
        st.session_state["quiz_score"] = score
        st.session_state["quiz_feedback"] = feedback

        pct = int((score / len(QUIZ_QUESTIONS)) * 100)
        if pct >= 80:
            st.balloons()
            st.success(f"Excellent! Your Score: **{score} / {len(QUIZ_QUESTIONS)} ({pct}%)**")
        elif pct >= 50:
            st.success(f"Good Effort! Your Score: **{score} / {len(QUIZ_QUESTIONS)} ({pct}%)**")
        else:
            st.error(f"Needs Review. Your Score: **{score} / {len(QUIZ_QUESTIONS)} ({pct}%)**")

        st.subheader("Detailed Pedagogical Review & Explanations")
        incorrect = len(feedback) - score - unanswered
        st.markdown(
            f"**Result breakdown:** :green[{score} Correct]  |  :red[{incorrect} Incorrect]"
            + (f"  |  :orange[{unanswered} Unanswered]" if unanswered else "")
        )
        for q, is_corr, sel in feedback:
            status_label = "Correct" if is_corr else "Incorrect"
            with st.expander(f"Q{q['id']}: {q['question']} — [{status_label}]", expanded=not is_corr):
                if is_corr:
                    st.success(f"**Your Answer:** {sel}\n\nThis is correct.")
                else:
                    st.error(f"**Your Answer:** {sel}\n\nThis is incorrect.")
                    st.success(f"**Correct Answer:** {q['options'][q['answer_index']]}")
                st.info(f"**Pedagogical Explanation:** {q['explanation']}")

    elif st.session_state.get("quiz_submitted", False):
        st.info(f"Quiz previously submitted in this session. Score: **{st.session_state.get('quiz_score', 0)} / {len(QUIZ_QUESTIONS)}**")


def render_report_section():
    """Renders Section 4: Dynamic Official Laboratory Report Generation & PDF Download."""
    st.header("Official Laboratory Report Generation")
    st.write("Compile student details, session trials, extracted knowledge triples, and quiz results into a verified PDF report.")

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        student_name = st.text_input("Student Full Name", value=st.session_state["student_info"].get("name", "Student Name"))
    with col_s2:
        student_id = st.text_input("Roll Number / ID", value=st.session_state["student_info"].get("id", "EXP7-001"))
    with col_s3:
        student_batch = st.text_input("Academic Batch", value=st.session_state["student_info"].get("batch", "D17A"))
    with col_s4:
        lab_date = st.date_input("Experiment Date", value=datetime.now())

    st.session_state["student_info"]["name"] = student_name
    st.session_state["student_info"]["id"] = student_id
    st.session_state["student_info"]["batch"] = student_batch
    st.session_state["student_info"]["date"] = str(lab_date)

    st.subheader("Student Analytical Observations & Conclusions")
    student_notes = st.text_area(
        "Enter your interpretation of results, comparisons across extraction methods, and conclusions:",
        value=st.session_state.get("student_notes", (
            "In this experiment, pattern-based relation extraction demonstrated high precision on active-voice "
            "corpora with direct verbal cues ('founded', 'acquired'). The co-occurrence baseline achieved higher "
            "recall but introduced false positives, underscoring the classic IR precision-recall trade-off. "
            "Confidence scoring successfully filtered noise prior to downstream graph construction."
        )),
        height=110
    )
    st.session_state["student_notes"] = student_notes

    trials_df = pd.DataFrame(st.session_state.get("trials", []))
    triples_df = pd.DataFrame(st.session_state.get("kg_triples", []))
    eval_metrics = st.session_state.get("exp7_last_eval")

    st.divider()
    st.subheader("Report Summary Preview")
    st.write(f"**Institution:** {EXPERIMENT_CONFIG['department']} | {EXPERIMENT_CONFIG['course']}")
    st.write(f"**Student:** {student_name} (Roll: `{student_id}`, Batch: `{student_batch}`) | **Date:** {lab_date}")
    st.write(f"**Assessment Score:** `{st.session_state.get('quiz_score', 0)} / {len(QUIZ_QUESTIONS)}` | **Recorded Trials:** `{len(trials_df)}`")

    pdf_bytes = generate_pdf_report(
        student_name=student_name,
        student_id=student_id,
        student_batch=student_batch,
        date_str=str(lab_date),
        trials_df=trials_df,
        quiz_score=st.session_state.get("quiz_score", 0),
        quiz_total=len(QUIZ_QUESTIONS),
        student_notes=student_notes,
        triples_df=triples_df,
        eval_metrics=eval_metrics
    )

    os.makedirs("static", exist_ok=True)
    with open("static/lab_report.pdf", "wb") as f:
        f.write(pdf_bytes)

    st.divider()
    st.subheader("Download Official Laboratory Report (.pdf)")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.download_button(
            label="Download Official lab_report.pdf",
            data=pdf_bytes,
            file_name=f"KGIRS_Exp7_Report_{student_id}.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True
        )
    with col_d2:
        st.link_button(
            "Open PDF Document in Browser Tab",
            url="/app/static/lab_report.pdf",
            use_container_width=True
        )


# ======================================================================================
# 12. INITIALIZATION & MAIN NAVIGATION
# ======================================================================================

def init_session_state():
    """Initializes session state keys."""
    if "trials" not in st.session_state:
        st.session_state["trials"] = []
    if "quiz_answers" not in st.session_state:
        st.session_state["quiz_answers"] = {}
    if "quiz_submitted" not in st.session_state:
        st.session_state["quiz_submitted"] = False
    if "quiz_score" not in st.session_state:
        st.session_state["quiz_score"] = 0
    if "student_info" not in st.session_state:
        st.session_state["student_info"] = {
            "name": "Student Name",
            "id": "EXP7-001",
            "batch": "D17A",
            "date": str(datetime.now().date())
        }
    if "student_notes" not in st.session_state:
        st.session_state["student_notes"] = ""
    if "kg_entities" not in st.session_state:
        st.session_state["kg_entities"] = []
    if "kg_triples" not in st.session_state:
        st.session_state["kg_triples"] = []
    if "kg_source_text" not in st.session_state:
        st.session_state["kg_source_text"] = ""
    if "graph_seed" not in st.session_state:
        st.session_state["graph_seed"] = 11
    if "theme_mode_choice" not in st.session_state:
        st.session_state["theme_mode_choice"] = "Auto"


def main():
    st.set_page_config(
        page_title="Exp 7: Relationship Extraction | KGIRS Virtual Lab",
        page_icon=None,
        layout="wide"
    )

    init_session_state()
    inject_custom_styles()

    theme = get_current_theme()
    inject_mode_css(theme.dark)

    # Formal Institutional Header, with the theme picker in the top-right corner
    header_col, theme_col = st.columns([6, 1])
    with header_col:
        st.markdown(
            f"<div style='border-bottom: 2px solid rgba(128,128,128,0.2); padding-bottom: 0.75rem; margin-bottom: 1.25rem;'>"
            f"<span style='font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.08em; color: #64748b; font-weight: 600;'>"
            f"{EXPERIMENT_CONFIG['department']} · {EXPERIMENT_CONFIG['portal']}</span>"
            f"<h2 style='margin: 0.2rem 0 0.1rem 0; font-weight: 800; letter-spacing: -0.02em;'>{EXPERIMENT_CONFIG['title']}</h2>"
            f"<span style='font-size: 0.88rem; color: #64748b;'>{EXPERIMENT_CONFIG['course']}</span>"
            f"</div>",
            unsafe_allow_html=True
        )
    with theme_col:
        st.selectbox(
            "Theme",
            options=["Auto", "Light", "Dark"],
            key="theme_mode_choice",
            help="Switches both the page chrome and every chart between light and dark mode.",
            label_visibility="collapsed"
        )

    # Sidebar: Lab Navigation & Controls
    st.sidebar.markdown("### Virtual Lab Navigation")
    section = st.sidebar.radio(
        "Select Section",
        options=["Theory", "Simulation", "Quiz", "Report Generation"],
        label_visibility="collapsed"
    )

    st.sidebar.divider()
    st.sidebar.markdown("### Session Progress Tracker")
    quiz_done = st.session_state.get("quiz_submitted", False)
    st.sidebar.write(f"- **Quiz Assessment:** {'Completed' if quiz_done else 'Pending'}")
    if quiz_done:
        st.sidebar.write(f"- **Quiz Score:** `{st.session_state.get('quiz_score', 0)} / {len(QUIZ_QUESTIONS)}`")
    st.sidebar.write(f"- **Trials Logged:** `{len(st.session_state['trials'])}`")
    st.sidebar.write(f"- **Active Triples in Session:** `{len(st.session_state.get('kg_triples', []))}`")

    st.sidebar.divider()
    st.sidebar.markdown("### Downstream Pipeline Contract")
    st.sidebar.caption(
        "Knowledge graph triples emitted here populate `st.session_state['kg_triples']` "
        "for Experiments 8 and 9 (Neo4j Graph Database & Cypher Import)."
    )

    # Section Router
    if section == "Theory":
        render_theory_section()
    elif section == "Simulation":
        render_simulation_section()
    elif section == "Quiz":
        render_quiz_section()
    elif section == "Report Generation":
        render_report_section()

    # Formal Footer
    st.markdown(
        f"<div style='margin-top: 3.5rem; padding-top: 1.25rem; border-top: 1px solid rgba(128,128,128,0.2); "
        f"font-size: 0.82rem; color: #64748b; text-align: center;'>"
        f"{EXPERIMENT_CONFIG['department']} · Virtual Laboratory Portal<br>"
        f"Course: {EXPERIMENT_CONFIG['course']} · Academic Year 2026–2027"
        f"</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
