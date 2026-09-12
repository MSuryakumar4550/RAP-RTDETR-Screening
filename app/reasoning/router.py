"""
Intent Router (Pure Python - Zero Frameworks)
---------------------------------------------
Analyzes user queries to determine whether executing the computer vision
detection model is necessary, or if the question is unrelated to image contents.

Hard Constraint #1 Compliant: No LangChain, CrewAI, or AutoGen.
"""

import re
from typing import Tuple

# Keywords strongly correlated with visual reasoning and PPE compliance
VISUAL_KEYWORDS = {
    "person", "people", "worker", "workers", "anyone", "someone", "everybody",
    "helmet", "helmets", "hard-hat", "hard-hats", "hardhat", "hardhats", "hat", "cap",
    "vest", "vests", "safety-vest", "jacket", "ppe", "equipment", "gear",
    "how many", "count", "number of", "wearing", "wear", "without", "missing",
    "detect", "see", "show", "image", "photo", "picture", "scene", "present",
    "most common", "standing", "working", "safety"
}

# Regex patterns for fast deterministic intent classification
COUNT_PATTERN = re.compile(r"\b(how many|count|number of)\b", re.IGNORECASE)
PPE_COMPLIANCE_PATTERN = re.compile(r"\b(helmet|hardhat|vest|wearing|safety|ppe|gear)\b", re.IGNORECASE)
OBJECT_QUERY_PATTERN = re.compile(r"\b(what('s| is)|who|where|is there|are there|can you see)\b", re.IGNORECASE)

class IntentRouter:
    """Classifies user intent without bloated LLM framework dependencies."""

    @staticmethod
    def classify_intent(query: str) -> Tuple[str, bool]:
        """
        Determines query intent.
        Returns:
            intent_label (str): e.g., 'IMAGE_OBJECT_QUERY' or 'UNRELATED_QUERY'
            requires_detector (bool): True if RT-DETR should be invoked.
        """
        clean_query = query.strip().lower()

        # Rule 1: Empty or extremely short nonsense input
        if len(clean_query) < 3:
            return "AMBIGUOUS_QUERY", False

        # Rule 2: Token match against visual inspection vocabulary
        tokens = set(re.findall(r"\b\w+(?:-\w+)?\b", clean_query))
        overlap = tokens.intersection(VISUAL_KEYWORDS)

        if overlap or COUNT_PATTERN.search(clean_query) or PPE_COMPLIANCE_PATTERN.search(clean_query):
            return "IMAGE_OBJECT_QUERY", True

        # Rule 3: General image inquiry
        if OBJECT_QUERY_PATTERN.search(clean_query) and any(w in clean_query for w in ["in this", "here", "image", "photo"]):
            return "IMAGE_OBJECT_QUERY", True

        # Rule 4: Unrelated inquiry (e.g. general knowledge, math, chit-chat)
        return "UNRELATED_QUERY", False

    @staticmethod
    def answer_unrelated_query(query: str) -> str:
        """Returns deterministic response for queries not pertaining to image analysis."""
        return (
            f"The question '{query}' does not appear to be related to the visual contents or "
            f"safety compliance of the uploaded image. The detection pipeline was not invoked."
        )

intent_router = IntentRouter()
