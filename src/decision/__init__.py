"""
Escalation & Decision Routing System

Core decision layer that determines:
- AUTO_HANDLE: Safe to automatically send reply
- ASSIST_AND_ESCALATE: Reply useful, but human should review
- HUMAN_ESCALATION: Route directly to human support

Based on empirical signals from intent classification, retrieval, and generation components.

Safety-first philosophy: False automation is more costly than unnecessary escalation.
"""

__version__ = "1.0.0"

from enum import Enum

class Decision(Enum):
    """Three-way decision outcomes."""
    AUTO_HANDLE = "AUTO_HANDLE"
    ASSIST_AND_ESCALATE = "ASSIST_AND_ESCALATE"
    HUMAN_ESCALATION = "HUMAN_ESCALATION"
