"""
tools.py
────────
Pipecat-compatible tool (function-calling) handlers.

Each function receives a FunctionCallParams object from Pipecat and must call
params.result_callback() with the result dict.

The JSON schemas below are what Pipecat sends to the LLM so it knows how to
invoke each tool.  We use Groq-compatible OpenAI-style tool schemas.
"""

import logging
from pipecat.services.llm_service import FunctionCallParams
from ticket_store import create_ticket as _create_ticket, update_ticket as _update_ticket

logger = logging.getLogger(__name__)

# ─── Tool schemas (sent to the LLM) ──────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "create_ticket",
            "description": (
                "Create a maintenance ticket once you have collected ALL required details: "
                "name, flat, issue, location, severity, and when it started. "
                "Only call this after the resident has confirmed the details."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Full name of the resident or staff member",
                    },
                    "flat": {
                        "type": "string",
                        "description": "Flat/unit number including block, e.g. 'A-302' or 'B-204'",
                    },
                    "issue": {
                        "type": "string",
                        "enum": [
                            "water_leakage",
                            "pipe_burst",
                            "no_water_supply",
                            "drainage_blockage",
                            "low_pressure",
                            "sewage_issue",
                            "tank_issue",
                            "pump_failure",
                        ],
                        "description": "Standardised issue type",
                    },
                    "location": {
                        "type": "string",
                        "description": "Exact location of the problem, e.g. 'kitchen sink', 'bathroom floor drain'",
                    },
                    "severity": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "critical"],
                        "description": "Severity: low=dripping, medium=moderate, high=heavy leak, critical=burst/flooding",
                    },
                    "started_when": {
                        "type": "string",
                        "description": "When the problem started, e.g. 'this morning', 'since yesterday evening'",
                    },
                },
                "required": ["name", "flat", "issue", "location", "severity", "started_when"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_ticket",
            "description": "Update an existing ticket if the resident provides corrections or new information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_id": {
                        "type": "string",
                        "description": "The ticket ID to update (returned by create_ticket)",
                    },
                    "updates": {
                        "type": "object",
                        "description": "Key-value pairs to update on the ticket",
                        "additionalProperties": {"type": "string"},
                    },
                },
                "required": ["ticket_id", "updates"],
            },
        },
    },
]


# ─── Handler functions ────────────────────────────────────────────────────────

async def handle_create_ticket(params: FunctionCallParams) -> None:
    """Called by Pipecat when the LLM invokes create_ticket."""
    args = params.arguments
    try:
        ticket = await _create_ticket(
            name=args["name"],
            flat=args["flat"],
            issue=args["issue"],
            location=args["location"],
            severity=args["severity"],
            started_when=args["started_when"],
        )
        logger.info("Ticket created: %s", ticket)
        await params.result_callback(
            {
                "success": True,
                "ticket_id": ticket["id"],
                "message": "Ticket created successfully.",
            }
        )
    except Exception as exc:
        logger.error("create_ticket failed: %s", exc)
        await params.result_callback(
            {"success": False, "error": str(exc)}
        )


async def handle_update_ticket(params: FunctionCallParams) -> None:
    """Called by Pipecat when the LLM invokes update_ticket."""
    args = params.arguments
    try:
        updated = await _update_ticket(args["ticket_id"], args["updates"])
        if updated:
            logger.info("Ticket updated: %s", updated)
            await params.result_callback({"success": True, "ticket": updated})
        else:
            await params.result_callback(
                {"success": False, "error": f"Ticket {args['ticket_id']} not found"}
            )
    except Exception as exc:
        logger.error("update_ticket failed: %s", exc)
        await params.result_callback({"success": False, "error": str(exc)})


# ─── Registration helper ──────────────────────────────────────────────────────

def register_tools(llm) -> None:
    """Register all tool handlers with the Pipecat LLM service."""
    llm.register_function("create_ticket", handle_create_ticket)
    llm.register_function("update_ticket", handle_update_ticket)
