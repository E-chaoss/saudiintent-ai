"""The 8 intents, in ONE fixed order shared by every model and the demo (Tasks 3-6).

The position in this list is the label id the models use (track_order = 0, ...).
Never reorder, rename, add or remove an intent without a new ADR in docs/decisions/:
a model trained with one order gives wrong answers when read with another order.
Definitions and examples are in docs/intents.md.
"""

INTENTS = [
    "track_order",
    "cancel_order",
    "change_order",
    "get_refund",
    "track_refund",
    "payment_issue",
    "complaint",
    "contact_human_agent",
]

LABEL2ID = {label: i for i, label in enumerate(INTENTS)}
ID2LABEL = {i: label for i, label in enumerate(INTENTS)}
