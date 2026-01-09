from enum import Enum, auto


class PlayerState(Enum):
    """
    High-level player movement / control states.
    Used to drive movement logic and transitions.
    """

    IDLE = auto()
    WALK = auto()
    SPRINT = auto()

    JUMP = auto()
    AIR = auto()

    CROUCH = auto()
    SLIDE = auto()
