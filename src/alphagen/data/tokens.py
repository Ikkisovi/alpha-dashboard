from enum import IntEnum
from typing import Type
from alphagen_qlib.stock_data import FeatureType
from alphagen.data.expression import Operator


class SequenceIndicatorType(IntEnum):
    BEG = 0
    SEP = 1


class Token:
    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return isinstance(other, Token) and str(self) == str(other)

    def __hash__(self):
        return hash(str(self))


class ConstantToken(Token):
    def __init__(self, constant: float) -> None:
        self.constant = constant

    def __str__(self): return str(self.constant)

    def __eq__(self, other):
        return isinstance(other, ConstantToken) and self.constant == other.constant

    def __hash__(self):
        return hash(self.constant)


class DeltaTimeToken(Token):
    def __init__(self, delta_time: int) -> None:
        self.delta_time = delta_time

    def __str__(self): return str(self.delta_time)

    def __eq__(self, other):
        return isinstance(other, DeltaTimeToken) and self.delta_time == other.delta_time

    def __hash__(self):
        return hash(self.delta_time)


class FeatureToken(Token):
    def __init__(self, feature: FeatureType) -> None:
        self.feature = feature

    def __str__(self): return '$' + self.feature.name.lower()

    def __eq__(self, other):
        return isinstance(other, FeatureToken) and self.feature == other.feature

    def __hash__(self):
        return hash(self.feature)


class OperatorToken(Token):
    def __init__(self, operator: Type[Operator]) -> None:
        self.operator = operator

    def __str__(self): return self.operator.__name__

    def __eq__(self, other):
        return isinstance(other, OperatorToken) and self.operator == other.operator

    def __hash__(self):
        return hash(self.operator)


class SequenceIndicatorToken(Token):
    def __init__(self, indicator: SequenceIndicatorType) -> None:
        self.indicator = indicator

    def __str__(self): return self.indicator.name

    def __eq__(self, other):
        return isinstance(other, SequenceIndicatorToken) and self.indicator == other.indicator

    def __hash__(self):
        return hash(self.indicator)


BEG_TOKEN = SequenceIndicatorToken(SequenceIndicatorType.BEG)
SEP_TOKEN = SequenceIndicatorToken(SequenceIndicatorType.SEP)


class MacroToken(Token):
    """
    Represents a macro action that expands into a sequence of tokens.

    Attributes:
        tokens: List of tokens this macro expands into
        name: Display name for the macro
        source: Origin of the macro ('static_seed' or 'dynamic_pool')
        category: Semantic category ('having', 'correlation', 'range', 'momentum', 'general')
    """
    def __init__(self, tokens: list[Token], name: str = "Macro",
                 source: str = "unknown", category: str = "general") -> None:
        self.tokens = tokens
        self.name = name
        self.source = source
        self.category = category

    def __str__(self): return f"[{self.name}]"

    def __repr__(self): return str(self)

    def __eq__(self, other):
        return isinstance(other, MacroToken) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    @property
    def is_empty(self) -> bool:
        """Check if macro has no tokens (placeholder)."""
        return len(self.tokens) == 0

    @property
    def length(self) -> int:
        """Number of tokens this macro expands into."""
        return len(self.tokens)

    def get_first_token(self):
        """Get the first token for validity checking."""
        return self.tokens[0] if self.tokens else None
