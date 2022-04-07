"""Protocol classes for type hints
"""
from __future__ import annotations
from abc import abstractmethod
from typing import (Protocol, SupportsComplex, SupportsFloat, SupportsIndex,
                    SupportsInt, runtime_checkable)
# pylint: disable=abstract-method, too-few-public-methods

@runtime_checkable
class Convertible(Protocol):
    """Types that can be converted to standard numeric types.

    Has methods `__complex__`, `__float_`, `__index__`, `__int__`.
    """
    @abstractmethod
    def __complex__(self) -> complex:
        pass
    @abstractmethod
    def __float__(self) -> float:
        pass
    @abstractmethod
    def __int__(self) -> int:
        pass
    @abstractmethod
    def __index__(self) -> int:
        pass


@runtime_checkable
class Ordered(Protocol):
    """Classes that can be used in arithmetic comparisons.

    Defines all of the comparison operators, `==`, `!=`, `<`, `<=`, `>`, `>=`.
    """
    @abstractmethod
    def __eq__(self, other: Ordered) -> bool:
        pass
    @abstractmethod
    def __ne__(self, other: Ordered) -> bool:
        pass
    @abstractmethod
    def __lt__(self, other: Ordered) -> bool:
        pass
    @abstractmethod
    def __le__(self, other: Ordered) -> bool:
        pass
    @abstractmethod
    def __gt__(self, other: Ordered) -> bool:
        pass
    @abstractmethod
    def __ge__(self, other: Ordered) -> bool:
        pass


@runtime_checkable
class Arithmetic(Protocol):
    """Classes that can be used in arithmetic operations.

    Defines all of the arithmetic operators, `==`, `!=`, `<`, `<=`, `>`, `>=`.
    """
    @abstractmethod
    def __eq__(self, other: Arithmetic) -> bool:
        pass
    @abstractmethod
    def __ne__(self, other: Arithmetic) -> bool:
        pass
    @abstractmethod
    def __add__(self, other: Arithmetic) -> Arithmetic:
        pass
    @abstractmethod
    def __radd__(self, other: Arithmetic) -> Arithmetic:
        pass
    @abstractmethod
    def __sub__(self, other: Arithmetic) -> Arithmetic:
        pass
    @abstractmethod
    def __rsub__(self, other: Arithmetic) -> Arithmetic:
        pass
    @abstractmethod
    def __mul__(self, other: Arithmetic) -> Arithmetic:
        pass
    @abstractmethod
    def __rmul__(self, other: Arithmetic) -> Arithmetic:
        pass
    @abstractmethod
    def __truediv__(self, other: Arithmetic) -> Arithmetic:
        pass
    @abstractmethod
    def __rtruediv__(self, other: Arithmetic) -> Arithmetic:
        pass
    @abstractmethod
    def __pow__(self, other: Arithmetic) -> Arithmetic:
        pass
    @abstractmethod
    def __rpow__(self, other: Arithmetic) -> Arithmetic:
        pass
    @abstractmethod
    def __neg__(self) -> Arithmetic:
        pass
    @abstractmethod
    def __pos__(self) -> Arithmetic:
        pass
    @abstractmethod
    def __abs__(self) -> Arithmetic:
        pass


@runtime_checkable
class Roundable(Protocol):
    """Classes that can be used in rounding/modular routines.

    Defines the operators `%`, `//`, and the functions  `divmod`, `round`,
    `math.floor,ceil,trunc`.
    """
    @abstractmethod
    def __floordiv__(self, other: Roundable) -> Roundable:
        pass
    @abstractmethod
    def __rfloordiv__(self, other: Roundable) -> Roundable:
        pass
    @abstractmethod
    def __mod__(self, other: Roundable) -> Roundable:
        pass
    @abstractmethod
    def __rmod__(self, other: Roundable) -> Roundable:
        pass
    @abstractmethod
    def __divmod__(self, other: Roundable) -> Roundable:
        pass
    @abstractmethod
    def __rdivmod__(self, other: Roundable) -> Roundable:
        pass
    @abstractmethod
    def __round__(self) -> Roundable:
        pass
    @abstractmethod
    def __trunc__(self) -> Roundable:
        pass
    @abstractmethod
    def __floor__(self) -> Roundable:
        pass
    @abstractmethod
    def __ceil__(self) -> Roundable:
        pass


@runtime_checkable
class BitWise(Protocol):
    """Classes that can be used in bit-wise operators.

    Defines all of the bit-wise operators: `<<`, `>>`, `&`, `^`, `|`, `~`.
    """
    @abstractmethod
    def __lshift__(self, other: SupportsInt) -> BitWise:
        pass
    @abstractmethod
    def __rlshift__(self, other: SupportsInt) -> BitWise:
        pass
    @abstractmethod
    def __rshift__(self, other: SupportsInt) -> BitWise:
        pass
    @abstractmethod
    def __rrshift__(self, other: SupportsInt) -> BitWise:
        pass
    @abstractmethod
    def __and__(self, other: BitWise) -> BitWise:
        pass
    @abstractmethod
    def __rand__(self, other: BitWise) -> BitWise:
        pass
    @abstractmethod
    def __xor__(self, other: BitWise) -> BitWise:
        pass
    @abstractmethod
    def __rxor__(self, other: BitWise) -> BitWise:
        pass
    @abstractmethod
    def __or__(self, other: BitWise) -> BitWise:
        pass
    @abstractmethod
    def __ror__(self, other: BitWise) -> BitWise:
        pass
    @abstractmethod
    def __invert__(self) -> BitWise:
        pass


@runtime_checkable
class InplaceArithmetic(Arithmetic, Protocol):
    """Classes that can be used with arithmetic in-place updaters.

    Defines the arithmetic operators `+=`, `-=`, `*=`, `/=`, `**=`.
    """
    @abstractmethod
    def __iadd__(self, other: Arithmetic) -> InplaceArithmetic:
        pass
    @abstractmethod
    def __isub__(self, other: Arithmetic) -> InplaceArithmetic:
        pass
    @abstractmethod
    def __imul__(self, other: Arithmetic) -> InplaceArithmetic:
        pass
    @abstractmethod
    def __itruediv__(self, other: Arithmetic) -> InplaceArithmetic:
        pass
    @abstractmethod
    def __ipow__(self, other: Arithmetic) -> InplaceArithmetic:
        pass


@runtime_checkable
class InplaceRoundable(Roundable, Protocol):
    """Classes that can be used with rounding/modular in-place updaters.

    Defines the operators `%=` and `//=`.
    """
    @abstractmethod
    def __ifloordiv__(self, other: Roundable) -> InplaceRoundable:
        pass
    @abstractmethod
    def __imod__(self, other: Roundable) -> InplaceRoundable:
        pass


@runtime_checkable
class InplaceBitWise(BitWise, Protocol):
    """Classes that can be used with bit-wise in-place updaters.

    Defines the bit-wise updaters: `<<=`, `>>=`, `&=`, `^=`, `|=`.
    """
    @abstractmethod
    def __ilshift__(self, other: BitWise) -> InplaceBitWise:
        pass
    @abstractmethod
    def __irshift__(self, other: BitWise) -> InplaceBitWise:
        pass
    @abstractmethod
    def __iand__(self, other: BitWise) -> InplaceBitWise:
        pass
    @abstractmethod
    def __ixor__(self, other: BitWise) -> InplaceBitWise:
        pass
    @abstractmethod
    def __ior__(self, other: BitWise) -> InplaceBitWise:
        pass
