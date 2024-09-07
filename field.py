from tinygrad import Tensor, dtypes
import numpy as np

def binmul(v1, v2, length=None):
    if v1 < 256 and v2 < 256 and rawmulcache[v1][v2] is not None:
        return rawmulcache[v1][v2]
    if v1 < 2 or v2 < 2:
        return v1 * v2
    if length is None:
        length = 1 << (max(v1, v2).bit_length() - 1).bit_length()
    halflen = length//2
    quarterlen = length//4
    halfmask = (1 << halflen)-1

    L1, R1 = v1 & halfmask, v1 >> halflen
    L2, R2 = v2 & halfmask, v2 >> halflen

    if (L1, R1) == (0, 1):
        outR = binmul(1 << quarterlen, R2, halflen) ^ L2
        return R2 ^ (outR << halflen)

    L1L2 = binmul(L1, L2, halflen)
    R1R2 = binmul(R1, R2, halflen)
    R1R2_high = binmul(1 << quarterlen, R1R2, halflen)
    Z3 = binmul(L1 ^ R1, L2 ^ R2, halflen)
    return (
        L1L2 ^
        R1R2 ^
        ((Z3 ^ L1L2 ^ R1R2 ^ R1R2_high) << halflen)
    )

class BinaryFieldElement:
    def __init__(self, value):
        if isinstance(value, BinaryFieldElement):
            self.value = value.value
        else:
            self.value = value

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return f"<{self.value}>"
    
    def __add__(self, other):
        othervalue = other if isinstance(other, int) else other.value
        if self.value < 256 and othervalue < 256:
            return addcache[self.value, othervalue].item()
        return BinaryFieldElement(self.value ^ othervalue)
    
    __sub__ = __add__

    def __neg__(self):
        return self
    
    def __mul__(self, other):
        othervalue = other if isinstance(other, int) else other.value
        if self.value < 256 and othervalue < 256:
            return mulcache[self.value, othervalue].item()
        return BinaryFieldElement(binmul(self.value, othervalue))

    def __pow__(self, other):
        if other == 0:
            return BinaryFieldElement(1)
        elif other == 1:
            return self
        elif other == 2:
            return self * self
        else:
            return self.__pow__(other % 2) * self.__pow__(other // 2) ** 2
        
    def inv(self):
        # This uses Fermat's Little Theorem for inversion in binary fields
        # Optimized version to reduce exponentiation
        L = 1 << (self.value.bit_length() - 1).bit_length()
        return self ** (2**L - 2)

    def __truediv__(self, other):
        if isinstance(other, int):
            other = BinaryFieldElement(other)
        return BinaryFieldElement(binmul(self.value, other.inv().value))
    
    def __eq__(self, other):
        othervalue = other if isinstance(other, int) else other.value
        return self.value == othervalue

    def bit_length(self):
        return 1 << (self.value.bit_length() - 1).bit_length()

    def to_bytes(self, length, byteorder):
        assert length >= (self.bit_length() + 7) // 8
        return self.value.to_bytes(length, byteorder)
    
    @classmethod
    def from_bytes(cls, b, byteorder):
        return cls(int.from_bytes(b, byteorder))

i = Tensor.arange(256, dtype=dtypes.uint8).reshape((256, 1))
j = Tensor.arange(256, dtype=dtypes.uint8).reshape((1, 256))

addcache = i ^ j


rawmulcache = [[None for _ in range(256)] for _ in range(256)]
mulcache = [[None for _ in range(256)] for _ in range(256)]

for i in range(256):
    for j in range(256):
        rawmulcache[i][j] = binmul(i, j)
        mulcache[i][j] = BinaryFieldElement(rawmulcache[i][j]).value

rawmulcache = Tensor(rawmulcache)
mulcache = Tensor(mulcache)