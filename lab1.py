# Logical operation OR
def or_gate(x1: int, x2: int) -> int:
    """
    The function performs the logical OR operation on two input values.
    Returns 1 if at least one of the values is 1.
    """
    return x1 | x2

# Logical operation AND
def and_gate(x1: int, x2: int) -> int:
    """
    The function performs the logical AND operation on two input values.
    Returns 1 if both values are 1.
    """
    return x1 & x2

# Logical operation XOR throw OR and AND
def xor_gate(x1: int, x2: int) -> int:
    """
    The function performs the logical XOR operation on two input values. XOR is implemented through OR and AND.
    Returns 1 if one of the operands is 1, but not both.
    """
    return or_gate(x1, x2) & ~and_gate(x1, x2)

# Testing
# Print the XOR value for all possible combinations of input values
for a in [0, 1]:
    for b in [0, 1]:
        print(f"xor({a}, {b}) = {xor_gate(a, b)}")
