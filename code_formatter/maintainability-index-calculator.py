import math

def calculate_halstead_volume(operators_count, operands_count, unique_operators, unique_operands):
    # Halstead Volume = (Operators Count + Operands Count) * log2(Unique Operators + Unique Operands)
    return (operators_count + operands_count) * math.log2(unique_operators + unique_operands)

def calculate_maintainability_index(cyclomatic_complexity, halstead_volume, loc):
    # Maintainability Index formula
    if loc == 0 or halstead_volume <= 0:  # Avoid division by zero or log of non-positive numbers
        return 0

    mi = 171 - 5.2 * math.log(halstead_volume) - 0.23 * cyclomatic_complexity - 16.2 * math.log(loc)
    # Scale it to 0-100
    return max(0, min(100, mi))
