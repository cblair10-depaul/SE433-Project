from typing import List, Optional
import copy


def add(left: Optional[List[int]], right: Optional[List[int]]) -> Optional[List[int]]:
    if left is None or right is None:
        return None

    left_copy = copy.deepcopy(left)
    right_copy = copy.deepcopy(right)

    left_copy.reverse()
    right_copy.reverse()

    result = []

    carry = 0
    for i in range(max(len(left_copy), len(right_copy))):
        left_digit = left_copy[i] if i < len(left_copy) else 0
        right_digit = right_copy[i] if i < len(right_copy) else 0

        if left_digit < 0 or left_digit > 9 or right_digit < 0 or right_digit > 9:
            raise ValueError()

        total = left_digit + right_digit + carry

        result.insert(0, total % 10)
        carry = total // 10

    if carry > 0:
        result.insert(0, carry)

    while len(result) > 1 and result[0] == 0:
        result.pop(0)

    return result

if __name__ == "__main__":
    print(add([2,6], [4]))