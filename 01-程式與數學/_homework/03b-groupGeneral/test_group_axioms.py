# test_group_axioms.py
import pytest
from group_elements import *

NUM_TEST_CASES = 100

# 1. 封閉性 (Closure)
def test_closure():
    for _ in range(NUM_TEST_CASES):
        a = group_random()
        b = group_random()
        result = group_operation(a, b)
        assert is_in_group(result), f"Closure failed: {a} + {b} = {result} is not in G"

# 2. 結合性 (Associativity)
def test_associativity():
    for _ in range(NUM_TEST_CASES):
        a = group_random()
        b = group_random()
        c = group_random()
        assert group_operation(group_operation(a, b), c) == group_operation(a, group_operation(b, c)), \
            f"Associativity failed: ({a} + {b}) + {c} != {a} + ({b} + {c})"

# 3. 單位元素 (Identity Element)
def test_identity_element():
    for _ in range(NUM_TEST_CASES):
        a = group_random()
        # 左單位元素
        assert group_operation(a, group_identity()) == a, \
            f"Left identity failed: {a} + {group_identity()} != {a}"
        # 右單位元素
        assert group_operation(group_identity(), a) == a, \
            f"Right identity failed: {group_identity()} + {a} != {a}"

# 4. 反元素 (Inverse Element)
def test_inverse_element():
    for _ in range(NUM_TEST_CASES):
        a = group_random()
        # 加法的反元素是負號
        # 這裡我們需要一個函式來計算反元素，因為它不是固定的運算

        a_inverse = group_inverse(a)
        
        # 檢查反元素是否也在集合 G 中 (對於整數，-a 仍然是整數)
        assert is_in_group(a_inverse), f"Inverse {a_inverse} for {a} is not in G"

        # 檢查左反元素
        assert group_operation(a, a_inverse) == group_identity(), \
            f"Left inverse failed: {a} + {a_inverse} != {group_identity()}"
        # 檢查右反元素
        assert group_operation(a_inverse, a) == group_identity(), \
            f"Right inverse failed: {a_inverse} + {a} != {group_identity()}"
