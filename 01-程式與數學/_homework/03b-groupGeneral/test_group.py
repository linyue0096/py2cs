import pytest
import group_axioms
import group_add
import group_even_add
import group_odd_add
import group_fractions_add
import group_fractions_mul
import group_float_add
import group_float_mul

def test_add_group():
    group_axioms.check_all_axioms(group_add)

def test_float_add_group():
    group_axioms.check_all_axioms(group_float_add)

def test_float_mul_group():
    group_axioms.check_all_axioms(group_float_mul)

def test_even_group():
    group_axioms.check_all_axioms(group_even_add)

def test_odd_group():
    group_axioms.check_all_axioms(group_odd_add)


def test_fractions_add_group():
    group_axioms.check_all_axioms(group_fractions_add)

def test_fractions_mul_group():
    group_axioms.check_all_axioms(group_fractions_mul)


