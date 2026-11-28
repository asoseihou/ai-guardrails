import random

import pytest
from hypothesis import given
from hypothesis import strategies as st

from docai.domain.cnpj import cnpj_check_digits, format_cnpj, is_valid_cnpj, normalize_cnpj
from docai.synth.cnpj import SYNTHETIC_BRANCH_RANGE, corrupt_cnpj, generate_cnpj


@pytest.mark.parametrize(
    "value",
    ["11.222.333/0001-81", "11222333000181", " 11 222 333 0001 81 ", "12.ABC.345/01DE-35"],
)
def test_known_valid(value):
    assert is_valid_cnpj(value)


@pytest.mark.parametrize(
    "value",
    [
        "11.222.333/0001-80",  # wrong second digit
        "11.222.333/0001-91",  # wrong first digit
        "1122233300018",  # 13 positions
        "112223330001811",  # 15 positions
        "00.000.000/0000-00",  # placeholder
        "11111111111111",
        "12.ABC.345/01DE-3A",  # check digits must be numeric
        "",
    ],
)
def test_known_invalid(value):
    assert not is_valid_cnpj(value)


def test_check_digits_and_format():
    assert cnpj_check_digits("112223330001") == "81"
    assert cnpj_check_digits("12ABC34501DE") == "35"
    assert format_cnpj("11222333000181") == "11.222.333/0001-81"
    assert normalize_cnpj("12.abc.345/01de-35") == "12ABC34501DE35"
    with pytest.raises(ValueError):
        cnpj_check_digits("123")


@given(st.integers(min_value=0, max_value=2**32))
def test_generated_cnpjs_are_valid_and_synthetic(seed):
    cnpj = generate_cnpj(random.Random(seed))
    assert len(cnpj) == 14 and cnpj.isdigit()
    assert is_valid_cnpj(cnpj)
    low, high = SYNTHETIC_BRANCH_RANGE
    assert low <= int(cnpj[8:12]) <= high


@given(st.integers(min_value=0, max_value=2**32), st.sampled_from(["swap", "substitute"]))
def test_corruption_always_changes_the_value(seed, mode):
    rng = random.Random(seed)
    cnpj = generate_cnpj(rng)
    assert corrupt_cnpj(cnpj, rng, mode) != cnpj


@given(st.integers(min_value=0, max_value=2**32), st.integers(12, 13), st.integers(1, 9))
def test_changing_a_check_digit_is_always_detected(seed, position, delta):
    cnpj = generate_cnpj(random.Random(seed))
    wrong = cnpj[:position] + str((int(cnpj[position]) + delta) % 10) + cnpj[position + 1 :]
    assert not is_valid_cnpj(wrong)


def test_single_digit_misreads_are_almost_always_detected():
    # mod 11 maps remainders 0 and 1 to the same digit, so detection is high but not 100%
    rng = random.Random(0)
    trials = 3000
    detected = sum(
        not is_valid_cnpj(corrupt_cnpj(generate_cnpj(rng), rng, "substitute"))
        for _ in range(trials)
    )
    assert detected / trials > 0.95
