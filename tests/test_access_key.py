import doctest
import random

import pytest
from hypothesis import given
from hypothesis import strategies as st

import docai.domain.access_key as access_key_module
import docai.synth.cnpj as cnpj_module
from docai.domain.access_key import (
    KEY_LENGTH,
    build_access_key,
    find_keys_in_text,
    format_access_key,
    is_valid_access_key,
    key_check_digit,
    parse_access_key,
)
from docai.synth.cnpj import generate_cnpj


def test_doctests():
    for module in (access_key_module, cnpj_module):
        assert doctest.testmod(module).failed == 0


def test_check_digit_of_a_published_layout_example():
    # 43-digit body from the NF-e integration manual's layout example -> DV 3
    assert key_check_digit("3508059999909091027055001000000001518005127") == "3"


@given(
    seed=st.integers(0, 2**32),
    year=st.integers(2006, 2099),
    month=st.integers(1, 12),
    model=st.sampled_from(["55", "57"]),
    series=st.integers(0, 999),
    number=st.integers(1, 999_999_999),
    code=st.integers(0, 99_999_999),
)
def test_build_parse_roundtrip(seed, year, month, model, series, number, code):
    cnpj = generate_cnpj(random.Random(seed))
    key = build_access_key(
        cuf="35",
        year=year,
        month=month,
        cnpj=cnpj,
        model=model,
        series=series,
        number=number,
        code=code,
    )
    assert len(key) == KEY_LENGTH and is_valid_access_key(key)
    parsed = parse_access_key(key)
    assert (parsed.cuf, parsed.cnpj, parsed.model) == ("35", cnpj, model)
    assert (parsed.year % 100, parsed.month) == (year % 100, month)
    assert (parsed.series, parsed.number, int(parsed.code)) == (series, number, code)
    assert parsed.value == key
    assert parsed.doc_type == {"55": "nfe", "57": "cte"}[model]


@given(st.integers(0, 2**32), st.integers(1, 9))
def test_wrong_check_digit_is_always_detected(seed, delta):
    rng = random.Random(seed)
    key = build_access_key(
        cuf="33",
        year=2025,
        month=rng.randint(1, 12),
        cnpj=generate_cnpj(rng),
        model="55",
        series=1,
        number=rng.randint(1, 10**6),
        code=rng.randint(0, 10**7),
    )
    wrong = key[:-1] + str((int(key[-1]) + delta) % 10)
    assert not is_valid_access_key(wrong)


def test_single_digit_misreads_are_mostly_detected():
    rng = random.Random(1)
    detected, trials = 0, 2000
    for _ in range(trials):
        key = build_access_key(
            cuf="35",
            year=2026,
            month=3,
            cnpj=generate_cnpj(rng),
            model="57",
            series=1,
            number=rng.randint(1, 10**6),
            code=rng.randint(0, 10**7),
        )
        pos = rng.randrange(43)
        wrong = key[:pos] + str((int(key[pos]) + rng.randint(1, 9)) % 10) + key[pos + 1 :]
        detected += not is_valid_access_key(wrong)
    assert detected / trials > 0.95


def test_format_and_find_in_text():
    key = build_access_key(
        cuf="35",
        year=2025,
        month=6,
        cnpj="11222333000181",
        model="55",
        series=1,
        number=12345,
        code=12345678,
    )
    printed = format_access_key(key)
    assert printed.count(" ") == 10
    text = f"FOLHA 1/1\nCHAVE DE ACESSO\n{printed}\nPROTOCOLO 135250000000001"
    assert find_keys_in_text(text) == [key]
    assert find_keys_in_text(f"x {key} y") == [key]  # unformatted
    assert find_keys_in_text(f"9{key}") == []  # glued to another digit: not a key


@pytest.mark.parametrize("bad", ["123", "a" * 44, "1" * 43])
def test_invalid_inputs(bad):
    assert not is_valid_access_key(bad)
    if len("".join(c for c in bad if c.isdigit())) != 44:
        with pytest.raises(ValueError):
            parse_access_key(bad)
