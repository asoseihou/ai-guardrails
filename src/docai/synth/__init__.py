"""Synthetic fiscal documents for tests and demos: PDFs, XMLs and ground truth.

Nothing here is used to *validate* documents; the rules live in ``docai.domain``.
"""

from docai.synth.cnpj import generate_cnpj
from docai.synth.corpus import generate_corpus

__all__ = ["generate_cnpj", "generate_corpus"]
