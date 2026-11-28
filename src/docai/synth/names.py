"""Word lists for synthetic parties, products and services.

Company names are random combinations of invented words; any resemblance to a
real business is coincidental. Cities are real public place names, used only as
addresses.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from docai.synth.cnpj import generate_cnpj

NAME_HEADS = [
    "Aurora",
    "Boreal",
    "Cerrado",
    "Delta Sul",
    "Estrela Guia",
    "Horizonte Azul",
    "Ipê Dourado",
    "Jatobá",
    "Litoral Norte",
    "Mirante",
    "Nascente Clara",
    "Orla",
    "Pampa Verde",
    "Quaresmeira",
    "Rio Sereno",
    "Serra Alta",
    "Tropical",
    "Vale Claro",
    "Xaréu",
    "Zênite",
    "Carnaúba",
    "Bromélia",
    "Ventania",
    "Maré Alta",
    "Pedra Branca",
    "Sol Poente",
    "Araucária",
    "Cajueiro",
]

NAME_SECTORS = [
    "Transportes",
    "Logística",
    "Comércio de Peças",
    "Indústria Metalúrgica",
    "Serviços Portuários",
    "Armazéns Gerais",
    "Engenharia",
    "Distribuidora",
    "Tecnologia Industrial",
    "Soluções em Carga",
    "Equipamentos",
    "Manutenção Industrial",
]

LEGAL_SUFFIXES = ["Ltda", "Ltda", "Ltda", "S.A.", "ME"]

CITIES = [
    ("São Paulo", "SP"),
    ("Campinas", "SP"),
    ("Santos", "SP"),
    ("Rio de Janeiro", "RJ"),
    ("Niterói", "RJ"),
    ("Macaé", "RJ"),
    ("Belo Horizonte", "MG"),
    ("Contagem", "MG"),
    ("Vitória", "ES"),
    ("Serra", "ES"),
    ("Curitiba", "PR"),
    ("Joinville", "SC"),
    ("Porto Alegre", "RS"),
    ("Salvador", "BA"),
    ("Recife", "PE"),
    ("Fortaleza", "CE"),
    ("Manaus", "AM"),
    ("Goiânia", "GO"),
    ("Brasília", "DF"),
    ("Belém", "PA"),
]

STREET_TYPES = ["Rua", "Avenida", "Rodovia", "Travessa", "Alameda"]
STREET_NAMES = [
    "das Acácias",
    "dos Ipês",
    "do Porto",
    "Industrial",
    "das Palmeiras",
    "do Comércio",
    "Beira Rio",
    "dos Navegantes",
    "da Estação",
    "das Indústrias",
    "do Contorno",
    "Central",
]

PRODUCTS = [
    ("Válvula de esfera 2 pol", "84818099"),
    ("Rolamento blindado 6204", "84821010"),
    ("Parafuso sextavado M12", "73181500"),
    ("Mangueira hidráulica 1/2", "40093100"),
    ("Junta de vedação metálica", "84841000"),
    ("Filtro de óleo industrial", "84212300"),
    ("Cabo de aço galvanizado", "73121090"),
    ("Chapa de aço carbono", "72085200"),
    ("Flange cego classe 150", "73079100"),
    ("Manômetro de pressão", "90262010"),
    ("Luva de proteção nitrílica", "40151900"),
    ("Tinta epóxi anticorrosiva", "32089010"),
]

SERVICES = [
    ("11.04", "Carga, descarga, arrumação e guarda de bens"),
    ("16.02", "Outros serviços de transporte de natureza municipal"),
    ("17.01", "Assessoria ou consultoria de qualquer natureza"),
    ("14.01", "Manutenção e conservação de máquinas e equipamentos"),
    ("7.10", "Limpeza e conservação de imóveis"),
]

FREIGHT_COMPONENTS = ["AD VALOREM", "GRIS", "PEDÁGIO", "TAXA DE COLETA", "TAXA DE ENTREGA"]

CARGO_DESCRIPTIONS = [
    "PEÇAS INDUSTRIAIS",
    "EQUIPAMENTOS",
    "MATERIAL ELÉTRICO",
    "TUBULAÇÕES",
    "CARGA GERAL",
]

BANKS = [("Banco Exemplo S.A.", "999"), ("Banco Fictício do Brasil", "998")]


@dataclass(frozen=True)
class Company:
    """A synthetic company (name, valid CNPJ in the synthetic branch range, address)."""

    name: str
    cnpj: str
    city: str
    uf: str
    street: str


def company_pool(rng: random.Random, size: int = 48) -> list[Company]:
    """A pool of distinct synthetic companies."""
    pool: list[Company] = []
    seen: set[str] = set()
    while len(pool) < size:
        name = f"{rng.choice(NAME_HEADS)} {rng.choice(NAME_SECTORS)} {rng.choice(LEGAL_SUFFIXES)}"
        if name in seen:
            continue
        seen.add(name)
        city, uf = rng.choice(CITIES)
        street = f"{rng.choice(STREET_TYPES)} {rng.choice(STREET_NAMES)}, {rng.randint(10, 4999)}"
        pool.append(Company(name=name, cnpj=generate_cnpj(rng), city=city, uf=uf, street=street))
    return pool
