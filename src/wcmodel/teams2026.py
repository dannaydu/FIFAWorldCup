"""2026 World Cup reference data: team pool, seed Elo, and the OFFICIAL draw.

`GROUPS_2026` is the real 2026 draw (team names normalized to the conventions in
the martj42 international-results dataset, e.g. "United States", "Turkey",
"Czech Republic", "DR Congo"). `canonicalize` maps the various spellings used by
Kalshi/Polymarket onto these canonical names so market contracts join to model
teams.

`SEED_ELO` are rough strength anchors used only by the *synthetic* demo generator
and the synthetic squad-value fallback. When real match history is loaded,
computed Elo and (if provided) real squad data replace them.
"""
from __future__ import annotations

# Rough strength anchors for the 48 qualifiers (demo / fallback only).
SEED_ELO: dict[str, float] = {
    "Argentina": 2100, "France": 2080, "Spain": 2070, "England": 2050,
    "Brazil": 2040, "Portugal": 2010, "Netherlands": 2000, "Belgium": 1980,
    "Germany": 1975, "Croatia": 1945, "Uruguay": 1940, "Colombia": 1930,
    "Morocco": 1925, "Switzerland": 1900, "United States": 1890, "Mexico": 1885,
    "Senegal": 1880, "Japan": 1875, "Ecuador": 1855, "Austria": 1850,
    "South Korea": 1840, "Canada": 1815, "Norway": 1825, "Australia": 1820,
    "Turkey": 1815, "Algeria": 1800, "Egypt": 1800, "Ivory Coast": 1795,
    "Iran": 1795, "Czech Republic": 1790, "Sweden": 1785, "Scotland": 1780,
    "Paraguay": 1770, "Tunisia": 1770, "Ghana": 1760, "Saudi Arabia": 1740,
    "Bosnia and Herzegovina": 1740, "DR Congo": 1740, "South Africa": 1730,
    "Uzbekistan": 1730, "Panama": 1730, "Cape Verde": 1720, "Qatar": 1700,
    "Jordan": 1700, "Iraq": 1690, "Haiti": 1680, "New Zealand": 1680,
    "Curaçao": 1660,
}

# OFFICIAL 2026 draw (pulled from market data; names canonicalized).
GROUPS_2026: dict[str, list[str]] = {
    "A": ["Mexico", "South Korea", "South Africa", "Czech Republic"],
    "B": ["Canada", "Qatar", "Bosnia and Herzegovina", "Switzerland"],
    "C": ["Scotland", "Brazil", "Haiti", "Morocco"],
    "D": ["Paraguay", "Turkey", "United States", "Australia"],
    "E": ["Curaçao", "Ecuador", "Germany", "Ivory Coast"],
    "F": ["Tunisia", "Japan", "Netherlands", "Sweden"],
    "G": ["New Zealand", "Iran", "Egypt", "Belgium"],
    "H": ["Cape Verde", "Uruguay", "Spain", "Saudi Arabia"],
    "I": ["Senegal", "Norway", "France", "Iraq"],
    "J": ["Algeria", "Jordan", "Argentina", "Austria"],
    "K": ["Colombia", "DR Congo", "Portugal", "Uzbekistan"],
    "L": ["England", "Ghana", "Croatia", "Panama"],
}

ALL_TEAMS_2026: list[str] = [t for teams in GROUPS_2026.values() for t in teams]

assert len(ALL_TEAMS_2026) == 48, f"expected 48 teams, got {len(ALL_TEAMS_2026)}"
assert len(set(ALL_TEAMS_2026)) == 48, "duplicate team in GROUPS_2026"

# Host nations (canonical names) — get a home bump in the simulator.
HOST_NATIONS = {"United States", "Mexico", "Canada"}

# Map external market spellings -> canonical (results.csv) names.
CANONICAL_NAME: dict[str, str] = {
    "USA": "United States",
    "US": "United States",
    "U.S.": "United States",
    "Turkiye": "Turkey",
    "Türkiye": "Turkey",
    "Czechia": "Czech Republic",
    "Korea Republic": "South Korea",
    "Republic of Korea": "South Korea",
    "the Democratic Republic of Congo": "DR Congo",
    "Democratic Republic of Congo": "DR Congo",
    "DR Congo": "DR Congo",
    "Congo DR": "DR Congo",
    "Cabo Verde": "Cape Verde",
    "Curacao": "Curaçao",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "IR Iran": "Iran",
}


def canonicalize(name: str) -> str:
    """Normalize a team name from any source to the canonical dataset spelling."""
    if name is None:
        return name
    n = name.strip()
    return CANONICAL_NAME.get(n, n)
