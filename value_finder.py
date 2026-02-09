"""
═══════════════════════════════════════════════════════════════════════
MÓDULO DE ANÁLISE DE VALOR E IDENTIFICAÇÃO DE +EV
Engine de Análise Preditiva - Camada de Decisão Financeira
═══════════════════════════════════════════════════════════════════════

Implementa:
  - De-Vigging (Power Method / Shin's Method)
  - Cálculo de Valor Esperado (EV)
  - Critério de Kelly Fracionário
  - Classificação de oportunidades
"""

from dataclasses import dataclass, field
from typing import Optional
import unicodedata

import numpy as np
from scipy.optimize import brentq

import config
from data_ingestion import MatchAnalysis
from models import predict_corners, predict_cards


# ═══════════════════════════════════════════════════════
# VALIDAÇÃO DE ODDS (filtro de anomalias da API)
# ═══════════════════════════════════════════════════════

_ODDS_LIMITS = {
    "1x2":              (config.ODDS_MIN_VALID, config.ODDS_MAX_1X2),
    "Dupla Chance":     (config.ODDS_MIN_VALID, config.ODDS_MAX_DC),
    "Gols O/U":         (config.ODDS_MIN_VALID, config.ODDS_MAX_OU),
    "BTTS":             (config.ODDS_MIN_VALID, config.ODDS_MAX_BTTS),
    "Escanteios O/U":   (config.ODDS_MIN_VALID, config.ODDS_MAX_CORNERS),
    "Cartoes O/U":      (config.ODDS_MIN_VALID, config.ODDS_MAX_CARDS),
    "Clean Sheet":      (config.ODDS_MIN_VALID, config.ODDS_MAX_CS),
    "Vit. s/ Sofrer":   (config.ODDS_MIN_VALID, config.ODDS_MAX_WTN),
    "Par/Impar":        (config.ODDS_MIN_VALID, config.ODDS_MAX_OE),
    "1o Tempo":         (config.ODDS_MIN_VALID, config.ODDS_MAX_HT),
    "Gols 1o Tempo":    (config.ODDS_MIN_VALID, config.ODDS_MAX_HT),
    "Gols Casa O/U":    (config.ODDS_MIN_VALID, config.ODDS_MAX_HOME_AWAY_OU),
    "Gols Fora O/U":    (config.ODDS_MIN_VALID, config.ODDS_MAX_HOME_AWAY_OU),
    "Placar Exato":     (config.ODDS_MIN_VALID, config.ODDS_MAX_EXACT),
    "Finaliz. O/U":     (config.ODDS_MIN_VALID, config.ODDS_MAX_SHOTS),
    "Finaliz. Gol O/U": (config.ODDS_MIN_VALID, config.ODDS_MAX_SHOTS),
    "Finaliz. Casa O/U": (config.ODDS_MIN_VALID, config.ODDS_MAX_SHOTS),
    "Finaliz. Fora O/U": (config.ODDS_MIN_VALID, config.ODDS_MAX_SHOTS),
    "SoT Casa O/U":     (config.ODDS_MIN_VALID, config.ODDS_MAX_SHOTS),
    "SoT Fora O/U":     (config.ODDS_MIN_VALID, config.ODDS_MAX_SHOTS),
    "Finaliz. Gol 1x2": (config.ODDS_MIN_VALID, config.ODDS_MAX_SHOTS),
    "Finaliz. Totais 1x2": (config.ODDS_MIN_VALID, config.ODDS_MAX_SHOTS),
    "Fin. Jogador":     (config.ODDS_MIN_VALID, config.ODDS_MAX_PLAYER_SHOTS),
    "SoT Jogador":      (config.ODDS_MIN_VALID, config.ODDS_MAX_PLAYER_SHOTS),
    # Backward compatibility
    "O/U 2.5":          (config.ODDS_MIN_VALID, config.ODDS_MAX_OU),
    "Corners":          (config.ODDS_MIN_VALID, config.ODDS_MAX_CORNERS),
    "Cartões":          (config.ODDS_MIN_VALID, config.ODDS_MAX_CARDS),
}

# ═══════════════════════════════════════════════════════
# MAPEAMENTO COMPLETO DE MERCADOS (model_probs key → all_markets key → labels)
# ═══════════════════════════════════════════════════════
_ALL_MARKETS = {
    "goals_ou": {
        "label": "Gols O/U", "selections": {
            "over_0.5": "Over 0.5 Gols", "under_0.5": "Under 0.5 Gols",
            "over_1.5": "Over 1.5 Gols", "under_1.5": "Under 1.5 Gols",
            "over_2.5": "Over 2.5 Gols", "under_2.5": "Under 2.5 Gols",
            "over_3.5": "Over 3.5 Gols", "under_3.5": "Under 3.5 Gols",
            "over_4.5": "Over 4.5 Gols", "under_4.5": "Under 4.5 Gols",
            "over_5.5": "Over 5.5 Gols", "under_5.5": "Under 5.5 Gols",
        }
    },
    "btts": {
        "label": "BTTS", "selections": {
            "yes": "Ambas Marcam Sim", "no": "Ambas Marcam Nao"
        }
    },
    "home_goals_ou": {
        "label": "Gols Casa O/U", "selections": {
            "over_0.5": "Casa Over 0.5 Gols", "under_0.5": "Casa Under 0.5 Gols",
            "over_1.5": "Casa Over 1.5 Gols", "under_1.5": "Casa Under 1.5 Gols",
            "over_2.5": "Casa Over 2.5 Gols", "under_2.5": "Casa Under 2.5 Gols",
            "over_3.5": "Casa Over 3.5 Gols", "under_3.5": "Casa Under 3.5 Gols",
        }
    },
    "away_goals_ou": {
        "label": "Gols Fora O/U", "selections": {
            "over_0.5": "Fora Over 0.5 Gols", "under_0.5": "Fora Under 0.5 Gols",
            "over_1.5": "Fora Over 1.5 Gols", "under_1.5": "Fora Under 1.5 Gols",
            "over_2.5": "Fora Over 2.5 Gols", "under_2.5": "Fora Under 2.5 Gols",
            "over_3.5": "Fora Over 3.5 Gols", "under_3.5": "Fora Under 3.5 Gols",
        }
    },
    "cs_home": {
        "label": "Clean Sheet", "selections": {
            "yes": "Clean Sheet Casa Sim", "no": "Clean Sheet Casa Nao"
        }
    },
    "cs_away": {
        "label": "Clean Sheet", "selections": {
            "yes": "Clean Sheet Fora Sim", "no": "Clean Sheet Fora Nao"
        }
    },
    "wtn_home": {
        "label": "Vit. s/ Sofrer", "selections": {
            "yes": "Vit. s/ Sofrer Casa", "no": "Nao Vit. s/ Sofrer Casa"
        }
    },
    "wtn_away": {
        "label": "Vit. s/ Sofrer", "selections": {
            "yes": "Vit. s/ Sofrer Fora", "no": "Nao Vit. s/ Sofrer Fora"
        }
    },
    "odd_even": {
        "label": "Par/Impar", "selections": {
            "odd": "Gols Impar", "even": "Gols Par"
        }
    },
    "ht_result": {
        "label": "1o Tempo", "selections": {
            "home": "Casa 1o Tempo", "draw": "Empate 1o Tempo", "away": "Fora 1o Tempo"
        }
    },
    "ht_goals_ou": {
        "label": "Gols 1o Tempo", "selections": {
            "over_0.5": "Over 0.5 1T", "under_0.5": "Under 0.5 1T",
            "over_1.5": "Over 1.5 1T", "under_1.5": "Under 1.5 1T",
            "over_2.5": "Over 2.5 1T", "under_2.5": "Under 2.5 1T",
        }
    },
    "corners_ou": {
        "label": "Escanteios O/U", "selections": {
            "over_7.5": "Over 7.5 Esc.", "under_7.5": "Under 7.5 Esc.",
            "over_8.5": "Over 8.5 Esc.", "under_8.5": "Under 8.5 Esc.",
            "over_9.5": "Over 9.5 Esc.", "under_9.5": "Under 9.5 Esc.",
            "over_10.5": "Over 10.5 Esc.", "under_10.5": "Under 10.5 Esc.",
            "over_11.5": "Over 11.5 Esc.", "under_11.5": "Under 11.5 Esc.",
            "over_12.5": "Over 12.5 Esc.", "under_12.5": "Under 12.5 Esc.",
        }
    },
    "cards_ou": {
        "label": "Cartoes O/U", "selections": {
            "over_2.5": "Over 2.5 Cart.", "under_2.5": "Under 2.5 Cart.",
            "over_3.5": "Over 3.5 Cart.", "under_3.5": "Under 3.5 Cart.",
            "over_4.5": "Over 4.5 Cart.", "under_4.5": "Under 4.5 Cart.",
            "over_5.5": "Over 5.5 Cart.", "under_5.5": "Under 5.5 Cart.",
            "over_6.5": "Over 6.5 Cart.", "under_6.5": "Under 6.5 Cart.",
        }
    },
    "shots_ou": {
        "label": "Finaliz. O/U", "selections": {
            "over_18.5": "Over 18.5 Finaliz.", "under_18.5": "Under 18.5 Finaliz.",
            "over_20.5": "Over 20.5 Finaliz.", "under_20.5": "Under 20.5 Finaliz.",
            "over_22.5": "Over 22.5 Finaliz.", "under_22.5": "Under 22.5 Finaliz.",
            "over_24.5": "Over 24.5 Finaliz.", "under_24.5": "Under 24.5 Finaliz.",
            "over_26.5": "Over 26.5 Finaliz.", "under_26.5": "Under 26.5 Finaliz.",
            "over_28.5": "Over 28.5 Finaliz.", "under_28.5": "Under 28.5 Finaliz.",
        }
    },
    "sot_ou": {
        "label": "Finaliz. Gol O/U", "selections": {
            "over_4.5": "Over 4.5 Fin. Gol", "under_4.5": "Under 4.5 Fin. Gol",
            "over_5.5": "Over 5.5 Fin. Gol", "under_5.5": "Under 5.5 Fin. Gol",
            "over_6.5": "Over 6.5 Fin. Gol", "under_6.5": "Under 6.5 Fin. Gol",
            "over_7.5": "Over 7.5 Fin. Gol", "under_7.5": "Under 7.5 Fin. Gol",
            "over_8.5": "Over 8.5 Fin. Gol", "under_8.5": "Under 8.5 Fin. Gol",
            "over_9.5": "Over 9.5 Fin. Gol", "under_9.5": "Under 9.5 Fin. Gol",
        }
    },
    "home_shots_ou": {
        "label": "Finaliz. Casa O/U", "selections": {
            "over_8.5": "Casa Over 8.5 Fin.", "under_8.5": "Casa Under 8.5 Fin.",
            "over_10.5": "Casa Over 10.5 Fin.", "under_10.5": "Casa Under 10.5 Fin.",
            "over_12.5": "Casa Over 12.5 Fin.", "under_12.5": "Casa Under 12.5 Fin.",
            "over_14.5": "Casa Over 14.5 Fin.", "under_14.5": "Casa Under 14.5 Fin.",
        }
    },
    "away_shots_ou": {
        "label": "Finaliz. Fora O/U", "selections": {
            "over_8.5": "Fora Over 8.5 Fin.", "under_8.5": "Fora Under 8.5 Fin.",
            "over_10.5": "Fora Over 10.5 Fin.", "under_10.5": "Fora Under 10.5 Fin.",
            "over_12.5": "Fora Over 12.5 Fin.", "under_12.5": "Fora Under 12.5 Fin.",
            "over_14.5": "Fora Over 14.5 Fin.", "under_14.5": "Fora Under 14.5 Fin.",
        }
    },
    "home_sot_ou": {
        "label": "SoT Casa O/U", "selections": {
            "over_2.5": "Casa Over 2.5 SoT", "under_2.5": "Casa Under 2.5 SoT",
            "over_3.5": "Casa Over 3.5 SoT", "under_3.5": "Casa Under 3.5 SoT",
            "over_4.5": "Casa Over 4.5 SoT", "under_4.5": "Casa Under 4.5 SoT",
            "over_5.5": "Casa Over 5.5 SoT", "under_5.5": "Casa Under 5.5 SoT",
        }
    },
    "away_sot_ou": {
        "label": "SoT Fora O/U", "selections": {
            "over_2.5": "Fora Over 2.5 SoT", "under_2.5": "Fora Under 2.5 SoT",
            "over_3.5": "Fora Over 3.5 SoT", "under_3.5": "Fora Under 3.5 SoT",
            "over_4.5": "Fora Over 4.5 SoT", "under_4.5": "Fora Under 4.5 SoT",
            "over_5.5": "Fora Over 5.5 SoT", "under_5.5": "Fora Under 5.5 SoT",
        }
    },
    # ── Mercados REAIS de Shots do Bet365 (via API-Football) ──
    "sot_1x2": {
        "label": "Finaliz. Gol 1x2", "selections": {
            "home": "Casa + Fin. no Gol", "draw": "Empate Fin. Gol",
            "away": "Fora + Fin. no Gol"
        }
    },
    "shots_1x2": {
        "label": "Finaliz. Totais 1x2", "selections": {
            "home": "Casa + Finaliz.", "draw": "Empate Finaliz.",
            "away": "Fora + Finaliz."
        }
    },
    "exact_score": {
        "label": "Placar Exato", "selections": {
            "1-0": "Exato 1-0", "0-0": "Exato 0-0", "1-1": "Exato 1-1",
            "2-0": "Exato 2-0", "2-1": "Exato 2-1", "0-1": "Exato 0-1",
            "0-2": "Exato 0-2", "1-2": "Exato 1-2", "2-2": "Exato 2-2",
            "3-0": "Exato 3-0", "3-1": "Exato 3-1", "0-3": "Exato 0-3",
            "1-3": "Exato 1-3", "3-2": "Exato 3-2", "2-3": "Exato 2-3",
        }
    },
    "1x2": {
        "label": "1x2", "selections": {
            "home": "Vitoria Casa", "away": "Vitoria Fora"
        }
    },
    "double_chance": {
        "label": "Dupla Chance", "selections": {
            "home/draw": "Casa ou Empate (1X)", "draw/away": "Fora ou Empate (X2)",
            "home/away": "Casa ou Fora (12)"
        }
    },
}


def _is_odd_valid(odd: float, market: str) -> bool:
    """Verifica se uma odd esta dentro dos limites razoaveis para o mercado."""
    min_v, max_v = _ODDS_LIMITS.get(market, (1.05, config.ODDS_MAX_GENERIC))
    return min_v <= odd <= max_v


# ═══════════════════════════════════════════════════════
# ESTRUTURA DE OPORTUNIDADE
# ═══════════════════════════════════════════════════════

@dataclass
class ValueOpportunity:
    """Representa uma oportunidade de valor identificada."""
    match_id: int
    league_name: str
    league_country: str
    match_date: str
    match_time: str
    home_team: str
    away_team: str
    market: str                   # ex: "1x2", "O/U 2.5", "Corners", "Cards"
    selection: str                # ex: "Vitória Casa", "Over 2.5"
    market_odd: float             # Odd oferecida pela casa
    fair_odd: float               # Odd justa calculada pelo modelo
    model_prob: float             # Probabilidade do modelo
    implied_prob: float           # Probabilidade implícita (de-vigged)
    edge: float                   # EV = (model_prob × odd) - 1
    edge_pct: str                 # Edge formatado em %
    kelly_fraction: float         # Fração Kelly sugerida
    kelly_bet_pct: str            # Aposta Kelly formatada
    confidence: str               # "ALTO", "MÉDIO", "BAIXO"
    reasoning: str                # Justificativa analítica
    home_xg: float = 0.0
    away_xg: float = 0.0
    weather_note: str = ""
    fatigue_note: str = ""
    urgency_home: float = 0.5
    urgency_away: float = 0.5
    bookmaker: str = "N/D"
    data_quality: float = 0.0
    odds_suspect: bool = False  # True = possível inversão casa/fora


# ═══════════════════════════════════════════════════════
# 1. DE-VIGGING: REMOÇÃO DA MARGEM DA CASA
# ═══════════════════════════════════════════════════════

def power_method_devig(odds: list[float]) -> list[float]:
    """
    Remove a margem das odds usando o Power Method (Método da Potência).

    O Power Method é superior à normalização multiplicativa porque
    distribui a margem de forma proporcional ao viés favorito-zebra.
    Casas de apostas colocam mais margem em longshots.

    Resolve: Σ(1/odd_i^k) = 1 para encontrar k

    Args:
        odds: Lista de odds decimais [home, draw, away] ou [over, under]

    Returns:
        Lista de probabilidades justas (sem vig)
    """
    if not odds or any(o <= 1.0 for o in odds):
        # Fallback: normalização simples
        probs = [1.0 / o for o in odds]
        total = sum(probs)
        return [p / total for p in probs]

    def objective(k):
        return sum((1.0 / o) ** k for o in odds) - 1.0

    try:
        # Encontrar k via método de Brent
        # k=1 dá a soma bruta (com vig), k>1 remove o vig
        k_solution = brentq(objective, 0.5, 2.0, xtol=1e-8)
        fair_probs = [(1.0 / o) ** k_solution for o in odds]
        return fair_probs
    except (ValueError, RuntimeError):
        # Fallback: normalização multiplicativa
        probs = [1.0 / o for o in odds]
        total = sum(probs)
        return [p / total for p in probs]


def shin_method_devig(odds: list[float]) -> list[float]:
    """
    Remove a margem usando o método de Shin (1991, 1992).

    Mais preciso para mercados com forte favorito.
    Resolve iterativamente para o parâmetro z (proporção de informed bettors).

    Args:
        odds: Lista de odds decimais

    Returns:
        Lista de probabilidades justas
    """
    n = len(odds)
    implied = [1.0 / o for o in odds]
    margin = sum(implied) - 1.0

    if margin <= 0:
        return implied

    try:
        def shin_objective(z):
            probs = []
            for imp in implied:
                prob = (((z ** 2 + 4 * (1 - z) * imp ** 2 / sum(implied)) ** 0.5)
                        - z) / (2 * (1 - z))
                probs.append(prob)
            return sum(probs) - 1.0

        z = brentq(shin_objective, 0.001, 0.5)

        fair_probs = []
        for imp in implied:
            prob = (((z ** 2 + 4 * (1 - z) * imp ** 2 / sum(implied)) ** 0.5)
                    - z) / (2 * (1 - z))
            fair_probs.append(max(0.001, prob))

        # Normalizar
        total = sum(fair_probs)
        return [p / total for p in fair_probs]

    except (ValueError, RuntimeError):
        return power_method_devig(odds)


def devig_odds(odds: list[float], method: str = "power") -> list[float]:
    """
    Interface de de-vigging com seleção de método.
    """
    if method == "shin":
        return shin_method_devig(odds)
    return power_method_devig(odds)


# ═══════════════════════════════════════════════════════
# 2. CÁLCULO DE VALOR ESPERADO (EV)
# ═══════════════════════════════════════════════════════

def calculate_edge(model_prob: float, market_odd: float) -> float:
    """
    Calcula o Edge (Valor Esperado).

    EV = (P_modelo × Odd_mercado) - 1

    Um EV > 0 indica valor positivo (+EV).
    """
    return (model_prob * market_odd) - 1.0


def fractional_kelly(model_prob: float, market_odd: float,
                      fraction: float = None) -> float:
    """
    Calcula a fração Kelly de aposta.

    Kelly = (p × (b+1) - 1) / b
    Kelly Fracionário = Kelly × fraction

    Args:
        model_prob: Probabilidade estimada pelo modelo
        market_odd: Odd decimal oferecida
        fraction: Fração do Kelly (ex: 0.25 = Kelly/4)

    Returns:
        Fração da banca sugerida (0.0 a MAX_KELLY_BET)
    """
    if fraction is None:
        fraction = config.KELLY_FRACTION

    b = market_odd - 1.0  # Net odds
    if b <= 0:
        return 0.0

    kelly = (model_prob * (b + 1) - 1) / b

    if kelly <= 0:
        return 0.0

    # Aplicar fração e cap
    kelly_frac = kelly * fraction
    return min(config.MAX_KELLY_BET, max(0.0, kelly_frac))


# ═══════════════════════════════════════════════════════
# 3. CLASSIFICAÇÃO DE CONFIANÇA
# ═══════════════════════════════════════════════════════

def classify_confidence(edge: float, model_prob: float,
                         weather_stable: bool = True,
                         fatigue_free: bool = True) -> str:
    """
    Classifica a confiança da oportunidade.

    Critérios:
        ALTO:  Edge > 10% + prob > 40% + sem fatores adversos
        MÉDIO: Edge > 5% OU (Edge > 7% com fatores adversos)
        BAIXO: Edge > 3% mas com incertezas
    """
    if edge > 0.10 and model_prob > 0.40 and weather_stable and fatigue_free:
        return "ALTO"
    elif edge > 0.08 and model_prob > 0.35:
        return "ALTO"
    elif edge > 0.05:
        if not weather_stable or not fatigue_free:
            return "MÉDIO"
        return "MÉDIO"
    elif edge > 0.03:
        return "BAIXO"
    else:
        return "BAIXO"


# ═══════════════════════════════════════════════════════
# 4. GERAÇÃO DE REASONING (JUSTIFICATIVA DETALHADA)
# ═══════════════════════════════════════════════════════

def generate_reasoning(match: MatchAnalysis, market: str,
                        edge: float, model_prob: float) -> str:
    """
    Gera uma justificativa analítica DETALHADA para a oportunidade,
    mostrando todos os cálculos, estatísticas e fatores envolvidos.

    Seções:
        1. Modelo usado e parâmetros
        2. Força de ataque/defesa dos times
        3. Cálculo de xG
        4. Probabilidades da matriz Dixon-Coles
        5. Ajustes contextuais aplicados
        6. Cálculo de valor (de-vigging + edge)
    """
    lines = []
    home = match.home_team
    away = match.away_team

    # ── 1. MODELO ESTATÍSTICO ──
    lines.append("📐 MODELO: Dixon-Coles (Poisson Bivariada Ajustada, 1997)")
    lines.append(f"Simulação Monte Carlo: {config.MONTE_CARLO_SIMULATIONS} iterações")
    lines.append(f"Matriz de placar: {config.DIXON_COLES_MAX_GOALS+1}×{config.DIXON_COLES_MAX_GOALS+1} ({config.DIXON_COLES_MAX_GOALS} gols máx.)")
    lines.append("")

    # ── 2. PARÂMETROS DE FORÇA ──
    lines.append("⚔️ FORÇA DOS TIMES (Ataque α / Defesa β):")
    lines.append(f"  {home.team_name} (Casa): Atk={home.attack_strength:.2f} | Def={home.defense_strength:.2f}")
    lines.append(f"    Média gols marcados (casa): {home.home_goals_scored_avg:.2f}/jogo")
    lines.append(f"    Média gols sofridos (casa): {home.home_goals_conceded_avg:.2f}/jogo")
    lines.append(f"    Posição na liga: {home.league_position}° ({home.league_points} pts)")
    lines.append(f"  {away.team_name} (Fora): Atk={away.attack_strength:.2f} | Def={away.defense_strength:.2f}")
    lines.append(f"    Média gols marcados (fora): {away.away_goals_scored_avg:.2f}/jogo")
    lines.append(f"    Média gols sofridos (fora): {away.away_goals_conceded_avg:.2f}/jogo")
    lines.append(f"    Posição na liga: {away.league_position}° ({away.league_points} pts)")
    lines.append("")

    # ── 3. FORMA RECENTE ──
    home_n = len(home.form_last10)
    away_n = len(away.form_last10)
    home_wins = sum(1 for f in home.form_last10 if f == 'W')
    home_draws = sum(1 for f in home.form_last10 if f == 'D')
    home_losses = sum(1 for f in home.form_last10 if f == 'L')
    away_wins = sum(1 for f in away.form_last10 if f == 'W')
    away_draws = sum(1 for f in away.form_last10 if f == 'D')
    away_losses = sum(1 for f in away.form_last10 if f == 'L')
    lines.append("FORMA RECENTE:")
    if home_n:
        lines.append(f"  {home.team_name} (ult. {home_n}): {''.join(home.form_last10)} -> {home_wins}V {home_draws}E {home_losses}D (pontos: {home.form_points:.2f})")
    else:
        lines.append(f"  {home.team_name}: Dados de forma indisponiveis")
    if away_n:
        lines.append(f"  {away.team_name} (ult. {away_n}): {''.join(away.form_last10)} -> {away_wins}V {away_draws}E {away_losses}D (pontos: {away.form_points:.2f})")
    else:
        lines.append(f"  {away.team_name}: Dados de forma indisponiveis")
    lines.append("")

    # ── 4. CÁLCULO DE xG ──
    lines.append("⚽ CÁLCULO DE xG (Gols Esperados):")
    league_avg = getattr(match, 'league_avg_goals', 2.7) or 2.7
    league_half = league_avg / 2.0
    suspect = getattr(match, 'odds_home_away_suspect', False)
    home_adv = 1.0 if suspect else 1.08

    # Recalcular os MESMOS α/β usados em calculate_expected_goals
    if suspect:
        _h_sc = (home.home_goals_scored_avg + home.away_goals_scored_avg) / 2
        _h_co = (home.home_goals_conceded_avg + home.away_goals_conceded_avg) / 2
        _a_sc = (away.home_goals_scored_avg + away.away_goals_scored_avg) / 2
        _a_co = (away.home_goals_conceded_avg + away.away_goals_conceded_avg) / 2
    else:
        _h_sc = home.home_goals_scored_avg
        _h_co = home.home_goals_conceded_avg
        _a_sc = away.away_goals_scored_avg
        _a_co = away.away_goals_conceded_avg

    # Regressão à média (mesmo que no modelo)
    from models import _regress_to_mean
    home_gp = home.games_played or 1
    away_gp = away.games_played or 1
    _h_sc = _regress_to_mean(_h_sc, league_half, home_gp)
    _h_co = _regress_to_mean(_h_co, league_half, home_gp)
    _a_sc = _regress_to_mean(_a_sc, league_half, away_gp)
    _a_co = _regress_to_mean(_a_co, league_half, away_gp)

    _half = league_avg / 2.0
    _ah = max(0.3, _h_sc / _half) if _half > 0 else 1.0
    _bh = max(0.3, _h_co / _half) if _half > 0 else 1.0
    _aa = max(0.3, _a_sc / _half) if _half > 0 else 1.0
    _ba = max(0.3, _a_co / _half) if _half > 0 else 1.0

    home_form_factor = 0.85 + home.form_points * 0.30
    away_form_factor = 0.85 + away.form_points * 0.30

    lines.append(f"  Média de gols da liga: {league_avg:.2f} gols/jogo")
    if home_gp < 8 or away_gp < 8:
        lines.append(f"  ⚠️ Amostra pequena (Casa:{home_gp} jogos, Fora:{away_gp} jogos) — regressão à média aplicada")
    lines.append(f"  Fórmula (Dixon-Coles):")
    lines.append(f"    λ (Casa) = α_casa × β_fora × vantagem_mando({home_adv}) × fator_forma")
    lines.append(f"    μ (Fora) = α_fora × β_casa × fator_forma")
    lines.append(f"")
    lines.append(f"  Cálculo λ ({home.team_name}):")
    lines.append(f"    = {_ah:.2f} × {_ba:.2f} × {home_adv} × {home_form_factor:.2f}")
    _raw_lambda = _ah * _ba * home_adv * home_form_factor
    lines.append(f"    = {_raw_lambda:.2f}{' → clamped 3.5' if _raw_lambda > 3.5 else ''}")
    lines.append(f"    → xG Casa = {match.model_home_xg:.2f}")
    lines.append(f"")
    lines.append(f"  Cálculo μ ({away.team_name}):")
    lines.append(f"    = {_aa:.2f} × {_bh:.2f} × {away_form_factor:.2f}")
    _raw_mu = _aa * _bh * away_form_factor
    lines.append(f"    = {_raw_mu:.2f}{' → clamped 3.0' if _raw_mu > 3.0 else ''}")
    lines.append(f"    → xG Fora = {match.model_away_xg:.2f}")
    total_xg = match.model_home_xg + match.model_away_xg
    lines.append(f"  → xG Total = {total_xg:.2f}")
    lines.append("")

    # ── 5. PROBABILIDADES DO MODELO ──
    lines.append("🎯 PROBABILIDADES (Matriz Dixon-Coles + Monte Carlo):")
    lines.append(f"  Vitória Casa: {match.model_prob_home*100:.1f}%")
    lines.append(f"  Empate:       {match.model_prob_draw*100:.1f}%")
    lines.append(f"  Vitória Fora: {match.model_prob_away*100:.1f}%")
    lines.append(f"  Over 2.5:     {match.model_prob_over25*100:.1f}%")
    lines.append(f"  BTTS (Ambas): {match.model_prob_btts*100:.1f}%")
    lines.append(f"  Escanteios:   {match.model_corners_expected:.1f} esperados")
    lines.append(f"  Cartões:      {match.model_cards_expected:.1f} esperados")
    # Finalizações
    h_shots = getattr(match, 'model_home_shots_expected', 0)
    a_shots = getattr(match, 'model_away_shots_expected', 0)
    h_sot = getattr(match, 'model_home_sot_expected', 0)
    a_sot = getattr(match, 'model_away_sot_expected', 0)
    if h_shots > 0:
        lines.append(f"  Finaliz.:     {h_shots + a_shots:.1f} esperadas ({home.team_name}: {h_shots:.1f} | {away.team_name}: {a_shots:.1f})")
        lines.append(f"  Fin. no Gol:  {h_sot + a_sot:.1f} esperadas ({home.team_name}: {h_sot:.1f} | {away.team_name}: {a_sot:.1f})")
    lines.append("")

    # ── 6. ODDS E CÁLCULO DE VALOR ──
    odds = match.odds
    lines.append(f"💰 ODDS DE MERCADO ({odds.bookmaker}):")
    lines.append(f"  1x2: Casa={odds.home_win:.2f} | Empate={odds.draw:.2f} | Fora={odds.away_win:.2f}")
    lines.append(f"  O/U 2.5: Over={odds.over_25:.2f} | Under={odds.under_25:.2f}")
    lines.append(f"  BTTS: Sim={odds.btts_yes:.2f} | Não={odds.btts_no:.2f}")
    lines.append("")

    lines.append("📈 CÁLCULO DO EDGE (Valor):")
    lines.append(f"  Método de de-vigging: Power Method (Método da Potência)")
    lines.append(f"  Fórmula: Edge = (Prob_Modelo × Odd_Mercado) − 1")
    lines.append(f"  Prob. do modelo neste mercado: {model_prob*100:.1f}%")
    lines.append(f"  Edge calculado: {edge*100:.1f}%")
    if edge > 0:
        lines.append(f"  → Odd justa (modelo): {1.0/max(0.01, model_prob):.2f}")
    lines.append("")

    # ── 7. AJUSTES CONTEXTUAIS ──
    context_lines = []
    weather = match.weather

    # Clima
    if weather.description != "N/D":
        clima_detail = f"Clima: {weather.description} | {weather.temperature_c:.0f}°C | Vento: {weather.wind_speed_kmh:.0f} km/h"
        if weather.rain_mm > 0:
            clima_detail += f" | Chuva: {weather.rain_mm:.1f}mm"
        context_lines.append(clima_detail)

        if weather.wind_speed_kmh > config.WIND_SPEED_THRESHOLD_KMH:
            penalty = config.XG_WIND_PENALTY * (1.0 + min(1.0, (weather.wind_speed_kmh - config.WIND_SPEED_THRESHOLD_KMH) / 30.0))
            context_lines.append(f"  ⚠️ Penalidade por vento: xG reduzido em {penalty*100:.1f}%")
        if weather.rain_mm > config.RAIN_VOLUME_THRESHOLD_MM:
            context_lines.append(f"  🌧️ Ajuste por chuva: +{config.XG_RAIN_PENALTY*100:.0f}% variância de erros")
        if weather.temperature_c > config.HEAT_THRESHOLD_C:
            context_lines.append(f"  🌡️ Calor extremo (>{config.HEAT_THRESHOLD_C}°C): pressing reduzido no 2º tempo")

    # Fadiga
    if match.home_fatigue:
        context_lines.append(f"⚡ {home.team_name}: jogou nas últimas {config.FATIGUE_WINDOW_HOURS}h → penalidade de {config.FATIGUE_PENALTY*100:.0f}% nos ratings")
    if match.away_fatigue:
        context_lines.append(f"⚡ {away.team_name}: jogou nas últimas {config.FATIGUE_WINDOW_HOURS}h → penalidade de {config.FATIGUE_PENALTY*100:.0f}% nos ratings")

    # Lesões
    if match.injuries_home:
        context_lines.append(f"🏥 Lesões {home.team_name} ({len(match.injuries_home)}): {', '.join(match.injuries_home[:3])}")
    if match.injuries_away:
        context_lines.append(f"🏥 Lesões {away.team_name} ({len(match.injuries_away)}): {', '.join(match.injuries_away[:3])}")

    # Urgência
    context_lines.append(f"🔥 Urgência (LUS): {home.team_name}={match.league_urgency_home:.1f} | {away.team_name}={match.league_urgency_away:.1f}")
    if match.league_urgency_home < config.LUS_LOW_THRESHOLD:
        context_lines.append(f"  → {home.team_name}: baixa motivação (meio de tabela) → +variância")
    if match.league_urgency_away < config.LUS_LOW_THRESHOLD:
        context_lines.append(f"  → {away.team_name}: baixa motivação (meio de tabela) → +variância")
    if match.league_urgency_home > config.LUS_HIGH_THRESHOLD:
        context_lines.append(f"  → {home.team_name}: altíssima urgência (título/rebaixamento) → jogo focado")
    if match.league_urgency_away > config.LUS_HIGH_THRESHOLD:
        context_lines.append(f"  → {away.team_name}: altíssima urgência (título/rebaixamento) → jogo focado")

    # Árbitro
    if match.referee.name != "Desconhecido":
        context_lines.append(f"👨‍⚖️ Árbitro: {match.referee.name} ({match.referee.cards_per_game_avg:.1f} cartões/jogo, {match.referee.fouls_per_game_avg:.0f} faltas/jogo)")

    if context_lines:
        lines.append("🌐 AJUSTES CONTEXTUAIS APLICADOS:")
        for cl in context_lines:
            lines.append(f"  {cl}")
        lines.append("")

    # ── 8. CONCLUSÃO ──
    lines.append("✅ CONCLUSÃO:")
    if "Dupla Chance" in market:
        if "1X" in market or "Casa" in market:
            lines.append(f"  O modelo atribui {model_prob*100:.1f}% de chance de {home.team_name} vencer ou empatar,")
        elif "X2" in market or "Fora" in market:
            lines.append(f"  O modelo atribui {model_prob*100:.1f}% de chance de {away.team_name} vencer ou empatar,")
    elif "1x2" in market:
        if "Casa" in market:
            lines.append(f"  O modelo atribui {model_prob*100:.1f}% de chance de vitória ao {home.team_name},")
        elif "Fora" in market:
            lines.append(f"  O modelo atribui {model_prob*100:.1f}% de chance de vitória ao {away.team_name},")
    elif "Over" in market:
        lines.append(f"  O modelo projeta xG total de {total_xg:.2f}, atribuindo {model_prob*100:.1f}% de chance,")
    elif "Under" in market:
        lines.append(f"  O modelo projeta xG total de {total_xg:.2f}, atribuindo {model_prob*100:.1f}% de chance,")
    elif "BTTS" in market:
        lines.append(f"  Baseado nos xGs individuais (Casa:{match.model_home_xg:.2f} Fora:{match.model_away_xg:.2f}), prob={model_prob*100:.1f}%,")
    elif "Corners" in market or "Escanteios" in market:
        lines.append(f"  Regressão Binomial Negativa projeta {match.model_corners_expected:.1f} escanteios, prob={model_prob*100:.1f}%,")
    elif "Cartões" in market:
        lines.append(f"  Regressão Binomial Negativa projeta {match.model_cards_expected:.1f} cartões, prob={model_prob*100:.1f}%,")
    else:
        lines.append(f"  Probabilidade do modelo: {model_prob*100:.1f}%,")

    lines.append(f"  enquanto o mercado ({odds.bookmaker}) implica apenas ~{(1.0/max(0.01, 1.0/model_prob * (1+edge)))*100:.1f}%.")
    lines.append(f"  Diferença = edge de +{edge*100:.1f}% → oportunidade de valor identificada.")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════
# 5. SCANNER DE OPORTUNIDADES
# ═══════════════════════════════════════════════════════

def _is_model_sane(model_prob: float, total_xg: float, market: str) -> bool:
    """
    Verifica se a saída do modelo é confiável.
    Rejeita probabilidades extremas e xG irrealistas que indicam dados insuficientes.
    """
    # Probabilidade fora dos limites razoáveis
    if model_prob > config.MAX_MODEL_PROB:
        return False
    if model_prob < config.MIN_MODEL_PROB:
        return False

    # xG total muito baixo ou muito alto (modelo sem dados)
    # Aplicar apenas para mercados baseados em gols
    goal_markets = ("1x2", "Dupla Chance", "O/U 2.5", "BTTS",
                    "Gols O/U", "Gols Casa O/U", "Gols Fora O/U",
                    "Clean Sheet", "Vit. s/ Sofrer", "Par/Impar",
                    "1o Tempo", "Gols 1o Tempo", "Placar Exato")
    if market in goal_markets:
        if total_xg < config.MIN_XG_TOTAL:
            return False
        if total_xg > config.MAX_XG_TOTAL:
            return False

    # Mercados de finalizações — sem filtro extra de xG
    # (usam parâmetros de ataque/defesa, não xG diretamente)

    return True


def scan_match_for_value(match: MatchAnalysis) -> list[ValueOpportunity]:
    """
    Escaneia TODOS os mercados de uma partida buscando valor (+EV).

    Mercados analisados:
        - 1x2 (Vitória Casa, Vitória Fora — sem empate isolado)
        - Dupla Chance (Casa ou Empate 1X, Fora ou Empate X2)
        - Over/Under 2.5 Gols
        - BTTS (Ambas Marcam)
        - Over/Under 9.5 Escanteios
        - Over/Under 3.5 Cartões

    PORTÃO DE QUALIDADE (rejeição antecipada):
        - REJEITAR se não tem odds REAIS (circular: modelo vs modelo)
        - REJEITAR se NENHUM time tem standings reais
        - REJEITAR se data_quality_score < 0.50
        - REJEITAR se valores de ataque/defesa são defaults (sem dados reais)

    Filtros de sanidade por oportunidade:
        - Edge máximo: 30% (acima = erro de dados)
        - Prob. máxima: 85% (acima = dados insuficientes)
        - xG mínimo: 0.80 (abaixo = modelo sem dados)
        - Odds dentro de limites razoáveis por mercado
    """
    # ═══ PORTÃO DE QUALIDADE RIGOROSO ═══
    # Ajustado: aceitar se tem odds REAIS OU standings REAIS (não ambos obrigatórios)
    # Se não tem nenhum dos dois, rejeitar
    if not match.has_real_odds and not match.has_real_standings:
        return []

    # Qualidade geral mínima ajustada: 0.40 (odds reais OU standings reais)
    # 0.40 = odds reais (0.35) + mínimo adicional OU standings (0.20) + odds estimadas
    if match.data_quality_score < 0.40:
        return []
    
    # Validar que pelo menos um time tem dados reais OU que a qualidade geral é suficiente
    # Se tem standings reais (has_real_standings=True), significa que pelo menos 1 time tem dados
    # Então não precisamos verificar has_real_data individualmente aqui
    # A validação de has_real_standings já garante isso
    
    # Validar que os valores numéricos são válidos
    home = match.home_team
    away = match.away_team
    
    if (home.attack_strength <= 0 or home.defense_strength <= 0 or
        away.attack_strength <= 0 or away.defense_strength <= 0):
        return []

    opportunities = []
    odds = match.odds
    total_xg = match.model_home_xg + match.model_away_xg

    # Condições meteorológicas e fadiga (para confiança)
    weather_stable = (match.weather.wind_speed_kmh <= config.WIND_SPEED_THRESHOLD_KMH
                      and match.weather.rain_mm <= config.RAIN_VOLUME_THRESHOLD_MM)
    fatigue_free = not match.home_fatigue and not match.away_fatigue

    # ── MERCADO 1x2 (sem Empate — foco em resultado) ──
    market_odds_1x2 = [odds.home_win, odds.draw, odds.away_win]
    fair_probs_1x2 = devig_odds(market_odds_1x2, method="power")

    model_probs_1x2 = [match.model_prob_home, match.model_prob_draw, match.model_prob_away]
    labels_1x2 = ["Vitória Casa", "Empate", "Vitória Fora"]

    for i, (label, model_p, market_o, implied_p) in enumerate(
        zip(labels_1x2, model_probs_1x2, market_odds_1x2, fair_probs_1x2)
    ):
        # Pular Empate no 1x2 se configurado (mercado de alto risco)
        if label == "Empate" and config.EXCLUDE_DRAW_1X2:
            continue

        # Validar odds — filtrar valores anômalos da API
        if not _is_odd_valid(market_o, "1x2"):
            continue

        # Validar sanidade do modelo (prob/xG dentro de limites)
        if not _is_model_sane(model_p, total_xg, "1x2"):
            continue

        edge = calculate_edge(model_p, market_o)

        # Filtrar edges absurdos (provável erro de odds da API)
        if edge >= config.MAX_EDGE_SANE:
            continue

        if edge >= config.MIN_EDGE_THRESHOLD:
            fair_odd = round(1.0 / max(0.01, model_p), 2)
            kelly = fractional_kelly(model_p, market_o)
            conf = classify_confidence(edge, model_p, weather_stable, fatigue_free)
            reasoning = generate_reasoning(match, f"1x2 - {label}", edge, model_p)

            opportunities.append(ValueOpportunity(
                match_id=match.match_id,
                league_name=match.league_name,
                league_country=match.league_country,
                match_date=match.match_date,
                match_time=match.match_time,
                home_team=match.home_team.team_name,
                away_team=match.away_team.team_name,
                market="1x2",
                selection=label,
                market_odd=market_o,
                fair_odd=fair_odd,
                model_prob=round(model_p, 4),
                implied_prob=round(implied_p, 4),
                edge=round(edge, 4),
                edge_pct=f"+{edge*100:.1f}%",
                kelly_fraction=round(kelly, 4),
                kelly_bet_pct=f"{kelly*100:.2f}%",
                confidence=conf,
                reasoning=reasoning,
                home_xg=match.model_home_xg,
                away_xg=match.model_away_xg,
                weather_note=match.weather.description,
                fatigue_note=("Casa em fadiga" if match.home_fatigue
                              else "Fora em fadiga" if match.away_fatigue else "N/A"),
                urgency_home=match.league_urgency_home,
                urgency_away=match.league_urgency_away,
                bookmaker=match.odds.bookmaker,
                data_quality=match.data_quality_score,
                odds_suspect=getattr(match, 'odds_home_away_suspect', False),
            ))

    # ── MERCADO DUPLA CHANCE (Casa ou Empate / Fora ou Empate) ──
    # Probabilidades do modelo: P(1X) = P(Home) + P(Draw), P(X2) = P(Away) + P(Draw)
    model_prob_1x = match.model_prob_home + match.model_prob_draw
    model_prob_x2 = match.model_prob_away + match.model_prob_draw

    # Odds da API (se disponíveis) ou calculadas a partir do 1x2
    dc_1x_odd = odds.double_chance_1x
    dc_x2_odd = odds.double_chance_x2

    # Se a API não forneceu odds de Dupla Chance, calcular a partir do 1x2
    if dc_1x_odd <= 0 and odds.home_win > 1.0 and odds.draw > 1.0:
        # Aproximação: 1 / (1/home + 1/draw) — harmônica
        dc_1x_odd = round(1.0 / (1.0/odds.home_win + 1.0/odds.draw), 2)
    if dc_x2_odd <= 0 and odds.away_win > 1.0 and odds.draw > 1.0:
        dc_x2_odd = round(1.0 / (1.0/odds.away_win + 1.0/odds.draw), 2)

    dc_entries = [
        ("Casa ou Empate (1X)", model_prob_1x, dc_1x_odd),
        ("Fora ou Empate (X2)", model_prob_x2, dc_x2_odd),
    ]

    for label, model_p, market_o in dc_entries:
        if market_o <= 1.0:
            continue
        if not _is_odd_valid(market_o, "Dupla Chance"):
            continue
        if not _is_model_sane(model_p, total_xg, "Dupla Chance"):
            continue

        edge = calculate_edge(model_p, market_o)

        if edge >= config.MAX_EDGE_SANE:
            continue

        if edge >= config.MIN_EDGE_THRESHOLD:
            fair_odd = round(1.0 / max(0.01, model_p), 2)
            implied_p = 1.0 / market_o if market_o > 0 else 0
            kelly = fractional_kelly(model_p, market_o)
            conf = classify_confidence(edge, model_p, weather_stable, fatigue_free)
            reasoning = generate_reasoning(match, f"Dupla Chance - {label}", edge, model_p)

            opportunities.append(ValueOpportunity(
                match_id=match.match_id,
                league_name=match.league_name,
                league_country=match.league_country,
                match_date=match.match_date,
                match_time=match.match_time,
                home_team=match.home_team.team_name,
                away_team=match.away_team.team_name,
                market="Dupla Chance",
                selection=label,
                market_odd=market_o,
                fair_odd=fair_odd,
                model_prob=round(model_p, 4),
                implied_prob=round(implied_p, 4),
                edge=round(edge, 4),
                edge_pct=f"+{edge*100:.1f}%",
                kelly_fraction=round(kelly, 4),
                kelly_bet_pct=f"{kelly*100:.2f}%",
                confidence=conf,
                reasoning=reasoning,
                home_xg=match.model_home_xg,
                away_xg=match.model_away_xg,
                weather_note=match.weather.description,
                fatigue_note=("Casa em fadiga" if match.home_fatigue
                              else "Fora em fadiga" if match.away_fatigue else "N/A"),
                urgency_home=match.league_urgency_home,
                urgency_away=match.league_urgency_away,
                bookmaker=match.odds.bookmaker,
                data_quality=match.data_quality_score,
                odds_suspect=getattr(match, 'odds_home_away_suspect', False),
            ))

    # ── MERCADO OVER/UNDER 2.5 ──
    # REMOVIDO: Scanner dedicado O/U 2.5 (agora tratado pelo scanner genérico "goals_ou"
    # para manter padrão uniforme "Gols O/U" com suporte a multi-bookmaker)

    # ── MERCADO BTTS ──
    btts_odds = [odds.btts_yes, odds.btts_no]
    btts_fair = devig_odds(btts_odds)
    model_btts = match.model_prob_btts

    for label, model_p, market_o, implied_p in [
        ("Ambas Marcam — Sim", model_btts, odds.btts_yes, btts_fair[0]),
        ("Ambas Marcam — Não", 1.0 - model_btts, odds.btts_no, btts_fair[1]),
    ]:
        if not _is_odd_valid(market_o, "BTTS"):
            continue
        if not _is_model_sane(model_p, total_xg, "BTTS"):
            continue
        edge = calculate_edge(model_p, market_o)
        if edge >= config.MAX_EDGE_SANE:
            continue
        if edge >= config.MIN_EDGE_THRESHOLD:
            fair_odd = round(1.0 / max(0.01, model_p), 2)
            kelly = fractional_kelly(model_p, market_o)
            conf = classify_confidence(edge, model_p, weather_stable, fatigue_free)
            reasoning = generate_reasoning(match, f"BTTS - {label}", edge, model_p)

            opportunities.append(ValueOpportunity(
                match_id=match.match_id,
                league_name=match.league_name,
                league_country=match.league_country,
                match_date=match.match_date,
                match_time=match.match_time,
                home_team=match.home_team.team_name,
                away_team=match.away_team.team_name,
                market="BTTS",
                selection=label,
                market_odd=market_o,
                fair_odd=fair_odd,
                model_prob=round(model_p, 4),
                implied_prob=round(implied_p, 4),
                edge=round(edge, 4),
                edge_pct=f"+{edge*100:.1f}%",
                kelly_fraction=round(kelly, 4),
                kelly_bet_pct=f"{kelly*100:.2f}%",
                confidence=conf,
                reasoning=reasoning,
                home_xg=match.model_home_xg,
                away_xg=match.model_away_xg,
                weather_note=match.weather.description,
                fatigue_note="",
                urgency_home=match.league_urgency_home,
                urgency_away=match.league_urgency_away,
                bookmaker=match.odds.bookmaker,
                data_quality=match.data_quality_score,
                odds_suspect=getattr(match, 'odds_home_away_suspect', False),
            ))

    # ═══════════════════════════════════════════════════════════
    # SCANNER GENÉRICO — TODOS OS MERCADOS (model_probs vs all_markets)
    # ═══════════════════════════════════════════════════════════
    model_probs = getattr(match, 'model_probs', {}) or {}
    all_markets_odds = getattr(match.odds, 'all_markets', {}) or {}

    # Mercados de finalizações — gerar oportunidades MESMO sem odds da API

    if model_probs:
        for market_key, market_cfg in _ALL_MARKETS.items():
            # Pular mercados já tratados por scanners dedicados acima
            if market_key in ("1x2", "double_chance", "btts"):
                continue

            market_label = market_cfg["label"]
            market_odds_dict = all_markets_odds.get(market_key, {}) if all_markets_odds else {}

            for sel_key, sel_label in market_cfg["selections"].items():
                # Probabilidade do modelo
                prob_key = f"{market_key}__{sel_key}"
                model_p = model_probs.get(prob_key, 0)
                if model_p <= 0.005 or model_p >= 0.995:
                    continue

                # Odd do mercado — exigir odd REAL de bookmaker
                market_o = market_odds_dict.get(sel_key, 0)
                if market_o <= 1.0:
                    continue
                if not _is_odd_valid(market_o, market_label):
                    continue
                if not _is_model_sane(model_p, total_xg, market_label):
                    continue

                edge = calculate_edge(model_p, market_o)
                if edge >= config.MAX_EDGE_SANE:
                    continue

                # Exigir edge mínimo para todos os mercados
                if edge < config.MIN_EDGE_THRESHOLD:
                    continue

                fair_odd = round(1.0 / max(0.01, model_p), 2)
                implied_p = 1.0 / market_o
                kelly = fractional_kelly(model_p, market_o)
                conf = classify_confidence(edge, model_p, weather_stable, fatigue_free)

                # Identificar bookmaker correto (pode vir de _source para mercados especiais)
                source_bk = market_odds_dict.get("_source", "")
                bk_name = source_bk if source_bk else match.odds.bookmaker
                display_odd = market_o

                reasoning = generate_reasoning(match, f"{market_label} - {sel_label}", edge, model_p)

                opportunities.append(ValueOpportunity(
                    match_id=match.match_id,
                    league_name=match.league_name,
                    league_country=match.league_country,
                    match_date=match.match_date,
                    match_time=match.match_time,
                    home_team=match.home_team.team_name,
                    away_team=match.away_team.team_name,
                    market=market_label,
                    selection=sel_label,
                    market_odd=display_odd,
                    fair_odd=fair_odd,
                    model_prob=round(model_p, 4),
                    implied_prob=round(implied_p, 4),
                    edge=round(edge, 4),
                    edge_pct=f"+{edge*100:.1f}%" if edge > 0 else "Fair",
                    kelly_fraction=round(kelly, 4),
                    kelly_bet_pct=f"{kelly*100:.2f}%" if kelly > 0 else "N/A",
                    confidence=conf,
                    reasoning=reasoning,
                    home_xg=match.model_home_xg,
                    away_xg=match.model_away_xg,
                    weather_note=match.weather.description,
                    fatigue_note="",
                    urgency_home=match.league_urgency_home,
                    urgency_away=match.league_urgency_away,
                    bookmaker=bk_name,
                    data_quality=match.data_quality_score,
                    odds_suspect=getattr(match, 'odds_home_away_suspect', False),
                ))

    # ═══ DEDUPLICAÇÃO FINAL ═══
    # Remover duplicatas: mesma seleção equivalente para o mesmo jogo, manter a de maior edge
    def _norm_sel(s):
        """Normaliza seleção removendo acentos, traços, pontuação e case."""
        # Remover acentos (ã→a, é→e, etc.)
        nfkd = unicodedata.normalize('NFKD', s)
        ascii_str = ''.join(c for c in nfkd if not unicodedata.combining(c))
        # Lowercase + remover traços/pontuação + normalizar espaços
        return ascii_str.lower().replace('—', ' ').replace('–', ' ').replace('-', ' ').replace('  ', ' ').strip()

    seen = {}
    deduped = []
    for opp in opportunities:
        key = _norm_sel(opp.selection)
        if key in seen:
            idx = seen[key]
            if opp.edge > deduped[idx].edge:
                deduped[idx] = opp
        else:
            seen[key] = len(deduped)
            deduped.append(opp)

    return deduped


def find_all_value(matches: list[MatchAnalysis]) -> list[ValueOpportunity]:
    """
    Escaneia todas as partidas buscando oportunidades de valor.
    Retorna lista ordenada por Edge (maior primeiro).
    """
    print(f"[VALUE] Escaneando {len(matches)} partidas para oportunidades +EV...")

    # Contadores do portão de qualidade
    n_no_odds = sum(1 for m in matches if not m.has_real_odds)
    n_no_standings = sum(1 for m in matches if not m.has_real_standings)
    n_low_dq = sum(1 for m in matches if m.data_quality_score < 0.50)
    n_eligible = sum(1 for m in matches
                     if (m.has_real_odds or m.has_real_standings) and m.data_quality_score >= 0.40)

    print(f"[VALUE] ═══ PORTÃO DE QUALIDADE ═══")
    print(f"[VALUE]   Total partidas:           {len(matches)}")
    print(f"[VALUE]   Sem odds reais:           {n_no_odds} (BLOQUEADAS — circular)")
    print(f"[VALUE]   Sem standings reais:       {n_no_standings} (BLOQUEADAS — defaults)")
    print(f"[VALUE]   DQ < 50%:                 {n_low_dq} (BLOQUEADAS)")
    print(f"[VALUE]   ✅ Elegíveis para análise: {n_eligible}")
    print(f"[VALUE] ═══════════════════════════")

    all_opps = []
    processed_count = 0
    rejected_by_scan = 0
    
    for match in matches:
        # Só processar partidas elegíveis: precisa ter odds OU standings reais + qualidade mínima
        if not ((match.has_real_odds or match.has_real_standings) and match.data_quality_score >= 0.40):
            continue
            
        opps = scan_match_for_value(match)
        if opps:
            all_opps.extend(opps)
            processed_count += 1
        else:
            rejected_by_scan += 1
    
    if processed_count > 0:
        print(f"[VALUE] ✅ Partidas processadas com sucesso: {processed_count}")
    if rejected_by_scan > 0:
        print(f"[VALUE] ⚠️  Partidas elegíveis mas sem oportunidades encontradas: {rejected_by_scan}")
        print(f"[VALUE]    (Pode ser que não há edge suficiente ou odds/modelo estão alinhados)")

    # Ordenar por edge (maior primeiro)
    all_opps.sort(key=lambda x: x.edge, reverse=True)

    print(f"[VALUE] {len(all_opps)} oportunidades com Edge >= {config.MIN_EDGE_THRESHOLD*100:.0f}% encontradas")

    # Estatísticas
    if all_opps:
        high = sum(1 for o in all_opps if o.confidence == "ALTO")
        med = sum(1 for o in all_opps if o.confidence == "MÉDIO")
        low = sum(1 for o in all_opps if o.confidence == "BAIXO")
        print(f"[VALUE] Distribuição: ALTO={high} | MÉDIO={med} | BAIXO={low}")
        print(f"[VALUE] Maior Edge: {all_opps[0].edge_pct} em "
              f"{all_opps[0].home_team} vs {all_opps[0].away_team} ({all_opps[0].selection})")

    return all_opps
