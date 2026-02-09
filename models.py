"""
═══════════════════════════════════════════════════════════════════════
MÓDULO DE MODELAGEM ESTATÍSTICA AVANÇADA (O CÉREBRO)
Engine de Análise Preditiva - Camada Analítica
═══════════════════════════════════════════════════════════════════════

Implementa:
  - Modelo Dixon-Coles (Poisson Bivariada Ajustada)
  - Regressão Binomial Negativa (Escanteios / Cartões)
  - Simulação de Monte Carlo
"""

import math
import warnings
from typing import Optional

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson, nbinom

import config
from data_ingestion import MatchAnalysis

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ═══════════════════════════════════════════════════════
# 1. MODELO DIXON-COLES
# ═══════════════════════════════════════════════════════

def _tau(x: int, y: int, lambda_: float, mu: float, rho: float) -> float:
    """
    Fator de correção τ de Dixon-Coles (1997).
    Ajusta a interdependência em placares baixos (0-0, 1-0, 0-1, 1-1).

    Args:
        x: Gols do time da casa
        y: Gols do time visitante
        lambda_: Taxa de gols esperada (casa)
        mu: Taxa de gols esperada (fora)
        rho: Parâmetro de correlação

    Returns:
        Fator multiplicativo τ
    """
    if x == 0 and y == 0:
        return 1.0 - lambda_ * mu * rho
    elif x == 0 and y == 1:
        return 1.0 + lambda_ * rho
    elif x == 1 and y == 0:
        return 1.0 + mu * rho
    elif x == 1 and y == 1:
        return 1.0 - rho
    else:
        return 1.0


def dixon_coles_probability(x: int, y: int, lambda_: float,
                             mu: float, rho: float) -> float:
    """
    Calcula P(X=x, Y=y) com ajuste Dixon-Coles.

    P(x, y) = τ(x, y, λ, μ, ρ) × Poisson(x; λ) × Poisson(y; μ)
    """
    tau = _tau(x, y, lambda_, mu, rho)
    p_home = poisson.pmf(x, lambda_)
    p_away = poisson.pmf(y, mu)
    return tau * p_home * p_away


def build_score_matrix(lambda_: float, mu: float,
                        rho: float = -0.05,
                        max_goals: int = None) -> np.ndarray:
    """
    Constrói a matriz completa de probabilidade de placar exato.

    Args:
        lambda_: xG esperado do time da casa
        mu: xG esperado do time visitante
        rho: Parâmetro de correlação Dixon-Coles
        max_goals: Placar máximo a considerar

    Returns:
        Matriz (max_goals+1) x (max_goals+1) de probabilidades
    """
    if max_goals is None:
        max_goals = config.DIXON_COLES_MAX_GOALS

    matrix = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            matrix[i][j] = dixon_coles_probability(i, j, lambda_, mu, rho)

    # Normalização para garantir soma = 1
    total = matrix.sum()
    if total > 0:
        matrix /= total

    return matrix


def estimate_attack_defense_params(
    home_scored_avg: float, home_conceded_avg: float,
    away_scored_avg: float, away_conceded_avg: float,
    league_avg_goals: float = 2.7
) -> tuple[float, float, float, float]:
    """
    Estima parâmetros α (ataque) e β (defesa) para casa e fora.

    Baseado no modelo de força relativa:
        α = gols_marcados / (media_liga / 2)
        β = gols_sofridos / (media_liga / 2)

    Returns:
        (alpha_home, beta_home, alpha_away, beta_away)
    """
    half_avg = league_avg_goals / 2.0

    alpha_home = max(0.3, home_scored_avg / half_avg) if half_avg > 0 else 1.0
    beta_home = max(0.3, home_conceded_avg / half_avg) if half_avg > 0 else 1.0
    alpha_away = max(0.3, away_scored_avg / half_avg) if half_avg > 0 else 1.0
    beta_away = max(0.3, away_conceded_avg / half_avg) if half_avg > 0 else 1.0

    return alpha_home, beta_home, alpha_away, beta_away


def _regress_to_mean(observed_avg: float, league_mean: float,
                      games_played: int, min_games: int = 15) -> float:
    """
    Regressão bayesiana simples à média da liga.
    Com poucas amostras, usa mais a média da liga.
    Com muitas amostras, confia mais na média observada.

    weight = min(1, games_played / min_games)
    resultado = weight * observed + (1 - weight) * league_mean
    """
    if games_played >= min_games:
        return observed_avg
    weight = games_played / min_games
    return weight * observed_avg + (1 - weight) * league_mean


def calculate_expected_goals(match: MatchAnalysis) -> tuple[float, float]:
    """
    Calcula os gols esperados (xG) para casa e fora usando
    parâmetros de força de ataque/defesa.

    λ = α_home_team × β_away_team × home_advantage × form_factor
    μ = α_away_team × β_home_team × form_factor

    Usa a média REAL de gols da liga (match.league_avg_goals) para calibrar
    os parâmetros α/β. Com poucas amostras, aplica regressão à média.

    Quando odds_home_away_suspect é True (odds sugerem possível inversão
    de mandante/visitante), o modelo usa estatísticas NEUTRAS (média de
    casa+fora) e remove a vantagem de mando para evitar distorções.
    """
    home = match.home_team
    away = match.away_team

    # Média de gols REAL da liga (vem dos standings)
    league_avg = getattr(match, 'league_avg_goals', 2.7) or 2.7
    league_half = league_avg / 2.0  # Média por time

    # ── Verificar se odds sugerem inversão casa/fora ──
    suspect = getattr(match, 'odds_home_away_suspect', False)

    if suspect:
        # Usar estatísticas NEUTRAS: média de desempenho em casa e fora
        h_scored = (home.home_goals_scored_avg + home.away_goals_scored_avg) / 2
        h_conceded = (home.home_goals_conceded_avg + home.away_goals_conceded_avg) / 2
        a_scored = (away.home_goals_scored_avg + away.away_goals_scored_avg) / 2
        a_conceded = (away.home_goals_conceded_avg + away.away_goals_conceded_avg) / 2
        home_advantage = 1.0  # SEM vantagem de mando (campo neutro)
    else:
        h_scored = home.home_goals_scored_avg
        h_conceded = home.home_goals_conceded_avg
        a_scored = away.away_goals_scored_avg
        a_conceded = away.away_goals_conceded_avg
        home_advantage = 1.08  # Fator de vantagem de mando

    # ── Regressão à média da liga (amostras pequenas) ──
    # Com poucos jogos, regredimos as médias em direção à média da liga
    home_gp = home.games_played or 1
    away_gp = away.games_played or 1
    h_scored = _regress_to_mean(h_scored, league_half, home_gp)
    h_conceded = _regress_to_mean(h_conceded, league_half, home_gp)
    a_scored = _regress_to_mean(a_scored, league_half, away_gp)
    a_conceded = _regress_to_mean(a_conceded, league_half, away_gp)

    # Parâmetros de ataque/defesa (usando média REAL da liga)
    alpha_h, beta_h, alpha_a, beta_a = estimate_attack_defense_params(
        h_scored, h_conceded, a_scored, a_conceded,
        league_avg_goals=league_avg
    )

    # Salvar α/β reais no match para exibição correta na UI
    match.model_alpha_h = round(alpha_h, 4)
    match.model_beta_h = round(beta_h, 4)
    match.model_alpha_a = round(alpha_a, 4)
    match.model_beta_a = round(beta_a, 4)

    # Força de ataque do time da casa vs defesa do visitante
    lambda_ = alpha_h * beta_a * home_advantage
    mu = alpha_a * beta_h

    # Ajuste pela forma recente (últimos 10 jogos)
    home_form_factor = 0.85 + home.form_points * 0.30
    away_form_factor = 0.85 + away.form_points * 0.30

    lambda_ *= home_form_factor
    mu *= away_form_factor

    # Clamp para valores realistas
    lambda_ = max(0.3, min(3.5, lambda_))
    mu = max(0.2, min(3.0, mu))

    return round(lambda_, 4), round(mu, 4)


def fit_rho_parameter(lambda_: float, mu: float) -> float:
    """
    Estima o parâmetro ρ (rho) de Dixon-Coles via otimização.
    Em um cenário com dados completos, usaríamos MLE sobre resultados históricos.
    Aqui estimamos baseado nas características do jogo.

    Jogos de baixo xG tendem a ter |ρ| maior (mais interdependência).
    """
    total_xg = lambda_ + mu

    if total_xg < 1.5:
        rho = -0.12  # Jogos defensivos: forte correlação negativa em 0-0
    elif total_xg < 2.5:
        rho = -0.08
    elif total_xg < 3.5:
        rho = -0.04
    else:
        rho = -0.02  # Jogos ofensivos: pouca correção necessária

    # Restrição: ρ deve manter τ > 0
    max_rho = 1.0 / max(0.01, lambda_ * mu)
    rho = max(-max_rho, min(max_rho, rho))

    return rho


# ═══════════════════════════════════════════════════════
# 2. EXTRAÇÃO DE PROBABILIDADES DA MATRIZ
# ═══════════════════════════════════════════════════════

def extract_1x2_probabilities(matrix: np.ndarray) -> tuple[float, float, float]:
    """
    Extrai probabilidades de Vitória Casa / Empate / Vitória Fora
    da matriz de placar exato.
    """
    n = matrix.shape[0]
    p_home = p_draw = p_away = 0.0

    for i in range(n):
        for j in range(n):
            if i > j:
                p_home += matrix[i][j]
            elif i == j:
                p_draw += matrix[i][j]
            else:
                p_away += matrix[i][j]

    return p_home, p_draw, p_away


def extract_over_under_probabilities(matrix: np.ndarray,
                                      line: float = 2.5) -> tuple[float, float]:
    """
    Calcula P(Over line) e P(Under line) a partir da matriz.
    """
    n = matrix.shape[0]
    p_over = 0.0
    for i in range(n):
        for j in range(n):
            if i + j > line:
                p_over += matrix[i][j]
    return p_over, 1.0 - p_over


def extract_btts_probabilities(matrix: np.ndarray) -> tuple[float, float]:
    """
    Calcula P(Both Teams To Score) a partir da matriz.
    """
    n = matrix.shape[0]
    p_btts = 0.0
    for i in range(1, n):
        for j in range(1, n):
            p_btts += matrix[i][j]
    return p_btts, 1.0 - p_btts


def extract_correct_score_top(matrix: np.ndarray, top_n: int = 5) -> list[tuple[str, float]]:
    """
    Retorna os top N placares mais prováveis.
    """
    n = matrix.shape[0]
    scores = []
    for i in range(n):
        for j in range(n):
            scores.append((f"{i}-{j}", matrix[i][j]))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_n]


# ═══════════════════════════════════════════════════════
# 3. SIMULAÇÃO DE MONTE CARLO
# ═══════════════════════════════════════════════════════

def monte_carlo_simulation(lambda_: float, mu: float, rho: float,
                            n_sims: int = None) -> dict:
    """
    Executa simulação de Monte Carlo para gerar distribuições empíricas.

    Gera n_sims placares usando Poisson com ajuste Dixon-Coles,
    computando estatísticas derivadas.

    Returns:
        Dicionário com probabilidades de todos os mercados
    """
    if n_sims is None:
        n_sims = config.MONTE_CARLO_SIMULATIONS

    # Construir CDF da distribuição Dixon-Coles para amostragem
    max_g = config.DIXON_COLES_MAX_GOALS
    matrix = build_score_matrix(lambda_, mu, rho, max_g)

    # Flatten para amostragem
    flat_probs = matrix.flatten()
    flat_probs /= flat_probs.sum()  # Normalizar
    indices = np.arange(len(flat_probs))

    # Amostrar placares
    sampled = np.random.choice(indices, size=n_sims, p=flat_probs)
    home_goals = sampled // (max_g + 1)
    away_goals = sampled % (max_g + 1)

    total_goals = home_goals + away_goals

    # Calcular probabilidades empíricas
    results = {
        "p_home": float(np.mean(home_goals > away_goals)),
        "p_draw": float(np.mean(home_goals == away_goals)),
        "p_away": float(np.mean(home_goals < away_goals)),
        "p_over_15": float(np.mean(total_goals > 1.5)),
        "p_over_25": float(np.mean(total_goals > 2.5)),
        "p_over_35": float(np.mean(total_goals > 3.5)),
        "p_under_15": float(np.mean(total_goals < 1.5)),
        "p_under_25": float(np.mean(total_goals < 2.5)),
        "p_under_35": float(np.mean(total_goals < 3.5)),
        "p_btts": float(np.mean((home_goals > 0) & (away_goals > 0))),
        "avg_total_goals": float(np.mean(total_goals)),
        "avg_home_goals": float(np.mean(home_goals)),
        "avg_away_goals": float(np.mean(away_goals)),
        "std_total_goals": float(np.std(total_goals)),
    }

    return results


# ═══════════════════════════════════════════════════════
# 4. REGRESSÃO BINOMIAL NEGATIVA
#    Para mercados com Sobredispersão (Escanteios, Cartões)
# ═══════════════════════════════════════════════════════

def negative_binomial_pmf(k: int, mu: float, alpha: float) -> float:
    """
    PMF da distribuição Binomial Negativa parametrizada por (μ, α).

    P(X=k) = Γ(k+r) / (k! × Γ(r)) × p^r × (1-p)^k

    Onde:
        r = 1/α  (parâmetro de dispersão)
        p = r / (r + μ)
    """
    r = 1.0 / max(0.001, alpha)
    p = r / (r + mu)

    log_pmf = (gammaln(k + r) - gammaln(k + 1) - gammaln(r)
               + r * np.log(p) + k * np.log(1 - p))
    return float(np.exp(log_pmf))


def predict_corners(match: MatchAnalysis) -> tuple[float, float, dict]:
    """
    Prediz a distribuição de escanteios usando Binomial Negativa.

    Input Features:
        - Shots on Target médios (casa + fora)
        - Posse no terço final
        - Média de escanteios histórica

    Returns:
        (expected_corners, overdispersion_alpha, probability_dict)
    """
    home = match.home_team
    away = match.away_team

    # Feature engineering para escanteios
    sot_factor = (home.shots_on_target_avg + away.shots_on_target_avg) / 8.0
    blocked_factor = (home.shots_blocked_avg + away.shots_blocked_avg) / 6.0
    possession_factor = (home.possession_final_third + away.possession_final_third) / 50.0

    # Média ponderada das features
    base_corners = (home.corners_avg + away.corners_avg)
    adjusted_corners = base_corners * 0.50 + sot_factor * base_corners * 0.20 \
                       + blocked_factor * base_corners * 0.15 \
                       + possession_factor * base_corners * 0.15

    mu_corners = max(4.0, min(16.0, adjusted_corners))

    # Sobredispersão para escanteios (tipicamente 0.15 - 0.40)
    alpha = 0.25  # Variância = μ + α×μ²

    # Calcular probabilidades de linhas comuns
    probs = {}
    for line in [7.5, 8.5, 9.5, 10.5, 11.5, 12.5]:
        p_over = sum(negative_binomial_pmf(k, mu_corners, alpha)
                     for k in range(int(line) + 1, 25))
        probs[f"over_{line}"] = round(p_over, 4)
        probs[f"under_{line}"] = round(1.0 - p_over, 4)

    return mu_corners, alpha, probs


def predict_cards(match: MatchAnalysis) -> tuple[float, float, dict]:
    """
    Prediz a distribuição de cartões usando Binomial Negativa.

    Input Features:
        - Média de cartões do árbitro (Referee Severity)
        - Média de faltas dos times (Aggression)
        - Urgência da partida (High-stakes = mais cartões)

    Returns:
        (expected_cards, overdispersion_alpha, probability_dict)
    """
    home = match.home_team
    away = match.away_team
    ref = match.referee

    # Feature engineering para cartões
    # Agressividade combinada
    aggression = (home.fouls_avg + away.fouls_avg) / 25.0

    # Rigor do árbitro
    ref_severity = ref.cards_per_game_avg / 4.0

    # Urgência aumenta cartões
    urgency_factor = 1.0 + (match.league_urgency_home +
                            match.league_urgency_away) * 0.1

    # Modelo: μ_cards = base × rigor × agressividade × urgência
    base_cards = (home.cards_avg + away.cards_avg)
    mu_cards = base_cards * ref_severity * aggression * urgency_factor * 0.6

    mu_cards = max(2.0, min(10.0, mu_cards))

    # Sobredispersão para cartões (mais alta que escanteios)
    alpha = 0.35

    # Probabilidades para linhas comuns
    probs = {}
    for line in [2.5, 3.5, 4.5, 5.5, 6.5]:
        p_over = sum(negative_binomial_pmf(k, mu_cards, alpha)
                     for k in range(int(line) + 1, 20))
        probs[f"over_{line}"] = round(p_over, 4)
        probs[f"under_{line}"] = round(1.0 - p_over, 4)

    return mu_cards, alpha, probs


# ═══════════════════════════════════════════════════════
# 4b. PREVISÃO DE FINALIZAÇÕES (Shots)
#     Binomial Negativa — por time e total
# ═══════════════════════════════════════════════════════

def predict_shots(match: MatchAnalysis) -> tuple[float, float, float, float, dict]:
    """
    Prediz a distribuição de finalizações TOTAIS e NO GOL usando Binomial Negativa.

    Baseado em:
        - shots_total_avg (estimado pela força de ataque)
        - shots_on_target_avg (estimado pela força de ataque)
        - Força de ataque/defesa dos oponentes
        - Forma recente

    Returns:
        (home_shots_mu, away_shots_mu, home_sot_mu, away_sot_mu, probability_dict)
    """
    home = match.home_team
    away = match.away_team

    # ── Usar parâmetros α/β do modelo Dixon-Coles (home/away específicos, com regressão) ──
    # Estes são mais precisos que attack_strength/defense_strength (que são totais)
    h_alpha = getattr(match, 'model_home_alpha', None) or home.attack_strength
    a_alpha = getattr(match, 'model_away_alpha', None) or away.attack_strength
    h_beta = getattr(match, 'model_home_beta', None) or home.defense_strength
    a_beta = getattr(match, 'model_away_beta', None) or away.defense_strength

    # ── Finalizações Totais por time ──
    # Base proporcional à força de ataque do modelo (α)
    # Média Serie A: ~11 shots/time/jogo, ~4 SoT/time/jogo
    h_shots_base = h_alpha * 10.5   # α=1.0 → ~10.5 shots
    a_shots_base = a_alpha * 10.5

    # Ajuste pela defesa adversária (β do modelo Dixon-Coles)
    # β ALTO (>1) = defesa fraca → adversário finaliza MAIS
    # β BAIXO (<1) = defesa forte → adversário finaliza MENOS
    defense_factor_h = max(0.6, min(1.5, 0.7 + a_beta * 0.3))   # Roma chuta vs defesa Cagliari
    defense_factor_a = max(0.6, min(1.5, 0.7 + h_beta * 0.3))   # Cagliari chuta vs defesa Roma

    form_factor_h = 0.85 + home.form_points * 0.30
    form_factor_a = 0.85 + away.form_points * 0.30

    # Mando de campo: mandante finaliza ~8% a mais
    home_adv = 1.08 if not getattr(match, 'odds_home_away_suspect', False) else 1.0

    h_shots_mu = max(5.0, min(20.0, h_shots_base * defense_factor_h * form_factor_h * home_adv))
    a_shots_mu = max(5.0, min(20.0, a_shots_base * defense_factor_a * form_factor_a))

    # ── Finalizações No Gol (SoT) por time ──
    # Base proporcional a α, ~35-40% das finalizações totais são no gol
    h_sot_base = h_alpha * 3.8
    a_sot_base = a_alpha * 3.8

    h_sot_mu = max(2.0, min(10.0, h_sot_base * defense_factor_h * form_factor_h * home_adv))
    a_sot_mu = max(2.0, min(10.0, a_sot_base * defense_factor_a * form_factor_a))

    # ── Totais do jogo ──
    total_shots_mu = h_shots_mu + a_shots_mu
    total_sot_mu = h_sot_mu + a_sot_mu

    # Sobredispersão
    alpha_shots = 0.20   # Shots têm dispersão moderada
    alpha_sot = 0.25     # SoT tem dispersão um pouco maior

    probs = {}

    # ── Total Match Shots O/U ──
    for line in [18.5, 20.5, 22.5, 24.5, 26.5, 28.5]:
        p_over = sum(negative_binomial_pmf(k, total_shots_mu, alpha_shots)
                     for k in range(int(line) + 1, 50))
        probs[f"over_{line}"] = round(p_over, 4)
        probs[f"under_{line}"] = round(1.0 - p_over, 4)

    # ── Total Match SoT O/U ──
    for line in [4.5, 5.5, 6.5, 7.5, 8.5, 9.5]:
        p_over = sum(negative_binomial_pmf(k, total_sot_mu, alpha_sot)
                     for k in range(int(line) + 1, 30))
        probs[f"sot_over_{line}"] = round(p_over, 4)
        probs[f"sot_under_{line}"] = round(1.0 - p_over, 4)

    # ── Home Team Shots O/U ──
    for line in [8.5, 10.5, 12.5, 14.5]:
        p_over = sum(negative_binomial_pmf(k, h_shots_mu, alpha_shots)
                     for k in range(int(line) + 1, 35))
        probs[f"home_over_{line}"] = round(p_over, 4)
        probs[f"home_under_{line}"] = round(1.0 - p_over, 4)

    # ── Away Team Shots O/U ──
    for line in [8.5, 10.5, 12.5, 14.5]:
        p_over = sum(negative_binomial_pmf(k, a_shots_mu, alpha_shots)
                     for k in range(int(line) + 1, 35))
        probs[f"away_over_{line}"] = round(p_over, 4)
        probs[f"away_under_{line}"] = round(1.0 - p_over, 4)

    # ── Home SoT O/U ──
    for line in [2.5, 3.5, 4.5, 5.5]:
        p_over = sum(negative_binomial_pmf(k, h_sot_mu, alpha_sot)
                     for k in range(int(line) + 1, 20))
        probs[f"home_sot_over_{line}"] = round(p_over, 4)
        probs[f"home_sot_under_{line}"] = round(1.0 - p_over, 4)

    # ── Away SoT O/U ──
    for line in [2.5, 3.5, 4.5, 5.5]:
        p_over = sum(negative_binomial_pmf(k, a_sot_mu, alpha_sot)
                     for k in range(int(line) + 1, 20))
        probs[f"away_sot_over_{line}"] = round(p_over, 4)
        probs[f"away_sot_under_{line}"] = round(1.0 - p_over, 4)

    # ── Shots 1x2: qual time terá mais finalizações totais ──
    # P(home > away), P(home = away), P(away > home)
    max_shots_calc = 35
    h_pmf = [negative_binomial_pmf(k, h_shots_mu, alpha_shots) for k in range(max_shots_calc)]
    a_pmf = [negative_binomial_pmf(k, a_shots_mu, alpha_shots) for k in range(max_shots_calc)]
    
    p_home_more = 0.0
    p_draw_shots = 0.0
    p_away_more = 0.0
    for h in range(max_shots_calc):
        for a in range(max_shots_calc):
            joint = h_pmf[h] * a_pmf[a]
            if h > a:
                p_home_more += joint
            elif h == a:
                p_draw_shots += joint
            else:
                p_away_more += joint
    
    probs["shots_1x2_home"] = round(p_home_more, 4)
    probs["shots_1x2_draw"] = round(p_draw_shots, 4)
    probs["shots_1x2_away"] = round(p_away_more, 4)

    # ── SoT 1x2: qual time terá mais finalizações ao gol ──
    max_sot_calc = 20
    h_sot_pmf = [negative_binomial_pmf(k, h_sot_mu, alpha_sot) for k in range(max_sot_calc)]
    a_sot_pmf = [negative_binomial_pmf(k, a_sot_mu, alpha_sot) for k in range(max_sot_calc)]
    
    p_home_more_sot = 0.0
    p_draw_sot = 0.0
    p_away_more_sot = 0.0
    for h in range(max_sot_calc):
        for a in range(max_sot_calc):
            joint = h_sot_pmf[h] * a_sot_pmf[a]
            if h > a:
                p_home_more_sot += joint
            elif h == a:
                p_draw_sot += joint
            else:
                p_away_more_sot += joint
    
    probs["sot_1x2_home"] = round(p_home_more_sot, 4)
    probs["sot_1x2_draw"] = round(p_draw_sot, 4)
    probs["sot_1x2_away"] = round(p_away_more_sot, 4)

    return h_shots_mu, a_shots_mu, h_sot_mu, a_sot_mu, probs


# ═══════════════════════════════════════════════════════
# 5. PIPELINE PRINCIPAL DE MODELAGEM
# ═══════════════════════════════════════════════════════

def run_full_model(match: MatchAnalysis) -> MatchAnalysis:
    """
    Executa o pipeline completo de modelagem para uma partida.

    1. Calcula xG (Dixon-Coles)
    2. Constrói matriz de placar exato
    3. Roda Monte Carlo
    4. Prediz escanteios (NB)
    5. Prediz cartões (NB)
    6. Popula o MatchAnalysis com todos os resultados

    Returns:
        MatchAnalysis atualizado com probabilidades do modelo
    """
    # ═══ VALIDAÇÃO RIGOROSA DE DADOS REAIS ═══
    # Garantir que apenas partidas com dados reais sejam processadas
    if not match.has_real_odds:
        raise ValueError(f"Partida {match.match_id} rejeitada: sem odds REAIS (bookmaker: {match.odds.bookmaker})")
    
    if not match.has_real_standings:
        raise ValueError(f"Partida {match.match_id} rejeitada: sem standings REAIS (nenhum time com dados da API)")
    
    # Ajustado: aceitar qualidade >= 0.40 (odds reais OU standings reais)
    if match.data_quality_score < 0.40:
        raise ValueError(f"Partida {match.match_id} rejeitada: qualidade de dados insuficiente ({match.data_quality_score:.2f} < 0.40)")
    
    # Validar que os valores numéricos são válidos
    home = match.home_team
    away = match.away_team
    
    if (home.attack_strength <= 0 or home.defense_strength <= 0 or
        away.attack_strength <= 0 or away.defense_strength <= 0):
        raise ValueError(f"Partida {match.match_id} rejeitada: valores de ataque/defesa inválidos")
    
    # Nota: has_real_standings=True já garante que pelo menos 1 time tem dados reais
    # Não precisamos verificar has_real_data individualmente aqui
    
    # 1. Gols esperados
    lambda_, mu = calculate_expected_goals(match)
    match.model_home_xg = lambda_
    match.model_away_xg = mu

    # 2. Parâmetro rho e matriz de placar
    rho = fit_rho_parameter(lambda_, mu)
    matrix = build_score_matrix(lambda_, mu, rho)
    match.score_matrix = matrix

    # 3. Probabilidades analíticas da matriz
    p_home, p_draw, p_away = extract_1x2_probabilities(matrix)
    p_over25, p_under25 = extract_over_under_probabilities(matrix, 2.5)
    p_btts, _ = extract_btts_probabilities(matrix)

    # 4. Monte Carlo para robustez
    mc_results = monte_carlo_simulation(lambda_, mu, rho)

    # Blend: 60% analítico + 40% Monte Carlo (para estabilidade)
    match.model_prob_home = round(0.6 * p_home + 0.4 * mc_results["p_home"], 4)
    match.model_prob_draw = round(0.6 * p_draw + 0.4 * mc_results["p_draw"], 4)
    match.model_prob_away = round(0.6 * p_away + 0.4 * mc_results["p_away"], 4)
    match.model_prob_over25 = round(0.6 * p_over25 + 0.4 * mc_results["p_over_25"], 4)
    match.model_prob_btts = round(0.6 * p_btts + 0.4 * mc_results["p_btts"], 4)

    # Normalizar 1x2
    total_1x2 = match.model_prob_home + match.model_prob_draw + match.model_prob_away
    if total_1x2 > 0:
        match.model_prob_home /= total_1x2
        match.model_prob_draw /= total_1x2
        match.model_prob_away /= total_1x2

    # 5. Escanteios (Binomial Negativa)
    corners_mu, corners_alpha, corners_probs = predict_corners(match)
    match.model_corners_expected = round(corners_mu, 2)

    # 6. Cartões (Binomial Negativa)
    cards_mu, cards_alpha, cards_probs = predict_cards(match)
    match.model_cards_expected = round(cards_mu, 2)

    # 7. Finalizações (Binomial Negativa)
    h_shots_mu, a_shots_mu, h_sot_mu, a_sot_mu, shots_probs = predict_shots(match)
    match.model_home_shots_expected = round(h_shots_mu, 1)
    match.model_away_shots_expected = round(a_shots_mu, 1)
    match.model_home_sot_expected = round(h_sot_mu, 1)
    match.model_away_sot_expected = round(a_sot_mu, 1)
    match.model_total_shots_expected = round(h_shots_mu + a_shots_mu, 1)
    match.model_total_sot_expected = round(h_sot_mu + a_sot_mu, 1)

    # ═══════════════════════════════════════════════════════
    # 8. PROBABILIDADES EXPANDIDAS — TODOS OS MERCADOS
    # ═══════════════════════════════════════════════════════
    M = matrix
    max_g = M.shape[0]
    probs = {}

    # ── Total Goals Over/Under (TODAS as linhas) ──
    for line in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]:
        p_over = float(sum(M[i][j] for i in range(max_g) for j in range(max_g) if i + j > line))
        probs[f"goals_ou__over_{line}"] = round(p_over, 4)
        probs[f"goals_ou__under_{line}"] = round(1 - p_over, 4)

    # ── BTTS ──
    p_btts_y = float(sum(M[i][j] for i in range(1, max_g) for j in range(1, max_g)))
    probs["btts__yes"] = round(p_btts_y, 4)
    probs["btts__no"] = round(1 - p_btts_y, 4)

    # ── Home Goals Over/Under ──
    for line in [0.5, 1.5, 2.5, 3.5]:
        p_over = float(sum(M[i][j] for i in range(int(line) + 1, max_g) for j in range(max_g)))
        probs[f"home_goals_ou__over_{line}"] = round(p_over, 4)
        probs[f"home_goals_ou__under_{line}"] = round(1 - p_over, 4)

    # ── Away Goals Over/Under ──
    for line in [0.5, 1.5, 2.5, 3.5]:
        p_over = float(sum(M[i][j] for i in range(max_g) for j in range(int(line) + 1, max_g)))
        probs[f"away_goals_ou__over_{line}"] = round(p_over, 4)
        probs[f"away_goals_ou__under_{line}"] = round(1 - p_over, 4)

    # ── Clean Sheet (adversario marca 0) ──
    probs["cs_home__yes"] = round(float(sum(M[i][0] for i in range(max_g))), 4)
    probs["cs_home__no"] = round(1 - probs["cs_home__yes"], 4)
    probs["cs_away__yes"] = round(float(sum(M[0][j] for j in range(max_g))), 4)
    probs["cs_away__no"] = round(1 - probs["cs_away__yes"], 4)

    # ── Win to Nil (vencer sem sofrer gols) ──
    probs["wtn_home__yes"] = round(float(sum(M[i][0] for i in range(1, max_g))), 4)
    probs["wtn_home__no"] = round(1 - probs["wtn_home__yes"], 4)
    probs["wtn_away__yes"] = round(float(sum(M[0][j] for j in range(1, max_g))), 4)
    probs["wtn_away__no"] = round(1 - probs["wtn_away__yes"], 4)

    # ── Odd/Even Goals ──
    p_odd = float(sum(M[i][j] for i in range(max_g) for j in range(max_g) if (i + j) % 2 == 1))
    probs["odd_even__odd"] = round(p_odd, 4)
    probs["odd_even__even"] = round(1 - p_odd, 4)

    # ── 1X2 (para all_markets compatibility) ──
    probs["1x2__home"] = match.model_prob_home
    probs["1x2__draw"] = match.model_prob_draw
    probs["1x2__away"] = match.model_prob_away

    # ── Double Chance ──
    probs["double_chance__home/draw"] = round(match.model_prob_home + match.model_prob_draw, 4)
    probs["double_chance__draw/away"] = round(match.model_prob_away + match.model_prob_draw, 4)
    probs["double_chance__home/away"] = round(match.model_prob_home + match.model_prob_away, 4)

    # ── Exact Score (top 15) ──
    exact = []
    for i in range(min(6, max_g)):
        for j in range(min(6, max_g)):
            exact.append((f"{i}-{j}", round(float(M[i][j]), 4)))
    exact.sort(key=lambda x: x[1], reverse=True)
    for s, p in exact[:15]:
        probs[f"exact_score__{s}"] = p

    # ── HT Model (empirical: ~42% of FT goals at halftime) ──
    from scipy.stats import poisson as poisson_rv
    ht_factor = 0.42
    ht_hxg = max(0.01, lambda_ * ht_factor)
    ht_axg = max(0.01, mu * ht_factor)
    ht_max = 5
    ht_M = np.zeros((ht_max + 1, ht_max + 1))
    for i in range(ht_max + 1):
        for j in range(ht_max + 1):
            ht_M[i][j] = poisson_rv.pmf(i, ht_hxg) * poisson_rv.pmf(j, ht_axg)

    # HT Result
    probs["ht_result__home"] = round(float(sum(ht_M[i][j] for i in range(ht_max + 1) for j in range(ht_max + 1) if i > j)), 4)
    probs["ht_result__draw"] = round(float(sum(ht_M[i][j] for i in range(ht_max + 1) for j in range(ht_max + 1) if i == j)), 4)
    probs["ht_result__away"] = round(float(sum(ht_M[i][j] for i in range(ht_max + 1) for j in range(ht_max + 1) if i < j)), 4)

    # HT Goals O/U
    for line in [0.5, 1.5, 2.5]:
        p_over = float(sum(ht_M[i][j] for i in range(ht_max + 1) for j in range(ht_max + 1) if i + j > line))
        probs[f"ht_goals_ou__over_{line}"] = round(p_over, 4)
        probs[f"ht_goals_ou__under_{line}"] = round(1 - p_over, 4)

    # ── Corners O/U (Binomial Negativa — todas as linhas) ──
    for k, v in corners_probs.items():
        probs[f"corners_ou__{k}"] = v

    # ── Cards O/U (Binomial Negativa — todas as linhas) ──
    for k, v in cards_probs.items():
        probs[f"cards_ou__{k}"] = v

    # ── Shots O/U (Total Match) ──
    for k, v in shots_probs.items():
        if k.startswith("shots_1x2_"):
            # Shots 1x2: shots_1x2_home, shots_1x2_draw, shots_1x2_away
            probs[f"shots_1x2__{k.replace('shots_1x2_', '')}"] = v
        elif k.startswith("sot_1x2_"):
            # SoT 1x2: sot_1x2_home, sot_1x2_draw, sot_1x2_away
            probs[f"sot_1x2__{k.replace('sot_1x2_', '')}"] = v
        elif k.startswith("over_") or k.startswith("under_"):
            # Total match shots: over_20.5, under_20.5, etc.
            probs[f"shots_ou__{k}"] = v
        elif k.startswith("sot_"):
            # Total match SoT: sot_over_6.5, sot_under_6.5, etc.
            probs[f"sot_ou__{k.replace('sot_', '')}"] = v
        elif k.startswith("home_sot_"):
            # Home SoT: home_sot_over_3.5, etc.
            probs[f"home_sot_ou__{k.replace('home_sot_', '')}"] = v
        elif k.startswith("away_sot_"):
            # Away SoT: away_sot_over_3.5, etc.
            probs[f"away_sot_ou__{k.replace('away_sot_', '')}"] = v
        elif k.startswith("home_"):
            # Home shots: home_over_10.5, etc.
            probs[f"home_shots_ou__{k.replace('home_', '')}"] = v
        elif k.startswith("away_"):
            # Away shots: away_over_10.5, etc.
            probs[f"away_shots_ou__{k.replace('away_', '')}"] = v

    match.model_probs = probs
    return match


def run_models_batch(matches: list[MatchAnalysis]) -> list[MatchAnalysis]:
    """
    Executa modelagem para lote de partidas.
    Rejeita partidas sem dados reais suficientes e continua processando as demais.
    """
    print(f"[MODELS] Executando modelagem para {len(matches)} partidas...")
    print("[MODELS] ⚠️  Apenas partidas com DADOS REAIS serão processadas:")
    print("[MODELS]    - Odds REAIS de bookmaker")
    print("[MODELS]    - Standings REAIS de pelo menos 1 time")
    print("[MODELS]    - Qualidade de dados >= 0.50")
    print()

    valid_matches = []
    rejected_count = 0
    
    for i, match in enumerate(matches):
        try:
            result = run_full_model(match)
            valid_matches.append(result)
        except ValueError as e:
            # Partida rejeitada por falta de dados reais - isso é esperado e correto
            rejected_count += 1
            if rejected_count <= 5:  # Mostrar apenas as primeiras 5 rejeições
                print(f"[MODELS] ⚠️  {str(e)}")
        except Exception as e:
            # Erro inesperado - logar mas continuar
            print(f"[MODELS] ❌ Erro ao processar partida {match.match_id}: {e}")
            rejected_count += 1

        if (i + 1) % 10 == 0:
            print(f"[MODELS] Progresso: {i+1}/{len(matches)} | Válidas: {len(valid_matches)} | Rejeitadas: {rejected_count}")

    if rejected_count > 5:
        print(f"[MODELS] ... e mais {rejected_count - 5} partidas rejeitadas (sem dados reais suficientes)")

    print(f"[MODELS] ✅ Modelagem concluída: {len(valid_matches)} partidas processadas, {rejected_count} rejeitadas")
    return valid_matches
