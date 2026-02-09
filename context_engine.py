"""
═══════════════════════════════════════════════════════════════════════
MÓDULO DE INTELIGÊNCIA CONTEXTUAL (AJUSTE FINO)
Engine de Análise Preditiva - Camada Contextual
═══════════════════════════════════════════════════════════════════════

Aplica modificadores sobre as probabilidades brutas:
  - Clima (vento, chuva, temperatura)
  - Índice de Urgência / Motivação (LUS)
  - Fadiga e Rotação de Elenco
  - Lesões de jogadores-chave
"""

from datetime import datetime, timedelta

import config
from data_ingestion import MatchAnalysis


# ═══════════════════════════════════════════════════════
# 1. ÍNDICE DE URGÊNCIA DA LIGA (LUS)
# ═══════════════════════════════════════════════════════

def calculate_league_urgency(points_to_title: int, points_to_relegation: int,
                              league_position: int, games_remaining: int,
                              total_teams: int = 20) -> float:
    """
    Calcula o League Urgency Score (LUS) para um time.

    Algoritmo:
        - Se Pontos para o Título < 3 OU Pontos para Rebaixamento < 3 → LUS = 1.0
        - Se Posição = Meio de Tabela E Jogos Restantes < 5 → LUS = 0.2
        - Interpolação linear entre extremos

    Returns:
        LUS entre 0.0 (nenhuma motivação) e 1.0 (máxima urgência)
    """
    # Extremos: luta pelo título ou contra rebaixamento
    if points_to_title <= 3:
        return 1.0
    if points_to_relegation <= 3:
        return 1.0

    # Meio de tabela com poucos jogos → complacência
    mid_start = total_teams // 3
    mid_end = 2 * total_teams // 3
    is_mid_table = mid_start <= league_position <= mid_end

    if is_mid_table and games_remaining <= 5:
        return 0.2

    # Gradiente baseado na proximidade de objetivos
    title_urgency = max(0, 1.0 - points_to_title / 20.0)
    relegation_urgency = max(0, 1.0 - points_to_relegation / 15.0)

    # Peso maior para quem está mais ameaçado
    base_urgency = max(title_urgency, relegation_urgency)

    # Boost se faltam poucos jogos (pressão temporal)
    if games_remaining <= 8:
        time_pressure = 1.0 + (8 - games_remaining) * 0.05
        base_urgency = min(1.0, base_urgency * time_pressure)

    # Garante intervalo [0.2, 1.0]
    return round(max(0.2, min(1.0, base_urgency + 0.2)), 3)


# ═══════════════════════════════════════════════════════
# 2. AJUSTE CLIMÁTICO
# ═══════════════════════════════════════════════════════

def calculate_weather_adjustments(match: MatchAnalysis) -> dict:
    """
    Calcula modificadores baseados nas condições meteorológicas.

    Regras baseadas em análise física:
        - Vento > 20km/h: Degrada passes longos e bolas paradas
        - Chuva > 5mm: Aumenta erros técnicos
        - Temperatura > 30°C: Reduz pressing no 2º tempo

    Returns:
        Dicionário com multiplicadores para cada dimensão
    """
    weather = match.weather
    adjustments = {
        "xg_multiplier": 1.0,
        "corners_multiplier": 1.0,
        "cards_multiplier": 1.0,
        "btts_boost": 0.0,
        "variance_multiplier": 1.0,
        "description": [],
    }

    # ── VENTO ──
    if weather.wind_speed_kmh > config.WIND_SPEED_THRESHOLD_KMH:
        wind_severity = min(1.0, (weather.wind_speed_kmh - config.WIND_SPEED_THRESHOLD_KMH) / 30.0)
        xg_penalty = config.XG_WIND_PENALTY * (1.0 + wind_severity)

        adjustments["xg_multiplier"] -= xg_penalty
        adjustments["corners_multiplier"] -= xg_penalty * 0.5  # Escanteios menos afetados
        adjustments["variance_multiplier"] += wind_severity * 0.15
        adjustments["description"].append(
            f"Vento forte ({weather.wind_speed_kmh:.0f} km/h): "
            f"xG reduzido em {xg_penalty*100:.1f}%"
        )

    # ── CHUVA ──
    if weather.rain_mm > config.RAIN_VOLUME_THRESHOLD_MM:
        rain_severity = min(1.0, (weather.rain_mm - config.RAIN_VOLUME_THRESHOLD_MM) / 15.0)
        xg_rain_adj = config.XG_RAIN_PENALTY * (1.0 + rain_severity)

        adjustments["xg_multiplier"] -= xg_rain_adj * 0.3  # Chuva tem efeito ambíguo
        adjustments["btts_boost"] = rain_severity * 0.05  # Mais erros → mais BTTS
        adjustments["variance_multiplier"] += rain_severity * 0.20
        adjustments["cards_multiplier"] += rain_severity * 0.10  # Mais faltas em piso molhado
        adjustments["description"].append(
            f"Chuva ({weather.rain_mm:.1f}mm): "
            f"Variância aumentada, BTTS +{rain_severity*5:.1f}%"
        )

    # ── TEMPERATURA EXTREMA ──
    if weather.temperature_c > config.HEAT_THRESHOLD_C:
        heat_severity = min(1.0, (weather.temperature_c - config.HEAT_THRESHOLD_C) / 10.0)

        adjustments["xg_multiplier"] -= heat_severity * 0.04  # Menos intensidade geral
        adjustments["corners_multiplier"] -= heat_severity * 0.10  # Ritmo lento
        adjustments["description"].append(
            f"Calor extremo ({weather.temperature_c:.0f}°C): "
            f"Pressing reduzido, menos escanteios"
        )
    elif weather.temperature_c < 5.0:
        cold_severity = min(1.0, (5.0 - weather.temperature_c) / 15.0)
        adjustments["variance_multiplier"] += cold_severity * 0.10
        adjustments["description"].append(
            f"Frio intenso ({weather.temperature_c:.0f}°C): "
            f"Variância ligeiramente aumentada"
        )

    # Clamp dos multiplicadores
    adjustments["xg_multiplier"] = max(0.75, min(1.10, adjustments["xg_multiplier"]))
    adjustments["corners_multiplier"] = max(0.70, min(1.10, adjustments["corners_multiplier"]))
    adjustments["cards_multiplier"] = max(0.85, min(1.25, adjustments["cards_multiplier"]))
    adjustments["variance_multiplier"] = max(1.0, min(1.50, adjustments["variance_multiplier"]))

    return adjustments


# ═══════════════════════════════════════════════════════
# 3. FADIGA E ROTAÇÃO
# ═══════════════════════════════════════════════════════

def check_fatigue(last_match_date: str, match_date: str) -> tuple[bool, float]:
    """
    Verifica se o time está em fadiga (jogou < 72h antes).

    Returns:
        (is_fatigued, penalty_factor)
    """
    if not last_match_date:
        return False, 1.0

    try:
        last = datetime.strptime(last_match_date, "%Y-%m-%d %H:%M")
        current = datetime.strptime(match_date, "%Y-%m-%d")
        hours_diff = (current - last).total_seconds() / 3600

        if hours_diff < config.FATIGUE_WINDOW_HOURS:
            # Penalidade proporcional: quanto menos descanso, mais fadiga
            rest_ratio = hours_diff / config.FATIGUE_WINDOW_HOURS
            penalty = config.FATIGUE_PENALTY * (1.0 - rest_ratio)
            return True, round(1.0 - penalty, 3)
        return False, 1.0
    except (ValueError, TypeError):
        return False, 1.0


# ═══════════════════════════════════════════════════════
# 4. IMPACTO DE LESÕES
# ═══════════════════════════════════════════════════════

def calculate_injury_impact(injuries: list[str]) -> float:
    """
    Estima o impacto das lesões no desempenho do time.

    Returns:
        Multiplicador (1.0 = sem impacto, 0.85 = impacto significativo)
    """
    if not injuries:
        return 1.0

    n_injuries = len(injuries)

    # Cada lesão reduz ~2-3% da eficácia
    impact = max(0.85, 1.0 - n_injuries * 0.025)

    # Verificar se há lesões graves (indicadas por "meses")
    for inj in injuries:
        if "meses" in inj.lower() or "months" in inj.lower():
            impact -= 0.01  # Lesão longa = jogador importante

    return round(max(0.80, impact), 3)


# ═══════════════════════════════════════════════════════
# 5. PIPELINE DE AJUSTE CONTEXTUAL
# ═══════════════════════════════════════════════════════

def apply_contextual_adjustments(match: MatchAnalysis) -> MatchAnalysis:
    """
    Aplica TODOS os ajustes contextuais sobre as probabilidades do modelo.

    Ordem de aplicação:
        1. Urgência / Motivação
        2. Clima
        3. Fadiga
        4. Lesões
    """
    # ── 1. URGÊNCIA (LUS) ──
    home_lus = calculate_league_urgency(
        match.home_team.points_to_title,
        match.home_team.points_to_relegation,
        match.home_team.league_position,
        match.home_team.games_remaining,
    )
    away_lus = calculate_league_urgency(
        match.away_team.points_to_title,
        match.away_team.points_to_relegation,
        match.away_team.league_position,
        match.away_team.games_remaining,
    )
    match.league_urgency_home = home_lus
    match.league_urgency_away = away_lus

    # Complacência: se ambos têm baixa urgência
    if home_lus < config.LUS_LOW_THRESHOLD and away_lus < config.LUS_LOW_THRESHOLD:
        # Reduzir probabilidade de vitória para ambos (mais empates)
        complacency = config.COMPLACENCY_PENALTY
        match.model_prob_home *= (1.0 - complacency * 0.5)
        match.model_prob_away *= (1.0 - complacency * 0.5)
        match.model_prob_draw += complacency

    # Alta urgência de um lado = vantagem motivacional
    elif abs(home_lus - away_lus) > 0.4:
        motivated_boost = 0.03
        if home_lus > away_lus:
            match.model_prob_home += motivated_boost
            match.model_prob_away -= motivated_boost
        else:
            match.model_prob_away += motivated_boost
            match.model_prob_home -= motivated_boost

    # ── 2. CLIMA ──
    weather_adj = calculate_weather_adjustments(match)

    match.model_home_xg *= weather_adj["xg_multiplier"]
    match.model_away_xg *= weather_adj["xg_multiplier"]
    match.model_corners_expected *= weather_adj["corners_multiplier"]
    match.model_cards_expected *= weather_adj["cards_multiplier"]

    if weather_adj["btts_boost"] > 0:
        match.model_prob_btts = min(0.95, match.model_prob_btts + weather_adj["btts_boost"])

    # Ajustar O/U com base no xG modificado
    if weather_adj["xg_multiplier"] < 0.95:
        penalty = 1.0 - weather_adj["xg_multiplier"]
        match.model_prob_over25 *= (1.0 - penalty * 0.8)

    # ── 3. FADIGA ──
    home_fatigued, home_fatigue_factor = check_fatigue(
        match.home_team.last_match_date, match.match_date
    )
    away_fatigued, away_fatigue_factor = check_fatigue(
        match.away_team.last_match_date, match.match_date
    )

    match.home_fatigue = home_fatigued
    match.away_fatigue = away_fatigued

    if home_fatigued:
        match.model_prob_home *= home_fatigue_factor
        match.model_prob_away *= (2.0 - home_fatigue_factor)  # Adversário beneficiado
        match.model_home_xg *= home_fatigue_factor
        match.model_corners_expected *= (home_fatigue_factor + 1.0) / 2.0

    if away_fatigued:
        match.model_prob_away *= away_fatigue_factor
        match.model_prob_home *= (2.0 - away_fatigue_factor)
        match.model_away_xg *= away_fatigue_factor
        match.model_corners_expected *= (away_fatigue_factor + 1.0) / 2.0

    # ── 4. LESÕES ──
    home_injury_impact = calculate_injury_impact(match.injuries_home)
    away_injury_impact = calculate_injury_impact(match.injuries_away)

    match.model_prob_home *= home_injury_impact
    match.model_away_xg /= max(0.85, away_injury_impact)  # Defesa enfraquecida

    match.model_prob_away *= away_injury_impact
    match.model_home_xg /= max(0.85, home_injury_impact)

    # ── NORMALIZAÇÃO FINAL ──
    total = match.model_prob_home + match.model_prob_draw + match.model_prob_away
    if total > 0:
        match.model_prob_home = round(match.model_prob_home / total, 4)
        match.model_prob_draw = round(match.model_prob_draw / total, 4)
        match.model_prob_away = round(match.model_prob_away / total, 4)

    # Clamp de probabilidades
    match.model_prob_over25 = round(max(0.05, min(0.95, match.model_prob_over25)), 4)
    match.model_prob_btts = round(max(0.05, min(0.95, match.model_prob_btts)), 4)
    match.model_home_xg = round(max(0.2, match.model_home_xg), 3)
    match.model_away_xg = round(max(0.15, match.model_away_xg), 3)
    match.model_corners_expected = round(max(3.0, match.model_corners_expected), 2)
    match.model_cards_expected = round(max(1.5, match.model_cards_expected), 2)

    return match


def apply_context_batch(matches: list[MatchAnalysis]) -> list[MatchAnalysis]:
    """
    Aplica ajustes contextuais a um lote de partidas.
    """
    print(f"[CONTEXT] Aplicando inteligência contextual a {len(matches)} partidas...")

    for i, match in enumerate(matches):
        matches[i] = apply_contextual_adjustments(match)

    print("[CONTEXT] Ajustes contextuais concluídos.")
    return matches
