"""
Configurações Globais do Sistema de Análise Quantitativa Esportiva.
"""

import os
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

load_dotenv()

# ═══════════════════════════════════════════════════════
# FUSO HORÁRIO — BRASIL (Brasília, UTC-3)
# ═══════════════════════════════════════════════════════
BR_TIMEZONE = timezone(timedelta(hours=-3))

# ═══════════════════════════════════════════════════════
# CHAVES DE API
# ═══════════════════════════════════════════════════════
API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY", "")
API_FOOTBALL_HOST = "v3.football.api-sports.io"
OPENWEATHER_KEY = os.getenv("OPENWEATHER_KEY", "")

# ═══════════════════════════════════════════════════════
# DATAS DE ANÁLISE (usando fuso de Brasília)
# ═══════════════════════════════════════════════════════
TODAY = datetime.now(BR_TIMEZONE).strftime("%Y-%m-%d")
TOMORROW = (datetime.now(BR_TIMEZONE) + timedelta(days=1)).strftime("%Y-%m-%d")
DAY_AFTER_TOMORROW = (datetime.now(BR_TIMEZONE) + timedelta(days=2)).strftime("%Y-%m-%d")
ANALYSIS_DATES = [TODAY, TOMORROW, DAY_AFTER_TOMORROW]  # T, T+1, T+2 — pode ser sobrescrito via /api/run


def get_default_dates() -> list[str]:
    """Retorna as datas padrão (hoje, amanhã e depois de amanhã em Brasília)."""
    now = datetime.now(BR_TIMEZONE)
    return [
        now.strftime("%Y-%m-%d"),
        (now + timedelta(days=1)).strftime("%Y-%m-%d"),
        (now + timedelta(days=2)).strftime("%Y-%m-%d"),
    ]


def build_date_range(date_from: str, date_to: str) -> list[str]:
    """Gera lista de datas YYYY-MM-DD entre date_from e date_to (inclusive).
    Suporta até 365 dias."""
    from datetime import date as _date
    try:
        start = _date.fromisoformat(date_from)
        end = _date.fromisoformat(date_to)
    except (ValueError, TypeError):
        return get_default_dates()
    if end < start:
        start, end = end, start
    if (end - start).days > 365:
        end = start + timedelta(days=365)
    dates = []
    current = start
    while current <= end:
        dates.append(current.isoformat())
        current += timedelta(days=1)
    return dates or get_default_dates()

# ═══════════════════════════════════════════════════════
# PARÂMETROS DO MODELO DIXON-COLES
# ═══════════════════════════════════════════════════════
DIXON_COLES_MAX_GOALS = 8          # Placar máximo na matriz de probabilidades
DIXON_COLES_DECAY_DAYS = 365       # Janela de dados históricos (dias)
DIXON_COLES_HALF_LIFE = 60         # Meia-vida do decaimento temporal (dias)
MONTE_CARLO_SIMULATIONS = 5000     # Iterações por jogo

# ═══════════════════════════════════════════════════════
# PARÂMETROS DE VALOR (VALUE BETTING)
# ═══════════════════════════════════════════════════════
MIN_EDGE_THRESHOLD = 0.03          # Edge mínimo de 3% para destaque (reduzido para captar mais dados)
KELLY_FRACTION = 0.25              # Kelly fracionário (1/4)
MAX_KELLY_BET = 0.05               # Máximo de 5% da banca por aposta
MAX_EDGE_SANE = 9.99               # SEM LIMITE — exibir todas as análises para refinamento do algoritmo
EXCLUDE_DRAW_1X2 = True            # Excluir empate sozinho do 1x2 (usar Dupla Chance)

# ═══════════════════════════════════════════════════════
# FILTROS DE SANIDADE DO MODELO (evitar lixo estatístico)
# ═══════════════════════════════════════════════════════
MAX_MODEL_PROB = 0.99              # SEM LIMITE prático — exibir tudo para refinamento
MIN_MODEL_PROB = 0.01              # SEM LIMITE prático — exibir tudo para refinamento
MIN_XG_TOTAL = 0.10                # Reduzido para captar mais dados
MAX_XG_TOTAL = 15.0                # Ampliado para não filtrar nada

# ═══════════════════════════════════════════════════════
# VALIDAÇÃO DE ODDS (filtro de anomalias)
# ═══════════════════════════════════════════════════════
ODDS_MIN_VALID = 1.05              # Odd mínima válida (abaixo = suspeita)
ODDS_MAX_1X2 = 25.0               # Odd máxima válida para 1x2
ODDS_MAX_DC = 5.0                  # Odd máxima válida para Dupla Chance
ODDS_MAX_OU = 15.0                 # Odd máxima válida para Over/Under Gols
ODDS_MAX_BTTS = 4.0                # Odd máxima válida para BTTS
ODDS_MAX_CORNERS = 8.0             # Odd máxima válida para Corners
ODDS_MAX_CARDS = 8.0               # Odd máxima válida para Cards
ODDS_MAX_CS = 6.0                  # Odd máxima Clean Sheet
ODDS_MAX_WTN = 15.0                # Odd máxima Win to Nil
ODDS_MAX_OE = 3.0                  # Odd máxima Odd/Even
ODDS_MAX_HT = 15.0                 # Odd máxima mercados HT
ODDS_MAX_HOME_AWAY_OU = 12.0       # Odd máxima Gols Time O/U
ODDS_MAX_EXACT = 200.0             # Odd máxima Exact Score
ODDS_MAX_GENERIC = 25.0            # Odd máxima genérica
ODDS_MAX_SHOTS = 8.0               # Odd máxima Finalizações O/U
ODDS_MAX_PLAYER_SHOTS = 8.0        # Odd máxima Finalizações Jogador O/U

# ═══════════════════════════════════════════════════════
# AJUSTES CONTEXTUAIS
# ═══════════════════════════════════════════════════════
WIND_SPEED_THRESHOLD_KMH = 20.0    # Limiar de vento (km/h)
RAIN_VOLUME_THRESHOLD_MM = 5.0     # Limiar de chuva (mm)
HEAT_THRESHOLD_C = 30.0            # Limiar de calor (°C)
XG_WIND_PENALTY = 0.08             # Redução de xG por vento forte
XG_RAIN_PENALTY = 0.05             # Ajuste por chuva
FATIGUE_PENALTY = 0.15             # Penalidade por fadiga (72h)
FATIGUE_WINDOW_HOURS = 72          # Janela de fadiga

# ═══════════════════════════════════════════════════════
# MOTIVAÇÃO / URGÊNCIA
# ═══════════════════════════════════════════════════════
LUS_HIGH_THRESHOLD = 0.9           # Urgência alta
LUS_LOW_THRESHOLD = 0.4            # Urgência baixa
COMPLACENCY_PENALTY = 0.07         # Penalidade por complacência

# ═══════════════════════════════════════════════════════
# CONTROLE DE API — PLANO PRO (7.500 req/dia, 300 req/min)
# ═══════════════════════════════════════════════════════
API_CALL_DELAY = 0.22              # PRO: 300 req/min → 0.2s entre chamadas
MAX_ODDS_FIXTURES = 500            # PRO: buscar odds para TODOS os fixtures
MAX_STANDINGS_LEAGUES = 80         # PRO: buscar standings para até 80 ligas
MAX_INJURIES_FIXTURES = 300        # PRO: buscar lesões para até 300 fixtures
PREFERRED_BOOKMAKERS = [           # Bookmakers preferidos (ordem de prioridade)
    "Bet365", "Pinnacle", "1xBet", "Unibet",
    "Marathonbet", "Betway", "Bwin", "William Hill",
]

# ═══════════════════════════════════════════════════════
# CLASSIFICAÇÃO DE RELEVÂNCIA DAS LIGAS
# S = Elite (Top 5 EU + UCL/UEL)
# A = Top (fortes EU/SA, Copas continentais)
# B = Boa (2a divisão top, ligas médias EU)
# C = Secundária (tudo o resto)
# ═══════════════════════════════════════════════════════
LEAGUE_TIERS = {
    # ── S — Elite ──
    "Premier League": "S", "La Liga": "S", "Serie A": "S",
    "Bundesliga": "S", "Ligue 1": "S",
    "UEFA Champions League": "S", "Champions League": "S",
    "UEFA Europa League": "S", "Europa League": "S",
    "UEFA Conference League": "S",
    # ── A — Top ──
    "Eredivisie": "A", "Primeira Liga": "A", "Liga Portugal": "A",
    "Pro League": "A", "Jupiler Pro League": "A",
    "Super Lig": "A", "Süper Lig": "A",
    "Scottish Premiership": "A", "Premiership": "A",
    "Premier League (Russia)": "A",
    "Serie A (Brazil)": "A", "Brasileirão": "A",
    "Serie A": "S",  # Italy (already set)
    "Primera División": "A", "Liga Profesional": "A",
    "MLS": "A", "Major League Soccer": "A",
    "Copa Libertadores": "A", "Copa Sudamericana": "A",
    "J1 League": "A", "K League 1": "A",
    "Saudi Pro League": "A", "Saudi Professional League": "A",
    "World Cup": "S", "Euro Championship": "S",
    "Copa America": "A", "AFCON": "A",
    "Nations League": "A", "UEFA Nations League": "A",
    # ── B — Boa ──
    "Championship": "B", "EFL Championship": "B",
    "Serie B": "B", "La Liga 2": "B", "Segunda División": "B",
    "2. Bundesliga": "B", "Bundesliga 2": "B",
    "Ligue 2": "B",
    "Super League": "B", "Swiss Super League": "B",
    "Bundesliga (Austria)": "B", "Austrian Bundesliga": "B",
    "Superliga": "B", "Danish Superliga": "B",
    "Eliteserien": "B", "Allsvenskan": "B",
    "Ekstraklasa": "B", "Czech Liga": "B",
    "Liga MX": "B", "Liga 1": "B",
    "Championship (Scotland)": "B",
    "Primeira Liga (Portugal)": "A",
    "A-League": "B", "Serie B (Brazil)": "B",
    "League One": "B", "League Two": "B",
    "Copa do Brasil": "B", "FA Cup": "B", "EFL Cup": "B",
    "Copa del Rey": "B", "DFB Pokal": "B", "Coupe de France": "B",
    "Coppa Italia": "B", "KNVB Beker": "B",
}

LEAGUE_TIER_BY_COUNTRY = {
    # Fallback por pais quando a liga exata nao esta no mapa
    "England": "A", "Spain": "A", "Italy": "A", "Germany": "A", "France": "A",
    "Netherlands": "A", "Portugal": "A", "Belgium": "A", "Turkey": "A",
    "Scotland": "B", "Brazil": "B", "Argentina": "B", "USA": "B",
    "Mexico": "B", "Japan": "B", "South-Korea": "B", "Saudi-Arabia": "B",
    "Austria": "B", "Switzerland": "B", "Denmark": "B", "Norway": "B",
    "Sweden": "B", "Poland": "B", "Czech-Republic": "B", "Australia": "B",
    "Russia": "B", "Ukraine": "B", "Greece": "B", "Croatia": "B",
    "Serbia": "B",
}

LEAGUE_TIER_LABELS = {
    "S": "Elite",
    "A": "Top",
    "B": "Boa",
    "C": "Secundaria",
}


def get_league_tier(league_name: str, league_country: str = "") -> str:
    """Retorna o tier (S/A/B/C) de uma liga."""
    if league_name in LEAGUE_TIERS:
        return LEAGUE_TIERS[league_name]
    ln = league_name.lower()
    for k, v in LEAGUE_TIERS.items():
        if k.lower() in ln or ln in k.lower():
            return v
    if league_country in LEAGUE_TIER_BY_COUNTRY:
        return LEAGUE_TIER_BY_COUNTRY[league_country]
    lc = league_country.lower().replace(" ", "-")
    for k, v in LEAGUE_TIER_BY_COUNTRY.items():
        if k.lower() == lc:
            return v
    return "C"

# ═══════════════════════════════════════════════════════
# SUPABASE (Banco de Dados na Nuvem)
# ═══════════════════════════════════════════════════════
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")

# ═══════════════════════════════════════════════════════
# MODO DE OPERAÇÃO
# ═══════════════════════════════════════════════════════
USE_MOCK_DATA = False              # True = dados sintéticos, False = API real
REPORT_OUTPUT_PATH = "DAILY_REPORT.md"
