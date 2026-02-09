"""
Configurações Globais do Sistema de Análise Quantitativa Esportiva.
"""

import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# ═══════════════════════════════════════════════════════
# CHAVES DE API
# ═══════════════════════════════════════════════════════
API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY", "")
API_FOOTBALL_HOST = "v3.football.api-sports.io"
OPENWEATHER_KEY = os.getenv("OPENWEATHER_KEY", "")

# ═══════════════════════════════════════════════════════
# DATAS DE ANÁLISE
# ═══════════════════════════════════════════════════════
TODAY = datetime.now().strftime("%Y-%m-%d")
TOMORROW = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
ANALYSIS_DATES = [TODAY, TOMORROW]

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
    "Pinnacle", "Bet365", "1xBet", "Unibet",
    "Marathonbet", "Betway", "Bwin", "William Hill",
]

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
