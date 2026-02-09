"""
═══════════════════════════════════════════════════════════════════════
MÓDULO DE AQUISIÇÃO E ENGENHARIA DE DADOS (ETL)
Engine de Análise Preditiva - Camada Sensorial

Suporta:
  - Modo Real: API-Football (v3) + OpenWeatherMap
  - Modo Demo: Dados sintéticos realistas (fallback)
  - Cache local de respostas da API (evita gastar créditos em re-execuções)
═══════════════════════════════════════════════════════════════════════
"""

import random
import math
import time
import json
import os
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import requests

import config


# ═══════════════════════════════════════════════════════
# CACHE DE RESPOSTAS DA API — LOCAL + SUPABASE
# Fluxo: 1) Cache local  2) Supabase  3) API real
# Salva em AMBOS para nunca perder dados.
# ═══════════════════════════════════════════════════════

import supabase_client

_API_CACHE_DIR = os.path.join(os.path.dirname(__file__), "_api_cache")
os.makedirs(_API_CACHE_DIR, exist_ok=True)

# TTL em horas por tipo de endpoint
_CACHE_TTL_HOURS = {
    "fixtures":             3,
    "standings":           12,
    "odds":                 4,
    "injuries":             6,
    "fixtures/lineups":   720,
    "fixtures/statistics": 720,
    "fixtures/players":   720,   # Dados de jogadores de partidas passadas
    "weather":              3,
    "status":               1,
    "team_history":        12,    # Resultado completo com análise EV+
}


def _cache_key(endpoint: str, params: dict) -> str:
    """Gera chave unica para cache baseada em endpoint + params."""
    params_str = json.dumps(params, sort_keys=True)
    h = hashlib.md5(f"{endpoint}_{params_str}".encode()).hexdigest()[:12]
    return f"{endpoint.replace('/', '_')}_{h}"


def _get_cached_response(endpoint: str, params: dict) -> dict | None:
    """Busca resposta em cache: 1) local  2) Supabase.
    Retorna None se nao encontrada ou expirada."""
    key = _cache_key(endpoint, params)
    filepath = os.path.join(_API_CACHE_DIR, f"{key}.json")
    ttl_hours = _CACHE_TTL_HOURS.get(endpoint, 4)

    # ── 1. CACHE LOCAL (mais rapido) ──
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                cached = json.load(f)
            cached_at = datetime.fromisoformat(cached.get("_cached_at", "2000-01-01"))
            age_hours = (datetime.now() - cached_at).total_seconds() / 3600
            if age_hours < ttl_hours:
                return cached.get("data", {})
        except Exception:
            pass

    # ── 2. SUPABASE (backup na nuvem) ──
    try:
        sb_data = supabase_client.get_api_response(endpoint, params)
        if sb_data is not None:
            # Salvar localmente para proxima vez ser mais rapido
            _save_to_local_cache(endpoint, params, sb_data)
            return sb_data
    except Exception:
        pass

    return None


def _save_to_cache(endpoint: str, params: dict, data: dict):
    """Salva resposta da API em AMBOS: cache local E Supabase."""
    # ── 1. CACHE LOCAL ──
    _save_to_local_cache(endpoint, params, data)

    # ── 2. SUPABASE (obrigatorio) ──
    try:
        supabase_client.save_api_response(endpoint, params, data)
    except Exception:
        pass  # Nao bloquear pipeline se Supabase falhar


def _save_to_local_cache(endpoint: str, params: dict, data: dict):
    """Salva resposta APENAS no cache local (disco)."""
    key = _cache_key(endpoint, params)
    filepath = os.path.join(_API_CACHE_DIR, f"{key}.json")
    try:
        cached = {
            "_cached_at": datetime.now().isoformat(),
            "_endpoint": endpoint,
            "_params": params,
            "data": data,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(cached, f, ensure_ascii=False)
    except Exception as e:
        print(f"    [CACHE] Erro ao salvar localmente: {e}")


def get_api_cache_stats() -> dict:
    """Retorna estatisticas do cache local + Supabase."""
    local_files = 0
    local_size = 0
    if os.path.exists(_API_CACHE_DIR):
        files = [f for f in os.listdir(_API_CACHE_DIR) if f.endswith(".json")]
        local_files = len(files)
        local_size = sum(os.path.getsize(os.path.join(_API_CACHE_DIR, f)) for f in files)

    sb_stats = supabase_client.get_api_cache_stats_supabase()

    return {
        "local_files": local_files,
        "local_size_mb": round(local_size / (1024 * 1024), 2),
        "supabase_total": sb_stats.get("total", 0),
        "supabase_endpoints": sb_stats.get("endpoints", {}),
    }


# ═══════════════════════════════════════════════════════
# ESTRUTURAS DE DADOS
# ═══════════════════════════════════════════════════════

@dataclass
class TeamStats:
    """Vetor de estatísticas agregadas de um time."""
    team_id: int
    team_name: str
    attack_strength: float = 0.0
    defense_strength: float = 0.0
    home_goals_scored_avg: float = 0.0
    home_goals_conceded_avg: float = 0.0
    away_goals_scored_avg: float = 0.0
    away_goals_conceded_avg: float = 0.0
    shots_total_avg: float = 0.0           # Total de finalizações por jogo
    shots_on_target_avg: float = 0.0
    shots_blocked_avg: float = 0.0
    corners_avg: float = 0.0
    cards_avg: float = 0.0
    fouls_avg: float = 0.0
    possession_final_third: float = 0.0
    form_last10: list = field(default_factory=list)
    form_points: float = 0.0
    league_position: int = 0
    league_points: int = 0
    games_played: int = 0
    games_remaining: int = 0
    points_to_title: int = 99
    points_to_relegation: int = 99
    last_match_date: Optional[str] = None
    has_real_data: bool = False            # True = dados de standings reais da API


@dataclass
class WeatherData:
    temperature_c: float = 20.0
    wind_speed_kmh: float = 5.0
    rain_mm: float = 0.0
    humidity_pct: float = 50.0
    description: str = "N/D"


@dataclass
class RefereeStats:
    name: str = "Desconhecido"
    cards_per_game_avg: float = 4.0
    yellow_avg: float = 3.5
    red_avg: float = 0.2
    fouls_per_game_avg: float = 25.0


@dataclass
class MarketOdds:
    home_win: float = 2.0
    draw: float = 3.3
    away_win: float = 3.5
    over_25: float = 1.85
    under_25: float = 1.95
    btts_yes: float = 1.80
    btts_no: float = 2.00
    over_95_corners: float = 1.90
    under_95_corners: float = 1.90
    over_35_cards: float = 1.85
    under_35_cards: float = 1.95
    double_chance_1x: float = 0.0   # Casa ou Empate (1X)
    double_chance_x2: float = 0.0   # Fora ou Empate (X2)
    double_chance_12: float = 0.0   # Casa ou Fora (12)
    asian_handicap_line: float = -0.5
    asian_handicap_home: float = 1.90
    asian_handicap_away: float = 1.90
    bookmaker: str = "N/D"
    # ── Todos os mercados da API (dict flexível) ──
    all_markets: dict = field(default_factory=dict)


@dataclass
class MatchAnalysis:
    match_id: int
    league_id: int
    league_name: str
    league_country: str
    match_date: str
    match_time: str
    venue_name: str = ""
    venue_lat: float = 0.0
    venue_lon: float = 0.0
    home_team: TeamStats = field(default_factory=TeamStats)
    away_team: TeamStats = field(default_factory=TeamStats)
    weather: WeatherData = field(default_factory=WeatherData)
    referee: RefereeStats = field(default_factory=RefereeStats)
    odds: MarketOdds = field(default_factory=MarketOdds)
    league_urgency_home: float = 0.5
    league_urgency_away: float = 0.5
    home_fatigue: bool = False
    away_fatigue: bool = False
    injuries_home: list = field(default_factory=list)
    injuries_away: list = field(default_factory=list)
    model_home_xg: float = 0.0
    model_away_xg: float = 0.0
    # Parâmetros α/β REAIS usados no cálculo de xG (para exibição na UI)
    model_alpha_h: float = 1.0
    model_beta_h: float = 1.0
    model_alpha_a: float = 1.0
    model_beta_a: float = 1.0
    model_prob_home: float = 0.0
    model_prob_draw: float = 0.0
    model_prob_away: float = 0.0
    model_prob_over25: float = 0.0
    model_prob_btts: float = 0.0
    model_corners_expected: float = 0.0
    model_cards_expected: float = 0.0
    model_home_shots_expected: float = 0.0
    model_away_shots_expected: float = 0.0
    model_home_sot_expected: float = 0.0
    model_away_sot_expected: float = 0.0
    model_total_shots_expected: float = 0.0
    model_total_sot_expected: float = 0.0
    score_matrix: Optional[np.ndarray] = None
    # ── Probabilidades expandidas (TODOS os mercados) ──
    model_probs: dict = field(default_factory=dict)
    # ── Qualidade dos dados ──
    has_real_odds: bool = False             # True = odds vieram de bookmaker real (API)
    has_real_standings: bool = False        # True = pelo menos 1 time tem standings reais
    has_real_weather: bool = False          # True = clima veio da API
    data_quality_score: float = 0.0        # 0.0 a 1.0 — índice de confiança dos dados
    # ── Validação casa/fora ──
    odds_home_away_suspect: bool = False   # True = odds sugerem que mandante/visitante pode estar invertido
    # ── Média de gols da liga (calculada dos standings) ──
    league_avg_goals: float = 2.7          # Gols por jogo na liga (default genérico)


# ═══════════════════════════════════════════════════════════════
#  API-FOOTBALL — CAMADA DE TRANSPORTE (com rate-limiting)
# ═══════════════════════════════════════════════════════════════

_api_call_count = 0


_api_cache_hits = 0
_api_cache_misses = 0


def _api_football_request(endpoint: str, params: dict, cache_only: bool = False) -> dict:
    """Chamada genérica com rate-limiting e CACHE à API-Football v3.
    Verifica cache local antes de fazer chamada real.
    Se cache_only=True, retorna {} se cache miss (sem chamada real à API)."""
    global _api_call_count, _api_cache_hits, _api_cache_misses

    # ── VERIFICAR CACHE PRIMEIRO ──
    # (não cachear "status" pois é sempre necessário em tempo real)
    use_cache = endpoint != "status"
    if use_cache:
        cached = _get_cached_response(endpoint, params)
        if cached is not None:
            _api_cache_hits += 1
            # Log a cada 50 hits de cache para não poluir
            if _api_cache_hits % 50 == 1 or _api_cache_hits <= 5:
                print(f"    [CACHE] {endpoint}({params}) -> HIT (economia de 1 request)")
            return cached

    # Se cache_only, não fazer chamada real — retornar vazio
    if cache_only:
        return {}

    _api_cache_misses += 1
    # ── CHAMADA REAL À API ──
    url = f"https://{config.API_FOOTBALL_HOST}/{endpoint}"
    headers = {"x-apisports-key": config.API_FOOTBALL_KEY}

    time.sleep(config.API_CALL_DELAY)
    _api_call_count += 1

    for attempt in range(2):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=30)
            remaining = resp.headers.get("x-ratelimit-requests-remaining", "?")
            print(f"    [API] {endpoint}({params}) -> {resp.status_code} | restantes={remaining}")

            if resp.status_code == 429:
                print("    [API] Rate limit HTTP 429! Aguardando 65s...")
                time.sleep(65)
                continue

            resp.raise_for_status()
            data = resp.json()

            errors = data.get("errors", {})
            if errors:
                if "rateLimit" in errors:
                    print(f"    [API] Rate limit no body! Aguardando 65s...")
                    time.sleep(65)
                    continue
                elif "plan" in errors:
                    return {}
                else:
                    print(f"    [API] Erros: {errors}")
                    return {}

            # ── SALVAR NO CACHE ──
            if use_cache and data:
                _save_to_cache(endpoint, params, data)

            return data
        except Exception as e:
            print(f"    [API] Falha em {endpoint}: {e}")
            return {}

    return {}


# ═══════════════════════════════════════════════════════════════
#  PIPELINE DE DADOS REAIS
# ═══════════════════════════════════════════════════════════════

def _fetch_fixtures(date: str) -> list[dict]:
    """Busca TODOS os jogos agendados para uma data."""
    print(f"  [ETL] Buscando fixtures para {date}...")
    data = _api_football_request("fixtures", {"date": date})
    raw = data.get("response", [])

    # Contagens por status
    status_counts = {}
    for f in raw:
        st = f.get("fixture", {}).get("status", {}).get("short", "?")
        status_counts[st] = status_counts.get(st, 0) + 1
    print(f"  [ETL] Status breakdown: {status_counts}")

    # Prioridade: jogos não iniciados
    ns_fixtures = [f for f in raw if f.get("fixture", {}).get("status", {}).get("short", "") == "NS"]

    # Se não há NS, incluir jogos em andamento
    live_status = {"1H", "HT", "2H", "LIVE", "ET", "P", "BT"}
    live_fixtures = [f for f in raw if f.get("fixture", {}).get("status", {}).get("short", "") in live_status]

    # Se não há NS nem live, incluir TBD
    tbd_fixtures = [f for f in raw if f.get("fixture", {}).get("status", {}).get("short", "") in {"TBD", "SUSP", "PST"}]

    # Combinar: NS primeiro, depois live, depois TBD
    fixtures = ns_fixtures + live_fixtures + tbd_fixtures

    # Se ainda 0, incluir jogos finalizados (FT) para análise retroativa
    if not fixtures and raw:
        ft_fixtures = [f for f in raw if f.get("fixture", {}).get("status", {}).get("short", "") in {"FT", "AET", "PEN"}]
        if ft_fixtures:
            print(f"  [ETL] ⚠️  Sem jogos futuros — incluindo {len(ft_fixtures)} jogos finalizados para análise")
            fixtures = ft_fixtures

    print(f"  [ETL] {len(fixtures)} jogos selecionados para {date} (de {len(raw)} total)")
    return fixtures


def _fetch_standings(league_id: int, season: int) -> list[dict]:
    """Busca classificação de uma liga/temporada.
    Unifica TODOS os grupos (conferências, grupos de fase, etc.)
    para garantir que todos os times sejam encontrados."""
    data = _api_football_request("standings", {"league": league_id, "season": season})
    response = data.get("response", [])
    if not response:
        return []
    league_data = response[0].get("league", {})
    standings_groups = league_data.get("standings", [])
    # Unificar TODOS os grupos em uma única lista
    # (Ex: Carioca Grupo A + Grupo B, Conferência Leste + Oeste, etc.)
    if standings_groups and isinstance(standings_groups[0], list):
        all_teams = []
        for group in standings_groups:
            all_teams.extend(group)
        return all_teams
    return standings_groups


def _fetch_odds_for_fixture(fixture_id: int) -> dict:
    """Busca odds para um fixture específico."""
    data = _api_football_request("odds", {"fixture": fixture_id})
    response = data.get("response", [])
    return response[0] if response else {}


def _fetch_injuries_for_fixture(fixture_id: int) -> dict:
    """Busca lesões para um fixture."""
    data = _api_football_request("injuries", {"fixture": fixture_id})
    return data.get("response", [])


def _fetch_weather_by_city(city: str, country_code: str = "") -> WeatherData:
    """Busca clima via OpenWeatherMap usando nome da cidade.
    Usa cache local + Supabase para nao repetir chamadas."""
    if not config.OPENWEATHER_KEY or not city:
        return WeatherData()

    q = f"{city},{country_code}" if country_code else city
    weather_params = {"q": q, "units": "metric"}

    # ── Verificar cache (local + Supabase) ──
    cached = _get_cached_response("weather", weather_params)
    if cached is not None:
        return _parse_weather_response(cached)

    # ── Chamada real à API ──
    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"q": q, "appid": config.OPENWEATHER_KEY, "units": "metric", "lang": "pt_br"}
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            return WeatherData()
        d = resp.json()

        # ── Salvar no cache (local + Supabase) ──
        _save_to_cache("weather", weather_params, d)

        return _parse_weather_response(d)
    except Exception:
        return WeatherData()


def _parse_weather_response(d: dict) -> WeatherData:
    """Converte resposta JSON do OpenWeatherMap em WeatherData."""
    return WeatherData(
        temperature_c=round(d.get("main", {}).get("temp", 20.0), 1),
        wind_speed_kmh=round(d.get("wind", {}).get("speed", 0) * 3.6, 1),
        rain_mm=round(d.get("rain", {}).get("1h", 0.0), 1),
        humidity_pct=d.get("main", {}).get("humidity", 50),
        description=d.get("weather", [{}])[0].get("description", "N/D"),
    )


# ═══════════════════════════════════════════════════════════════
#  PARSING — Converter JSON da API em objetos do sistema
# ═══════════════════════════════════════════════════════════════

COUNTRY_CODES = {
    "England": "GB", "Spain": "ES", "Italy": "IT", "Germany": "DE",
    "France": "FR", "Portugal": "PT", "Netherlands": "NL", "Belgium": "BE",
    "Turkey": "TR", "Greece": "GR", "Scotland": "GB", "Brazil": "BR",
    "Argentina": "AR", "USA": "US", "Mexico": "MX", "Japan": "JP",
    "South-Korea": "KR", "China": "CN", "Australia": "AU", "Saudi-Arabia": "SA",
    "Egypt": "EG", "Colombia": "CO", "Chile": "CL", "Uruguay": "UY",
    "Paraguay": "PY", "Peru": "PE", "Ecuador": "EC", "Russia": "RU",
    "Ukraine": "UA", "Poland": "PL", "Czech-Republic": "CZ", "Austria": "AT",
    "Switzerland": "CH", "Denmark": "DK", "Sweden": "SE", "Norway": "NO",
    "Finland": "FI", "Croatia": "HR", "Serbia": "RS", "Romania": "RO",
    "Hungary": "HU", "Bulgaria": "BG", "Ireland": "IE", "Wales": "GB",
    "Israel": "IL", "India": "IN", "Indonesia": "ID", "Thailand": "TH",
    "Vietnam": "VN", "Malaysia": "MY", "Algeria": "DZ", "Morocco": "MA",
    "Tunisia": "TN", "Nigeria": "NG", "South-Africa": "ZA", "Kenya": "KE",
    "Ghana": "GH", "Cameroon": "CM", "Ivory-Coast": "CI", "Senegal": "SN",
    "Costa-Rica": "CR", "Honduras": "HN", "Jamaica": "JM", "Panama": "PA",
    "Bolivia": "BO", "Venezuela": "VE", "Canada": "CA", "Iceland": "IS",
    "Cyprus": "CY", "Malta": "MT", "Luxembourg": "LU", "Albania": "AL",
    "Bosnia-and-Herzegovina": "BA", "North-Macedonia": "MK", "Montenegro": "ME",
    "Georgia": "GE", "Armenia": "AM", "Azerbaijan": "AZ", "Kazakhstan": "KZ",
    "Uzbekistan": "UZ", "Iran": "IR", "Iraq": "IQ", "Jordan": "JO",
    "Qatar": "QA", "UAE": "AE", "Bahrain": "BH", "Kuwait": "KW", "Oman": "OM",
    "World": "", "Europe": "", "Africa": "", "Asia": "", "South-America": "",
    "North-&-Central-America": "",
}


def _parse_form_string(form_str: str) -> list[str]:
    """Converte string de forma 'WWDLW' em lista ['W','W','D','L','W'].
    NÃO duplica — retorna apenas os resultados reais da API."""
    if not form_str:
        return []          # sem dados reais → lista vazia
    result = []
    for ch in form_str.upper():
        if ch in ("W", "D", "L"):
            result.append(ch)
    return result          # sem duplicação — mostra só o que a API retorna


def _form_points(form: list[str]) -> float:
    """Pontos normalizados da forma (0-1)."""
    pts = sum(3 if r == "W" else 1 if r == "D" else 0 for r in form)
    return pts / (3.0 * max(1, len(form)))


def _build_team_from_standings(team_id: int, team_name: str,
                                standings: list[dict],
                                is_home: bool) -> TeamStats:
    """Constrói TeamStats a partir dos dados de classificação da API."""
    team_data = None
    for entry in standings:
        tid = entry.get("team", {}).get("id", 0)
        if tid == team_id:
            team_data = entry
            break

    if not team_data:
        # Time não encontrado no standings — usar defaults
        # has_real_data = False → dados NÃO CONFIÁVEIS
        return TeamStats(
            team_id=team_id, team_name=team_name,
            attack_strength=1.2, defense_strength=1.0,
            home_goals_scored_avg=1.3, home_goals_conceded_avg=1.1,
            away_goals_scored_avg=1.0, away_goals_conceded_avg=1.3,
            shots_total_avg=11.0, shots_on_target_avg=4.0, shots_blocked_avg=3.0,
            corners_avg=5.0, cards_avg=2.0, fouls_avg=12.0,
            possession_final_third=30.0,
            form_last10=[], form_points=0.0,       # ← sem dados reais de forma
            league_position=0, league_points=0,
            games_played=0, games_remaining=0,
            has_real_data=False,
        )

    # Extrair dados do standing
    rank = team_data.get("rank", 10)
    points = team_data.get("points", 0)
    form_str = team_data.get("form", "DDDDD")

    all_stats = team_data.get("all", {})
    home_stats = team_data.get("home", {})
    away_stats = team_data.get("away", {})

    all_played = all_stats.get("played", 1) or 1
    home_played = home_stats.get("played", 1) or 1
    away_played = away_stats.get("played", 1) or 1

    home_gf = home_stats.get("goals", {}).get("for", 0) or 0
    home_ga = home_stats.get("goals", {}).get("against", 0) or 0
    away_gf = away_stats.get("goals", {}).get("for", 0) or 0
    away_ga = away_stats.get("goals", {}).get("against", 0) or 0

    total_gf = (home_gf + away_gf) or 1
    total_ga = (home_ga + away_ga) or 1

    # Calcular médias de gols da liga inteira
    league_total_gf = 0
    league_total_played = 0
    for e in standings:
        a = e.get("all", {})
        league_total_gf += (a.get("goals", {}).get("for", 0) or 0)
        league_total_played += (a.get("played", 0) or 0)

    league_avg_gpg = (league_total_gf / max(1, league_total_played)) * 2  # gols por jogo
    league_avg_gpg = max(1.5, league_avg_gpg)

    # Força de ataque e defesa relativas à liga
    team_gpg_scored = total_gf / all_played
    team_gpg_conceded = total_ga / all_played
    half_avg = league_avg_gpg / 2.0

    attack = max(0.3, team_gpg_scored / max(0.3, half_avg))
    defense = max(0.3, team_gpg_conceded / max(0.3, half_avg))

    form = _parse_form_string(form_str)

    # Pontos para título e rebaixamento
    top_pts = standings[0].get("points", points) if standings else points
    n_teams = len(standings) or 20
    relegation_zone_idx = max(0, n_teams - 3)
    rel_pts = standings[relegation_zone_idx].get("points", 0) if len(standings) > relegation_zone_idx else 0
    total_rounds = max(all_played + 5, 34)  # estimativa

    return TeamStats(
        team_id=team_id,
        team_name=team_name,
        attack_strength=round(attack, 3),
        defense_strength=round(defense, 3),
        home_goals_scored_avg=round(home_gf / home_played, 2),
        home_goals_conceded_avg=round(home_ga / home_played, 2),
        away_goals_scored_avg=round(away_gf / away_played, 2),
        away_goals_conceded_avg=round(away_ga / away_played, 2),
        shots_total_avg=round(attack * 9.5, 1),       # ESTIMADO (~10-13 per team)
        shots_on_target_avg=round(attack * 3.5, 1),
        shots_blocked_avg=round(defense * 3.0, 1),
        corners_avg=round(attack * 3.8, 1),         # ESTIMADO (API não fornece corners)
        cards_avg=round(1.5 + defense * 0.8, 1),    # ESTIMADO
        fouls_avg=round(10.0 + defense * 3.0, 1),   # ESTIMADO
        possession_final_third=round(attack * 25.0, 1),  # ESTIMADO
        form_last10=form,
        form_points=_form_points(form),
        league_position=rank,
        league_points=points,
        games_played=all_played,
        games_remaining=max(0, total_rounds - all_played),
        points_to_title=max(0, top_pts - points),
        points_to_relegation=max(0, points - rel_pts),
        has_real_data=True,  # ← dados REAIS da API de standings
    )


def _bet_to_market_key(bet_id: int, bet_name: str) -> str | None:
    """Classifica um bet da API em uma chave de mercado all_markets."""
    bn = bet_name.lower()

    if bet_id == 1 or "match winner" in bn:
        return "1x2"
    if bet_id == 5 or \
       ("goals" in bn and "over" in bn and "half" not in bn and "second" not in bn) or \
       ("over/under" in bn and "half" not in bn and "team" not in bn):
        return "goals_ou"
    if bet_id in (6, 25) or ("first half" in bn and "over" in bn) or ("1st half" in bn and "over" in bn):
        return "ht_goals_ou"
    if bet_id in (7, 26) or ("second half" in bn and "over" in bn) or ("2nd half" in bn and "over" in bn):
        return "h2_goals_ou"
    if bet_id == 34 or ("both teams" in bn and "first half" in bn):
        return "btts_ht"
    if bet_id == 35 or ("both teams" in bn and "second half" in bn):
        return "btts_h2"
    if bet_id == 24 or ("result" in bn and "both teams" in bn):
        return "result_btts"
    if bet_id == 49 or ("total" in bn and "both teams" in bn):
        return "total_btts"
    if bet_id == 8 or bn in ("both teams score", "both teams to score"):
        return "btts"
    if bet_id == 9 or "exact score" in bn:
        return "exact_score"
    if bet_id == 11 or ("half time" in bn and "full time" in bn):
        return "ht_ft"
    if bet_id == 23 or ("double chance" in bn and "half" in bn):
        return "ht_double_chance"
    if bet_id == 12 or "double chance" in bn:
        return "double_chance"
    if bet_id == 13 or "first half winner" in bn:
        return "ht_result"
    if bet_id in (14, 63) or ("home" in bn and "goal" in bn and "over" in bn):
        return "home_goals_ou"
    if bet_id in (15, 64) or ("away" in bn and "goal" in bn and "over" in bn):
        return "away_goals_ou"
    if bet_id in (16, 21) or ("odd/even" in bn and "half" not in bn and "home" not in bn and "away" not in bn):
        return "odd_even"
    if bet_id == 17 or ("clean sheet" in bn and "home" in bn):
        return "cs_home"
    if bet_id == 18 or ("clean sheet" in bn and "away" in bn):
        return "cs_away"
    if bet_id == 19 or ("win to nil" in bn and "home" in bn):
        return "wtn_home"
    if bet_id == 20 or ("win to nil" in bn and "away" in bn):
        return "wtn_away"
    if bet_id == 22 or "win both halves" in bn:
        return "win_both_halves"
    if bet_id == 27 or ("both halves" in bn and "score" in bn):
        return "both_halves_score"
    if bet_id == 28 or ("result" in bn and "total" in bn):
        return "result_total"
    if bet_id == 4 or "asian" in bn or "handicap" in bn:
        return "asian_handicap"
    if "corner" in bn:
        return "corners_ou"
    if "card" in bn:
        return "cards_ou"
    # Shots markets
    if bet_id == 87 or ("shotongoal" in bn.replace(" ", "") and "1x2" not in bn):
        return "sot_ou"
    if bet_id == 176 or ("shotontarget" in bn.replace(" ", "") and "1x2" in bn):
        return "sot_1x2"
    if bet_id == 340 or ("shots" in bn and "1x2" in bn):
        return "shots_1x2"
    if "shot" in bn and "over" in bn and "player" not in bn and "target" not in bn and "goal" not in bn:
        return "shots_ou"
    if "shot" in bn and "target" in bn and "over" in bn and "player" not in bn and "1x2" not in bn:
        return "sot_ou"
    if "home" in bn and "shot" in bn:
        return "home_shots_ou"
    if "away" in bn and "shot" in bn:
        return "away_shots_ou"
    if "player" in bn and "shot" in bn:
        return "player_shots_ou"
    return None


def _parse_odds_response(odds_raw: dict) -> MarketOdds:
    """Converte resposta de odds da API em MarketOdds."""
    bookmakers = odds_raw.get("bookmakers", [])
    if not bookmakers:
        return MarketOdds()

    # Escolher bookmaker preferido (Bet365 é prioridade)
    chosen = None
    for pref in config.PREFERRED_BOOKMAKERS:
        for bk in bookmakers:
            if pref.lower() in bk.get("name", "").lower():
                chosen = bk
                break
        if chosen:
            break

    if not chosen:
        chosen = bookmakers[0]  # Fallback: primeiro disponível

    bk_name = chosen.get("name", "Desconhecido")
    bets = chosen.get("bets", [])

    odds = MarketOdds(bookmaker=bk_name)

    for bet in bets:
        bet_id = bet.get("id", 0)
        bet_name = bet.get("name", "").lower()
        values = bet.get("values", [])

        val_map = {}
        for v in values:
            val_map[str(v.get("value", "")).lower()] = float(v.get("odd", 0))

        # ═══ 1X2 — Match Winner (bet id 1) ═══
        if bet_id == 1 or "match winner" in bet_name:
            odds.home_win = val_map.get("home", odds.home_win)
            odds.draw = val_map.get("draw", odds.draw)
            odds.away_win = val_map.get("away", odds.away_win)
            odds.all_markets["1x2"] = {
                "home": val_map.get("home", 0), "draw": val_map.get("draw", 0),
                "away": val_map.get("away", 0)
            }

        # ═══ Goals Over/Under — TODAS as linhas (bet id 5) ═══
        elif bet_id == 5 or \
             ("goals" in bet_name and "over" in bet_name and "half" not in bet_name and "second" not in bet_name) or \
             ("over/under" in bet_name and "half" not in bet_name and "team" not in bet_name):
            odds.over_25 = val_map.get("over 2.5", odds.over_25)
            odds.under_25 = val_map.get("under 2.5", odds.under_25)
            g_ou = {}
            for k, v in val_map.items():
                g_ou[k.replace(" ", "_")] = v
            odds.all_markets["goals_ou"] = g_ou

        # ═══ First Half Goals O/U (bet id 6 / 25) ═══
        elif bet_id in (6, 25) or ("first half" in bet_name and "over" in bet_name) or \
             ("1st half" in bet_name and "over" in bet_name):
            ht_ou = {}
            for k, v in val_map.items():
                ht_ou[k.replace(" ", "_")] = v
            odds.all_markets["ht_goals_ou"] = ht_ou

        # ═══ Second Half Goals O/U (bet id 7 / 26) ═══
        elif bet_id in (7, 26) or ("second half" in bet_name and "over" in bet_name) or \
             ("2nd half" in bet_name and "over" in bet_name):
            h2_ou = {}
            for k, v in val_map.items():
                h2_ou[k.replace(" ", "_")] = v
            odds.all_markets["h2_goals_ou"] = h2_ou

        # ═══ BTTS 1° Tempo — Both Teams Score First Half (bet id 34) ═══
        elif bet_id == 34 or ("both teams" in bet_name and "first half" in bet_name):
            odds.all_markets["btts_ht"] = {"yes": val_map.get("yes", 0), "no": val_map.get("no", 0)}

        # ═══ BTTS 2° Tempo — Both Teams Score Second Half (bet id 35) ═══
        elif bet_id == 35 or ("both teams" in bet_name and "second half" in bet_name):
            odds.all_markets["btts_h2"] = {"yes": val_map.get("yes", 0), "no": val_map.get("no", 0)}

        # ═══ Resultado + BTTS — Results/Both Teams Score (bet id 24) ═══
        elif bet_id == 24 or ("result" in bet_name and "both teams" in bet_name):
            odds.all_markets["result_btts"] = dict(val_map)

        # ═══ Total Gols + BTTS — Total Goals/Both Teams To Score (bet id 49) ═══
        elif bet_id == 49 or ("total" in bet_name and "both teams" in bet_name):
            odds.all_markets["total_btts"] = dict(val_map)

        # ═══ BTTS — Both Teams Score (bet id 8) — PARTIDA COMPLETA ═══
        # IMPORTANTE: este bloco deve ficar DEPOIS dos mercados BTTS
        # derivados (1° Tempo, 2° Tempo, Result/BTTS) para evitar que
        # a condição genérica "both teams" capture mercados errados.
        elif bet_id == 8 or bet_name in ("both teams score", "both teams to score"):
            odds.btts_yes = val_map.get("yes", odds.btts_yes)
            odds.btts_no = val_map.get("no", odds.btts_no)
            odds.all_markets["btts"] = {"yes": val_map.get("yes", 0), "no": val_map.get("no", 0)}

        # ═══ Exact Score (bet id 9) ═══
        elif bet_id == 9 or "exact score" in bet_name:
            odds.all_markets["exact_score"] = dict(val_map)

        # ═══ HT/FT (bet id 11) ═══
        elif bet_id == 11 or ("half time" in bet_name and "full time" in bet_name):
            odds.all_markets["ht_ft"] = dict(val_map)

        # ═══ Double Chance (bet id 12) ═══
        elif bet_id == 12 or "double chance" in bet_name:
            odds.double_chance_1x = val_map.get("home/draw", odds.double_chance_1x)
            odds.double_chance_x2 = val_map.get("draw/away", odds.double_chance_x2)
            odds.double_chance_12 = val_map.get("home/away", odds.double_chance_12)
            odds.all_markets["double_chance"] = dict(val_map)

        # ═══ First Half Winner (bet id 13) ═══
        elif bet_id == 13 or "first half winner" in bet_name or "1st half" in bet_name:
            odds.all_markets["ht_result"] = {
                "home": val_map.get("home", 0), "draw": val_map.get("draw", 0),
                "away": val_map.get("away", 0)
            }

        # ═══ Home Team Goals O/U (bet id 14 / 63) ═══
        elif bet_id in (14, 63) or ("home" in bet_name and "goal" in bet_name and "over" in bet_name):
            hg_ou = {}
            for k, v in val_map.items():
                hg_ou[k.replace(" ", "_")] = v
            odds.all_markets["home_goals_ou"] = hg_ou

        # ═══ Away Team Goals O/U (bet id 15 / 64) ═══
        elif bet_id in (15, 64) or ("away" in bet_name and "goal" in bet_name and "over" in bet_name):
            ag_ou = {}
            for k, v in val_map.items():
                ag_ou[k.replace(" ", "_")] = v
            odds.all_markets["away_goals_ou"] = ag_ou

        # ═══ Odd/Even (bet id 16 / 21) ═══
        elif bet_id in (16, 21) or ("odd/even" in bet_name and "half" not in bet_name and "home" not in bet_name and "away" not in bet_name):
            odds.all_markets["odd_even"] = {"odd": val_map.get("odd", 0), "even": val_map.get("even", 0)}

        # ═══ Clean Sheet Home (bet id 17) ═══
        elif bet_id == 17 or ("clean sheet" in bet_name and "home" in bet_name):
            odds.all_markets["cs_home"] = {"yes": val_map.get("yes", 0), "no": val_map.get("no", 0)}

        # ═══ Clean Sheet Away (bet id 18) ═══
        elif bet_id == 18 or ("clean sheet" in bet_name and "away" in bet_name):
            odds.all_markets["cs_away"] = {"yes": val_map.get("yes", 0), "no": val_map.get("no", 0)}

        # ═══ Win to Nil Home (bet id 19) ═══
        elif bet_id == 19 or ("win to nil" in bet_name and "home" in bet_name):
            odds.all_markets["wtn_home"] = {"yes": val_map.get("yes", 0), "no": val_map.get("no", 0)}

        # ═══ Win to Nil Away (bet id 20) ═══
        elif bet_id == 20 or ("win to nil" in bet_name and "away" in bet_name):
            odds.all_markets["wtn_away"] = {"yes": val_map.get("yes", 0), "no": val_map.get("no", 0)}

        # ═══ Win Both Halves (bet id 22) ═══
        elif bet_id == 22 or "win both halves" in bet_name:
            odds.all_markets["win_both_halves"] = dict(val_map)

        # ═══ Double Chance First Half (bet id 23) ═══
        elif bet_id == 23 or ("double chance" in bet_name and "half" in bet_name):
            odds.all_markets["ht_double_chance"] = dict(val_map)

        # ═══ Both Halves Over (bet id 27) / To Score In Both Halves ═══
        elif bet_id == 27 or "both halves" in bet_name:
            odds.all_markets["both_halves_score"] = dict(val_map)

        # ═══ Result / Total Goals (bet id 28) ═══
        elif bet_id == 28 or ("result" in bet_name and "total" in bet_name):
            odds.all_markets["result_total"] = dict(val_map)

        # ═══ Asian Handicap (bet id 4) ═══
        elif bet_id == 4 or "asian" in bet_name or "handicap" in bet_name:
            for v in values:
                val_str = str(v.get("value", ""))
                odd_v = float(v.get("odd", 0))
                if "home" in val_str.lower():
                    odds.asian_handicap_home = odd_v
                elif "away" in val_str.lower():
                    odds.asian_handicap_away = odd_v
            odds.all_markets["asian_handicap"] = dict(val_map)

        # ═══ Corners (varios bet ids) ═══
        elif "corner" in bet_name:
            c_ou = {}
            for k, v in val_map.items():
                c_ou[k.replace(" ", "_")] = v
            odds.all_markets["corners_ou"] = c_ou
            # Compatibilidade
            odds.over_95_corners = val_map.get("over 9.5", odds.over_95_corners)
            odds.under_95_corners = val_map.get("under 9.5", odds.under_95_corners)

        # ═══ Cards (varios bet ids) ═══
        elif "card" in bet_name:
            k_ou = {}
            for k, v in val_map.items():
                k_ou[k.replace(" ", "_")] = v
            odds.all_markets["cards_ou"] = k_ou
            odds.over_35_cards = val_map.get("over 3.5", odds.over_35_cards)
            odds.under_35_cards = val_map.get("under 3.5", odds.under_35_cards)

        # ═══ Total ShotOnGoal O/U (bet id 87 — Bet365) ═══
        # Valores: "Over 7.5", "Under 7.5" etc — finalizações ao gol (SoT)
        elif bet_id == 87 or ("shotongoal" in bet_name.replace(" ", "") and "1x2" not in bet_name):
            sot_ou = {}
            for k, v in val_map.items():
                sot_ou[k.replace(" ", "_")] = v
            odds.all_markets["sot_ou"] = sot_ou

        # ═══ ShotOnTarget 1x2 (bet id 176 — Bet365) ═══
        # Valores: "Home", "Draw", "Away" — qual time terá mais finalizações ao gol
        elif bet_id == 176 or ("shotontarget" in bet_name.replace(" ", "") and "1x2" in bet_name):
            odds.all_markets["sot_1x2"] = {
                "home": val_map.get("home", 0),
                "draw": val_map.get("draw", 0),
                "away": val_map.get("away", 0),
            }

        # ═══ Shots.1x2 (bet id 340 — Bet365) ═══
        # Valores: "Home", "Draw", "Away" — qual time terá mais finalizações totais
        elif bet_id == 340 or ("shots" in bet_name and "1x2" in bet_name):
            odds.all_markets["shots_1x2"] = {
                "home": val_map.get("home", 0),
                "draw": val_map.get("draw", 0),
                "away": val_map.get("away", 0),
            }

        # ═══ Total Shots O/U (genérico — caso a API forneça futuramente) ═══
        elif "shot" in bet_name and "over" in bet_name and "player" not in bet_name and "target" not in bet_name and "goal" not in bet_name:
            s_ou = {}
            for k, v in val_map.items():
                s_ou[k.replace(" ", "_")] = v
            odds.all_markets["shots_ou"] = s_ou

        # ═══ SoT O/U genérico (caso não foi capturado pelo id 87) ═══
        elif "shot" in bet_name and "target" in bet_name and "over" in bet_name and "player" not in bet_name and "1x2" not in bet_name:
            sot_ou = {}
            for k, v in val_map.items():
                sot_ou[k.replace(" ", "_")] = v
            if "sot_ou" not in odds.all_markets:
                odds.all_markets["sot_ou"] = sot_ou

        # ═══ Home Team Shots O/U ═══
        elif "home" in bet_name and "shot" in bet_name:
            hs_ou = {}
            for k, v in val_map.items():
                hs_ou[k.replace(" ", "_")] = v
            odds.all_markets["home_shots_ou"] = hs_ou

        # ═══ Away Team Shots O/U ═══
        elif "away" in bet_name and "shot" in bet_name:
            as_ou = {}
            for k, v in val_map.items():
                as_ou[k.replace(" ", "_")] = v
            odds.all_markets["away_shots_ou"] = as_ou

        # ═══ Player Shots O/U ═══
        elif "player" in bet_name and "shot" in bet_name:
            ps_ou = {}
            for k, v in val_map.items():
                ps_ou[k.replace(" ", "_")] = v
            if "player_shots_ou" not in odds.all_markets:
                odds.all_markets["player_shots_ou"] = {}
            odds.all_markets["player_shots_ou"].update(ps_ou)

        # ═══ Catch-all: qualquer outro mercado ═══
        else:
            if val_map and bet_name:
                key = bet_name.replace(" ", "_").replace("/", "_")[:40]
                odds.all_markets[key] = dict(val_map)

    # ═══════════════════════════════════════════════════════════════════
    # SEGUNDA PASSAGEM: Mercados especializados de OUTROS bookmakers
    # Ex: Pinnacle pode ter Shots 1x2, ShotOnGoal O/U que Bet365 não tem
    # (ou vice-versa) — preencher mercados que NÃO existem no bookmaker principal
    # ═══════════════════════════════════════════════════════════════════
    _SPECIALIZED_BET_IDS = {87, 176, 340, 212, 213, 214, 215}  # shots, player props
    
    for bk in bookmakers:
        if bk.get("name", "") == bk_name:
            continue  # Já processado acima
        for bet in bk.get("bets", []):
            bet_id = bet.get("id", 0)
            bet_name_extra = bet.get("name", "").lower()
            
            # Só buscar mercados especializados que não existem no bookmaker principal
            is_shots = bet_id in _SPECIALIZED_BET_IDS or "shot" in bet_name_extra
            if not is_shots:
                continue
            
            values = bet.get("values", [])
            val_map = {}
            for v in values:
                val_map[str(v.get("value", "")).lower()] = float(v.get("odd", 0))
            
            if not val_map:
                continue
            
            # Total ShotOnGoal O/U (bet id 87)
            if bet_id == 87 or ("shotongoal" in bet_name_extra.replace(" ", "") and "1x2" not in bet_name_extra):
                if "sot_ou" not in odds.all_markets:
                    sot_ou = {}
                    for k, v in val_map.items():
                        sot_ou[k.replace(" ", "_")] = v
                    odds.all_markets["sot_ou"] = sot_ou
            
            # ShotOnTarget 1x2 (bet id 176)
            elif bet_id == 176 or ("shotontarget" in bet_name_extra.replace(" ", "") and "1x2" in bet_name_extra):
                if "sot_1x2" not in odds.all_markets:
                    odds.all_markets["sot_1x2"] = {
                        "home": val_map.get("home", 0),
                        "draw": val_map.get("draw", 0),
                        "away": val_map.get("away", 0),
                        "_source": bk.get("name", "?"),
                    }
            
            # Shots.1x2 (bet id 340)
            elif bet_id == 340 or ("shots" in bet_name_extra and "1x2" in bet_name_extra):
                if "shots_1x2" not in odds.all_markets:
                    odds.all_markets["shots_1x2"] = {
                        "home": val_map.get("home", 0),
                        "draw": val_map.get("draw", 0),
                        "away": val_map.get("away", 0),
                        "_source": bk.get("name", "?"),
                    }

    # ═══════════════════════════════════════════════════════════════════
    # TERCEIRA PASSAGEM: Coletar odds de TODOS os bookmakers por mercado
    # Armazena em all_markets[mkt]["_bookmakers"] = {bk_name: {sel: odd}}
    # Permite comparação de odds entre casas no frontend
    # ═══════════════════════════════════════════════════════════════════
    for bk in bookmakers:
        current_bk = bk.get("name", "Desconhecido")
        for bet in bk.get("bets", []):
            bet_id = bet.get("id", 0)
            bet_name_raw = bet.get("name", "")
            mk = _bet_to_market_key(bet_id, bet_name_raw)
            if not mk:
                continue

            values = bet.get("values", [])
            val_map = {}
            for v in values:
                val_map[str(v.get("value", "")).lower().replace(" ", "_")] = float(v.get("odd", 0))
            if not val_map:
                continue

            # Só adicionar bookmaker data para mercados que existem em all_markets
            if mk not in odds.all_markets:
                continue

            # Inicializar sub-dict _bookmakers se não existe
            if "_bookmakers" not in odds.all_markets[mk]:
                odds.all_markets[mk]["_bookmakers"] = {}

            odds.all_markets[mk]["_bookmakers"][current_bk] = val_map

    return odds


def _parse_injuries(injuries_raw: list, team_id: int) -> list[str]:
    """Converte resposta de lesões da API em lista de strings."""
    result = []
    for inj in injuries_raw:
        player_data = inj.get("player", {})
        if inj.get("team", {}).get("id") == team_id:
            name = player_data.get("name", "Desconhecido")
            reason = player_data.get("reason", "N/D")
            ptype = player_data.get("type", "")
            result.append(f"{name} ({reason} - {ptype})")
    return result


def _parse_fixture_to_match(
    fix_raw: dict,
    standings_cache: dict,
    odds_cache: dict,
    injuries_cache: dict,
) -> Optional[MatchAnalysis]:
    """Converte um fixture JSON da API em MatchAnalysis completo."""
    try:
        fixture = fix_raw.get("fixture", {})
        league = fix_raw.get("league", {})
        teams = fix_raw.get("teams", {})

        fix_id = fixture.get("id", 0)
        league_id = league.get("id", 0)
        league_name = league.get("name", "Desconhecida")
        league_country = league.get("country", "N/D")

        # Data e hora (converter UTC → Brasília)
        date_str = fixture.get("date", "")
        try:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            dt_br = dt.astimezone(config.BR_TIMEZONE)
            match_date = dt_br.strftime("%Y-%m-%d")
            match_time = dt_br.strftime("%H:%M")
        except (ValueError, TypeError):
            match_date = config.TODAY
            match_time = "00:00"

        # Venue
        venue = fixture.get("venue", {}) or {}
        venue_name = venue.get("name", "N/D") or "N/D"
        venue_city = venue.get("city", "") or ""

        # Times
        home_info = teams.get("home", {})
        away_info = teams.get("away", {})
        home_id = home_info.get("id", 0)
        away_id = away_info.get("id", 0)
        home_name = home_info.get("name", "Casa")
        away_name = away_info.get("name", "Fora")

        # Standings
        standings = standings_cache.get(league_id, [])
        home_stats = _build_team_from_standings(home_id, home_name, standings, True)
        away_stats = _build_team_from_standings(away_id, away_name, standings, False)

        # Calcular média real de gols da liga a partir dos standings
        _league_total_gf = 0
        _league_total_played = 0
        for _e in standings:
            _a = _e.get("all", {})
            _league_total_gf += (_a.get("goals", {}).get("for", 0) or 0)
            _league_total_played += (_a.get("played", 0) or 0)
        _league_avg_gpg = (_league_total_gf / max(1, _league_total_played)) * 2 if _league_total_played > 0 else 2.7
        _league_avg_gpg = max(1.5, _league_avg_gpg)

        # Odds
        odds_raw = odds_cache.get(fix_id, {})
        odds = _parse_odds_response(odds_raw) if odds_raw else MarketOdds()

        # Lesões
        injuries_raw = injuries_cache.get(fix_id, [])
        injuries_home = _parse_injuries(injuries_raw, home_id) if injuries_raw else []
        injuries_away = _parse_injuries(injuries_raw, away_id) if injuries_raw else []

        # Árbitro
        ref_data = fixture.get("referee")
        referee = RefereeStats()
        if ref_data and isinstance(ref_data, str):
            referee.name = ref_data

        # Calcular qualidade dos dados
        _has_real_odds = odds.bookmaker not in ("N/D", "Modelo (Estimado)", "")
        _has_real_standings = home_stats.has_real_data or away_stats.has_real_data
        _both_standings = home_stats.has_real_data and away_stats.has_real_data

        # Score de qualidade: 0.0 a 1.0
        #   +0.40 se AMBOS os times têm standings reais
        #   +0.20 se pelo menos 1 time tem standings reais
        #   +0.35 se tem odds REAIS de bookmaker
        #   +0.10 se tem árbitro identificado
        #   +0.15 se tem lesões checadas
        dq = 0.0
        if _both_standings:
            dq += 0.40
        elif _has_real_standings:
            dq += 0.20
        if _has_real_odds:
            dq += 0.35
        if referee.name != "Desconhecido":
            dq += 0.10
        if injuries_raw:
            dq += 0.15

        # ── Detecção de possível inversão Casa/Fora ──
        # Se a odd do "mandante" é muito maior que a do "visitante", a API
        # pode ter invertido quem é casa/fora.  Isso é comum em competições
        # continentais (Champions League, Libertadores) e jogos em campo neutro.
        _odds_suspect = False
        if _has_real_odds and odds.home_win > 1.0 and odds.away_win > 1.0:
            odds_ratio = odds.home_win / odds.away_win
            if odds_ratio > 2.0:
                _odds_suspect = True
                print(f"    [WARN] {home_name} vs {away_name}: "
                      f"odds sugerem possivel inversao casa/fora "
                      f"(home_odd={odds.home_win}, away_odd={odds.away_win}, "
                      f"ratio={odds_ratio:.2f}).  Modelo usara stats neutras.")

        return MatchAnalysis(
            match_id=fix_id,
            league_id=league_id,
            league_name=league_name,
            league_country=league_country,
            match_date=match_date,
            match_time=match_time,
            venue_name=venue_name,
            venue_lat=0.0,
            venue_lon=0.0,
            home_team=home_stats,
            away_team=away_stats,
            weather=WeatherData(),  # preenchido depois
            referee=referee,
            odds=odds,
            injuries_home=injuries_home,
            injuries_away=injuries_away,
            has_real_odds=_has_real_odds,
            has_real_standings=_has_real_standings,
            data_quality_score=round(dq, 2),
            odds_home_away_suspect=_odds_suspect,
            league_avg_goals=round(_league_avg_gpg, 2),
        )
    except Exception as e:
        print(f"    [PARSE] ❌ Erro ao parsear fixture: {e}")
        return None


# ═══════════════════════════════════════════════════════════════
#  ORQUESTRADOR DE INGESTÃO REAL
# ═══════════════════════════════════════════════════════════════

def _check_api_plan() -> dict:
    """Verifica o plano da conta e limites restantes."""
    data = _api_football_request("status", {})
    resp = data.get("response", {})
    plan = resp.get("subscription", {}).get("plan", "Unknown")
    limit = resp.get("requests", {}).get("limit_day", 100)
    current = resp.get("requests", {}).get("current", 0)
    return {"plan": plan, "limit": limit, "used": current, "available": limit - current}


def _ingest_real_data(analysis_dates: list[str] = None) -> list[MatchAnalysis]:
    """
    Pipeline COMPLETO de ingestão — Plano PRO.
    7.500 req/dia | 300 req/min | ALL endpoints | ALL seasons.

      1. Status da conta
      2. Fixtures globais (datas solicitadas)
      3. Standings REAIS (season atual) para TODAS as ligas
      4. Odds REAIS (Pinnacle/Bet365) para TODOS os fixtures
      5. Lesões REAIS para todos os fixtures
      6. Clima real (OpenWeatherMap)
    """
    global _api_call_count
    _api_call_count = 0

    if analysis_dates is None:
        analysis_dates = config.ANALYSIS_DATES

    # ── PASSO 0: Verificar plano ──
    print("[ETL] ═══ PASSO 0: Verificando status da conta ═══")
    plan_info = _check_api_plan()
    plan_name = plan_info["plan"]
    api_available = plan_info["available"]
    is_free = plan_name.lower() == "free"
    print(f"  Plano: {plan_name} | Limite: {plan_info['limit']}/dia | Usadas: {plan_info['used']} | Disponíveis: {api_available}")

    API_BUDGET = min(api_available - 50, 2000)  # Usar até 2000 por run, guardar reserva

    if API_BUDGET < 10:
        print("[ETL] ❌ Orçamento insuficiente para buscar dados reais!")
        print("[ETL] ⚠️  O sistema NÃO processará dados sintéticos/exemplos.")
        print("[ETL] ⚠️  Aguarde o reset diário da API ou aumente o plano.")
        return []  # Retornar vazio em vez de dados sintéticos

    all_fixtures_raw = []

    # ── PASSO 1: Fixtures globais ──
    print(f"[ETL] ═══ PASSO 1: Buscando fixtures globais ({len(analysis_dates)} datas) ═══")
    for date in analysis_dates:
        fixes = _fetch_fixtures(date)
        all_fixtures_raw.extend(fixes)

    if not all_fixtures_raw:
        print("[ETL] ⚠️  Nenhum fixture encontrado!")
        return []

    print(f"[ETL] Total: {len(all_fixtures_raw)} fixtures | API calls: {_api_call_count}")

    # ── PASSO 2: Standings para TODAS as ligas (season atual) ──
    print("[ETL] ═══ PASSO 2: Buscando classificações (ALL ligas) ═══")

    league_fixture_count = {}
    league_info = {}
    for f in all_fixtures_raw:
        lid = f.get("league", {}).get("id", 0)
        season = f.get("league", {}).get("season", 2025)
        lname = f.get("league", {}).get("name", "?")
        lcountry = f.get("league", {}).get("country", "?")
        if lid:
            league_fixture_count[lid] = league_fixture_count.get(lid, 0) + 1
            league_info[lid] = (lname, lcountry, season)

    sorted_leagues = sorted(league_fixture_count.items(), key=lambda x: x[1], reverse=True)
    max_leagues = min(len(sorted_leagues), config.MAX_STANDINGS_LEAGUES)

    print(f"  {len(sorted_leagues)} ligas únicas | Buscando top {max_leagues}")

    standings_cache = {}
    for lid, count in sorted_leagues[:max_leagues]:
        if _api_call_count >= API_BUDGET:
            print(f"  ⚠️  Budget atingido ({_api_call_count}/{API_BUDGET})")
            break
        lname, lcountry, season = league_info.get(lid, ("?", "?", 2025))
        standings = _fetch_standings(lid, season)
        if standings:
            standings_cache[lid] = standings
            print(f"    ✅ {lname} ({lcountry}): {len(standings)} times [season {season}] | {count} fixtures")

    print(f"  Standings: {len(standings_cache)} ligas | API calls: {_api_call_count}")

    # ── PASSO 3: Odds REAIS para todos os fixtures ──
    print("[ETL] ═══ PASSO 3: Buscando ODDS REAIS de mercado ═══")
    odds_cache = {}
    fixture_ids = [f.get("fixture", {}).get("id", 0) for f in all_fixtures_raw if f.get("fixture", {}).get("id")]
    max_odds = min(len(fixture_ids), config.MAX_ODDS_FIXTURES, API_BUDGET - _api_call_count - 50)

    print(f"  Buscando odds para {max_odds} de {len(fixture_ids)} fixtures...")
    odds_found = 0
    for i, fid in enumerate(fixture_ids[:max_odds]):
        if _api_call_count >= API_BUDGET:
            break
        oraw = _fetch_odds_for_fixture(fid)
        if oraw:
            odds_cache[fid] = oraw
            odds_found += 1
        if (i + 1) % 50 == 0:
            print(f"    Progresso: {i+1}/{max_odds} | Odds encontradas: {odds_found}")

    print(f"  Odds obtidas: {odds_found} fixtures | API calls: {_api_call_count}")

    # ── PASSO 4: Lesões REAIS ──
    print("[ETL] ═══ PASSO 4: Buscando LESÕES REAIS ═══")
    injuries_cache = {}
    max_injuries = min(len(fixture_ids), config.MAX_INJURIES_FIXTURES, API_BUDGET - _api_call_count - 10)

    print(f"  Buscando lesões para {max_injuries} fixtures...")
    injuries_found = 0
    for i, fid in enumerate(fixture_ids[:max_injuries]):
        if _api_call_count >= API_BUDGET:
            break
        inj_raw = _fetch_injuries_for_fixture(fid)
        if inj_raw:
            injuries_cache[fid] = inj_raw
            injuries_found += 1

    print(f"  Lesões: {injuries_found} fixtures com dados | API calls: {_api_call_count}")

    # ── PASSO 5: Converter em MatchAnalysis ──
    print("[ETL] ═══ PASSO 5: Convertendo dados em MatchAnalysis ═══")
    all_matches = []
    for fix_raw in all_fixtures_raw:
        match = _parse_fixture_to_match(fix_raw, standings_cache, odds_cache, injuries_cache)
        if match:
            all_matches.append(match)

    print(f"  {len(all_matches)} partidas convertidas")

    # ── PASSO 6: Clima real (OpenWeatherMap — API separada) ──
    if config.OPENWEATHER_KEY:
        print("[ETL] ═══ PASSO 6: Buscando clima real (OpenWeatherMap) ═══")
        weather_cache = {}
        calls = 0
        max_weather = min(len(all_matches), 100)
        for match in all_matches:
            if calls >= max_weather:
                break
            city = match.venue_name
            if not city or city == "N/D":
                continue
            country = match.league_country
            cc = COUNTRY_CODES.get(country, "")
            ck = f"{city}_{cc}"
            if ck not in weather_cache:
                w = _fetch_weather_by_city(city, cc)
                weather_cache[ck] = w
                calls += 1
                time.sleep(0.1)
            match.weather = weather_cache[ck]
            if weather_cache[ck].description != "N/D":
                match.has_real_weather = True
                match.data_quality_score = min(1.0, match.data_quality_score + 0.10)
        print(f"  Clima: {len(weather_cache)} locais")
    else:
        print("[ETL] ⚠️  OpenWeather não configurada")

    # ── RESUMO FINAL ──
    n_leagues = len(set(m.league_name for m in all_matches))
    n_standings = sum(1 for m in all_matches if m.home_team.league_position > 0)
    n_real_odds = sum(1 for m in all_matches if m.odds.bookmaker not in ("N/D", "Modelo (Estimado)"))
    n_injuries = sum(1 for m in all_matches if m.injuries_home or m.injuries_away)
    cache_stats = get_api_cache_stats()

    print(f"\n[ETL] =======================================")
    print(f"[ETL]   INGESTAO CONCLUIDA - PLANO {plan_name.upper()}")
    print(f"[ETL] =======================================")
    print(f"[ETL] API-Football calls REAIS: {_api_call_count} de {api_available}")
    print(f"[ETL] Cache HITS (economia): {_api_cache_hits} requests salvos!")
    print(f"[ETL] Cache MISSES (novos):  {_api_cache_misses}")
    print(f"[ETL] Cache local: {cache_stats['local_files']} arquivos | {cache_stats['local_size_mb']}MB")
    print(f"[ETL] Supabase: {cache_stats['supabase_total']} respostas armazenadas na nuvem")
    print(f"[ETL] Partidas: {len(all_matches)} | Ligas: {n_leagues}")
    print(f"[ETL] Com standings: {n_standings}")
    print(f"[ETL] Com odds REAIS: {n_real_odds}")
    print(f"[ETL] Com lesoes: {n_injuries}")
    print(f"[ETL] =======================================")

    return all_matches


def get_cached_player_shots(team_id: int, n_last: int = 10) -> list[dict]:
    """
    Tenta obter dados de finalizações de jogadores do CACHE (sem API calls).
    Retorna lista de top jogadores com avg_shots, avg_sot, e linhas O/U.
    Retorna [] se não houver dados em cache.
    """
    # Tentar buscar últimos fixtures do time (cache only)
    fixtures_data = _api_football_request(
        "fixtures", {"team": team_id, "last": n_last, "status": "FT"},
        cache_only=True
    )
    fixtures = fixtures_data.get("response", [])
    if not fixtures:
        return []

    # Coletar dados de jogadores de cada fixture (cache only)
    player_map = {}
    games_found = 0
    for fix in fixtures[:n_last]:
        fix_id = fix.get("fixture", {}).get("id", 0)
        if not fix_id:
            continue

        players_data = _api_football_request(
            "fixtures/players", {"fixture": fix_id},
            cache_only=True
        )
        players_response = players_data.get("response", [])
        if not players_response:
            continue

        games_found += 1
        for team_block in players_response:
            tid = team_block.get("team", {}).get("id", 0)
            if tid != team_id:
                continue
            for p in team_block.get("players", []):
                pi = p.get("player", {})
                for ps in p.get("statistics", []):
                    shots_info = ps.get("shots", {}) or {}
                    games_info = ps.get("games", {}) or {}
                    minutes = games_info.get("minutes", 0) or 0
                    if minutes < 1:
                        continue
                    total_shots = shots_info.get("total") or 0
                    shots_on = shots_info.get("on") or 0
                    key = pi.get("name", "?")
                    if key not in player_map:
                        player_map[key] = {
                            "id": pi.get("id"), "name": key,
                            "position": games_info.get("position", "?"),
                            "matches": 0, "total_shots": 0, "total_sot": 0,
                            "_shots_h": [], "_sot_h": [],
                        }
                    pl = player_map[key]
                    pl["matches"] += 1
                    pl["total_shots"] += total_shots
                    pl["total_sot"] += shots_on
                    pl["_shots_h"].append(total_shots)
                    pl["_sot_h"].append(shots_on)

    if games_found < 2:
        return []

    # Calcular rankings e fair odds
    rankings = sorted(
        [p for p in player_map.values() if p["matches"] >= 2],
        key=lambda p: p["total_shots"] / max(1, p["matches"]),
        reverse=True
    )[:10]

    for p in rankings:
        nm = p["matches"]
        p["avg_shots"] = round(p["total_shots"] / nm, 1)
        p["avg_sot"] = round(p["total_sot"] / nm, 1)
        p["shots_lines"] = {}
        for line in [0.5, 1.5, 2.5, 3.5]:
            oc = sum(1 for s in p["_shots_h"] if s > line)
            pct = round(oc / nm * 100, 0)
            fair = round(nm / max(1, oc), 2) if oc > 0 else 99.0
            p["shots_lines"][f"over_{line}"] = {"pct": pct, "fair_odd": fair, "sample": nm}
        p["sot_lines"] = {}
        for line in [0.5, 1.5, 2.5]:
            oc = sum(1 for s in p["_sot_h"] if s > line)
            pct = round(oc / nm * 100, 0)
            fair = round(nm / max(1, oc), 2) if oc > 0 else 99.0
            p["sot_lines"][f"over_{line}"] = {"pct": pct, "fair_odd": fair, "sample": nm}
        # Limpar dados temporários
        del p["_shots_h"]
        del p["_sot_h"]

    return rankings


# ═══════════════════════════════════════════════════════════════
#  HISTÓRICO DE TIMES — Dados detalhados para análise comparativa
# ═══════════════════════════════════════════════════════════════

def _compute_ev_analysis(all_matches: list, league_matches: list) -> dict:
    """Computa analise EV+ completa: gols, escanteios, cartoes, finalizacoes, jogadores.
    Retorna dict com fair odds para todas as linhas de cada mercado."""

    def _ou_lines(values: list, lines: list) -> dict:
        """Calcula Over/Under com probabilidade historica e fair odd."""
        if not values:
            return {}
        n = len(values)
        avg = round(sum(values) / n, 2)
        result = {"avg": avg, "sample": n}
        for line in lines:
            oc = sum(1 for v in values if v > line)
            pct = round(oc / n * 100, 1)
            fair = round(n / oc, 2) if oc > 0 else 99.99
            result[f"o{line}"] = {"count": oc, "total": n, "pct": pct, "fair_odd": fair}
        return result

    def _analyze_set(matches: list) -> dict:
        """Analisa um conjunto de partidas (liga ou todas)."""
        if not matches:
            return {}
        n = len(matches)

        # ── Gols ──
        team_goals = []
        opp_goals = []
        total_goals = []
        for m in matches:
            mg = m.get("score_home", 0) if m.get("is_home") else m.get("score_away", 0)
            og = m.get("score_away", 0) if m.get("is_home") else m.get("score_home", 0)
            team_goals.append(mg)
            opp_goals.append(og)
            total_goals.append(m.get("total_goals", 0))
        btts_n = sum(1 for t, o in zip(team_goals, opp_goals) if t > 0 and o > 0)
        cs_n = sum(1 for o in opp_goals if o == 0)
        fts_n = sum(1 for t in team_goals if t == 0)

        goals = {
            "total": _ou_lines(total_goals, [0.5, 1.5, 2.5, 3.5, 4.5]),
            "team": _ou_lines(team_goals, [0.5, 1.5, 2.5, 3.5]),
            "opp": _ou_lines(opp_goals, [0.5, 1.5, 2.5]),
            "btts_pct": round(btts_n / n * 100, 1),
            "btts_fair": round(n / btts_n, 2) if btts_n > 0 else 99.99,
            "cs_pct": round(cs_n / n * 100, 1),
            "fts_pct": round(fts_n / n * 100, 1),
        }

        # ── Finalizacoes ──
        t_shots = [int(m["stats"]["team_shots"]) for m in matches
                   if m.get("stats", {}).get("team_shots") is not None]
        t_sot = [int(m["stats"]["team_shots_on_target"]) for m in matches
                 if m.get("stats", {}).get("team_shots_on_target") is not None]
        o_shots = [int(m["stats"]["opp_shots"]) for m in matches
                   if m.get("stats", {}).get("opp_shots") is not None]
        o_sot = [int(m["stats"]["opp_shots_on_target"]) for m in matches
                 if m.get("stats", {}).get("opp_shots_on_target") is not None]

        shots = {
            "team_shots": _ou_lines(t_shots, [7.5, 8.5, 9.5, 10.5, 11.5, 12.5]),
            "team_sot": _ou_lines(t_sot, [1.5, 2.5, 3.5, 4.5, 5.5]),
            "opp_shots": _ou_lines(o_shots, [7.5, 8.5, 9.5, 10.5, 11.5]),
            "opp_sot": _ou_lines(o_sot, [1.5, 2.5, 3.5, 4.5, 5.5]),
        }

        # ── Escanteios ──
        c_total = [int(m["stats"]["total_corners"]) for m in matches
                   if m.get("stats", {}).get("total_corners") is not None]
        c_team = [int(m["stats"]["team_corners"]) for m in matches
                  if m.get("stats", {}).get("team_corners") is not None]
        c_opp = [int(m["stats"]["opp_corners"]) for m in matches
                 if m.get("stats", {}).get("opp_corners") is not None]

        corners = {
            "total": _ou_lines(c_total, [7.5, 8.5, 9.5, 10.5, 11.5]),
            "team": _ou_lines(c_team, [3.5, 4.5, 5.5, 6.5]),
            "opp": _ou_lines(c_opp, [3.5, 4.5, 5.5, 6.5]),
        }

        # ── Cartoes ──
        k_total = [int(m["stats"]["total_cards"]) for m in matches
                   if m.get("stats", {}).get("total_cards") is not None]
        cards = {
            "total": _ou_lines(k_total, [2.5, 3.5, 4.5, 5.5, 6.5]),
        }

        # ── HT (1o Tempo) ──
        ht_total = [m["ht_total"] for m in matches if m.get("ht_total") is not None]
        ht_team = [(m["ht_home"] if m.get("is_home") else m["ht_away"])
                    for m in matches if m.get("ht_home") is not None]
        ht_opp = [(m["ht_away"] if m.get("is_home") else m["ht_home"])
                   for m in matches if m.get("ht_away") is not None]

        ht = {
            "total": _ou_lines(ht_total, [0.5, 1.5, 2.5]),
            "team": _ou_lines(ht_team, [0.5, 1.5]),
            "opp": _ou_lines(ht_opp, [0.5, 1.5]),
        }
        # HT Result
        ht_home_wins = sum(1 for m in matches if m.get("ht_home") is not None
                          and ((m["is_home"] and m["ht_home"] > m["ht_away"]) or
                               (not m["is_home"] and m["ht_away"] > m["ht_home"])))
        ht_draws = sum(1 for m in matches if m.get("ht_home") is not None
                       and m["ht_home"] == m["ht_away"])
        ht_with_data = sum(1 for m in matches if m.get("ht_home") is not None)
        if ht_with_data > 0:
            ht["ht_win_pct"] = round(ht_home_wins / ht_with_data * 100, 1)
            ht["ht_draw_pct"] = round(ht_draws / ht_with_data * 100, 1)
            ht["ht_loss_pct"] = round((ht_with_data - ht_home_wins - ht_draws) / ht_with_data * 100, 1)
            ht["ht_sample"] = ht_with_data

        # ── Clean Sheet & Win to Nil ──
        cs_pct = round(cs_n / n * 100, 1) if n else 0
        cs_fair = round(n / cs_n, 2) if cs_n > 0 else 99.99
        fts_fair = round(n / fts_n, 2) if fts_n > 0 else 99.99
        wtn_n = sum(1 for t, o in zip(team_goals, opp_goals) if t > 0 and o == 0)
        wtn_pct = round(wtn_n / n * 100, 1) if n else 0
        wtn_fair = round(n / wtn_n, 2) if wtn_n > 0 else 99.99

        specials = {
            "cs_pct": cs_pct, "cs_fair": cs_fair, "cs_count": cs_n,
            "fts_pct": round(fts_n / n * 100, 1) if n else 0, "fts_fair": fts_fair,
            "wtn_pct": wtn_pct, "wtn_fair": wtn_fair, "wtn_count": wtn_n,
            "btts_pct": goals["btts_pct"], "btts_fair": goals["btts_fair"],
        }

        # ── Odd/Even ──
        odd_n = sum(1 for tg in total_goals if tg % 2 == 1)
        even_n = n - odd_n
        specials["odd_pct"] = round(odd_n / n * 100, 1) if n else 0
        specials["even_pct"] = round(even_n / n * 100, 1) if n else 0
        specials["odd_fair"] = round(n / odd_n, 2) if odd_n > 0 else 99.99
        specials["even_fair"] = round(n / even_n, 2) if even_n > 0 else 99.99

        # ── Posse, xG, Passes ──
        poss_vals = [float(m["stats"]["possession"]) for m in matches
                     if m.get("stats", {}).get("possession") is not None]
        xg_vals = [float(m["stats"]["expected_goals"]) for m in matches
                   if m.get("stats", {}).get("expected_goals") is not None]
        pass_pct = [float(m["stats"]["passes_pct"]) for m in matches
                    if m.get("stats", {}).get("passes_pct") is not None]
        offsides = [int(m["stats"]["offsides"]) for m in matches
                    if m.get("stats", {}).get("offsides") is not None]
        fouls_team = [int(m["stats"]["fouls"]) for m in matches
                      if m.get("stats", {}).get("fouls") is not None]
        gk_saves = [int(m["stats"]["gk_saves"]) for m in matches
                    if m.get("stats", {}).get("gk_saves") is not None]

        advanced = {
            "possession_avg": round(sum(poss_vals) / len(poss_vals), 1) if poss_vals else None,
            "xg_avg": round(sum(xg_vals) / len(xg_vals), 2) if xg_vals else None,
            "pass_pct_avg": round(sum(pass_pct) / len(pass_pct), 1) if pass_pct else None,
            "offsides_avg": round(sum(offsides) / len(offsides), 1) if offsides else None,
            "fouls_avg": round(sum(fouls_team) / len(fouls_team), 1) if fouls_team else None,
            "gk_saves_avg": round(sum(gk_saves) / len(gk_saves), 1) if gk_saves else None,
        }

        return {
            "sample": n, "goals": goals, "shots": shots,
            "corners": corners, "cards": cards, "ht": ht,
            "specials": specials, "advanced": advanced
        }

    # ── Analise de Jogadores ──
    player_map = {}
    for m in all_matches:
        if not m.get("players"):
            continue
        for p in m["players"]:
            if not p.get("name") or not p.get("minutes") or p["minutes"] < 1:
                continue
            key = p.get("id") or p["name"]
            if key not in player_map:
                player_map[key] = {
                    "id": p.get("id"), "name": p["name"], "position": p.get("position", "?"),
                    "matches": 0, "total_shots": 0, "total_sot": 0,
                    "total_goals": 0, "total_assists": 0, "total_minutes": 0,
                    "_shots_h": [], "_sot_h": [],
                }
            pl = player_map[key]
            pl["matches"] += 1
            pl["total_shots"] += p.get("total_shots", 0)
            pl["total_sot"] += p.get("shots_on_target", 0)
            pl["total_goals"] += p.get("goals", 0)
            pl["total_assists"] += p.get("assists", 0)
            pl["total_minutes"] += p["minutes"]
            pl["_shots_h"].append(p.get("total_shots", 0))
            pl["_sot_h"].append(p.get("shots_on_target", 0))

    player_rankings = sorted(
        [p for p in player_map.values() if p["matches"] >= 2],
        key=lambda p: p["total_shots"] / max(1, p["matches"]),
        reverse=True
    )[:15]

    for p in player_rankings:
        nm = p["matches"]
        p["avg_shots"] = round(p["total_shots"] / nm, 1)
        p["avg_sot"] = round(p["total_sot"] / nm, 1)
        # O/U Finalizacoes
        p["shots_lines"] = {}
        for line in [0.5, 1.5, 2.5, 3.5, 4.5]:
            oc = sum(1 for s in p["_shots_h"] if s > line)
            pct = round(oc / nm * 100, 1)
            fair = round(nm / oc, 2) if oc > 0 else 99.99
            p["shots_lines"][f"o{line}"] = {"count": oc, "total": nm, "pct": pct, "fair_odd": fair}
        # O/U Finalizacoes em Gol
        p["sot_lines"] = {}
        for line in [0.5, 1.5, 2.5, 3.5]:
            oc = sum(1 for s in p["_sot_h"] if s > line)
            pct = round(oc / nm * 100, 1)
            fair = round(nm / oc, 2) if oc > 0 else 99.99
            p["sot_lines"][f"o{line}"] = {"count": oc, "total": nm, "pct": pct, "fair_odd": fair}
        # Remover dados brutos
        del p["_shots_h"]
        del p["_sot_h"]

    # ── Montar top oportunidades (fair odd + probabilidade) ──
    top_opps = []

    # Oportunidades de time (usar all_matches analysis)
    all_analysis = _analyze_set(all_matches)
    if all_analysis:
        # Top shots lines
        for market_key, market_label in [
            ("team_shots", "Finalizacoes Time"), ("team_sot", "Fin. em Gol Time"),
            ("opp_shots", "Finalizacoes Adversario"), ("opp_sot", "Fin. em Gol Advers."),
        ]:
            data = all_analysis.get("shots", {}).get(market_key, {})
            for k, v in data.items():
                if isinstance(v, dict) and v.get("pct") is not None and v["pct"] >= 55:
                    top_opps.append({
                        "market": market_label, "line": k.replace("o", "Over "),
                        "pct": v["pct"], "fair_odd": v["fair_odd"], "sample": v["total"],
                    })
        # Top corners lines
        for market_key, market_label in [("total", "Escanteios Total"), ("team", "Escanteios Time")]:
            data = all_analysis.get("corners", {}).get(market_key, {})
            for k, v in data.items():
                if isinstance(v, dict) and v.get("pct") is not None and v["pct"] >= 55:
                    top_opps.append({
                        "market": f"{market_label}", "line": k.replace("o", "Over "),
                        "pct": v["pct"], "fair_odd": v["fair_odd"], "sample": v["total"],
                    })
        # Top cards lines
        data = all_analysis.get("cards", {}).get("total", {})
        for k, v in data.items():
            if isinstance(v, dict) and v.get("pct") is not None and v["pct"] >= 55:
                top_opps.append({
                    "market": "Cartoes Total", "line": k.replace("o", "Over "),
                    "pct": v["pct"], "fair_odd": v["fair_odd"], "sample": v["total"],
                })
        # Top goals lines
        for market_key, market_label in [("total", "Gols Total"), ("team", "Gols Time")]:
            data = all_analysis.get("goals", {}).get(market_key, {})
            for k, v in data.items():
                if isinstance(v, dict) and v.get("pct") is not None and v["pct"] >= 55:
                    top_opps.append({
                        "market": f"{market_label}", "line": k.replace("o", "Over "),
                        "pct": v["pct"], "fair_odd": v["fair_odd"], "sample": v["total"],
                    })

    # Oportunidades de jogadores
    for p in player_rankings:
        for line_key, v in p.get("shots_lines", {}).items():
            if v.get("pct", 0) >= 60:
                top_opps.append({
                    "market": f"Fin. {p['name']}", "line": line_key.replace("o", "Over "),
                    "pct": v["pct"], "fair_odd": v["fair_odd"], "sample": v["total"],
                })
        for line_key, v in p.get("sot_lines", {}).items():
            if v.get("pct", 0) >= 60:
                top_opps.append({
                    "market": f"Fin. Gol {p['name']}", "line": line_key.replace("o", "Over "),
                    "pct": v["pct"], "fair_odd": v["fair_odd"], "sample": v["total"],
                })

    top_opps.sort(key=lambda x: x["pct"], reverse=True)

    return {
        "all_analysis": all_analysis,
        "league_analysis": _analyze_set(league_matches),
        "player_rankings": player_rankings,
        "top_opportunities": top_opps[:30],
    }


def fetch_team_history(team_id: int, league_id: int = None, last: int = 10) -> dict:
    """
    Busca historico completo de um time com analise EV+ pre-computada.
    Resultado inteiro e cacheado no Supabase (nao precisa rodar 2x).

    Retorna dict com:
      {
        "team_id": int,
        "all_matches": [...],
        "league_matches": [...],
        "ev_analysis": { analise EV+ completa com fair odds },
      }
    """
    # ── Verificar cache do resultado completo ──
    cache_params = {"team": team_id, "league": league_id or 0, "last": last}
    cached_result = _get_cached_response("team_history", cache_params)
    if cached_result is not None:
        print(f"  [CACHE] team_history({team_id}) -> HIT (resultado completo com analise EV+)")
        return cached_result

    result = {"team_id": team_id, "all_matches": [], "league_matches": []}

    # ── 1. Buscar últimos jogos (todos os campeonatos) ──
    all_data = _api_football_request("fixtures", {
        "team": team_id, "last": last, "status": "FT-AET-PEN"
    })
    all_fixtures = all_data.get("response", [])

    # ── 2. Buscar últimos jogos no campeonato específico ──
    league_fixtures = []
    if league_id:
        lg_data = _api_football_request("fixtures", {
            "team": team_id, "league": league_id, "last": last, "status": "FT-AET-PEN"
        })
        league_fixtures = lg_data.get("response", [])

    # ── 3. Processar cada fixture ──
    def _process_fixture(fix_raw: dict, team_id: int) -> dict:
        """Extrai dados relevantes de um fixture passado."""
        fixture = fix_raw.get("fixture", {})
        league = fix_raw.get("league", {})
        teams = fix_raw.get("teams", {})
        goals = fix_raw.get("goals", {})
        score = fix_raw.get("score", {})

        fix_id = fixture.get("id", 0)
        date_str = fixture.get("date", "")
        try:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            dt_br = dt.astimezone(config.BR_TIMEZONE)
            match_date = dt_br.strftime("%Y-%m-%d")
            match_time = dt_br.strftime("%H:%M")
        except (ValueError, TypeError):
            match_date = "N/D"
            match_time = "N/D"

        home_info = teams.get("home", {})
        away_info = teams.get("away", {})
        home_id = home_info.get("id", 0)
        away_id = away_info.get("id", 0)

        is_home = (home_id == team_id)
        opponent_id = away_id if is_home else home_id
        opponent_name = away_info.get("name", "?") if is_home else home_info.get("name", "?")

        score_home = goals.get("home", 0) or 0
        score_away = goals.get("away", 0) or 0

        # HT scores
        ht = score.get("halftime", {}) or {}
        ht_home = ht.get("home")
        ht_away = ht.get("away")

        # Resultado do time
        my_goals = score_home if is_home else score_away
        opp_goals = score_away if is_home else score_home
        if my_goals > opp_goals:
            result_letter = "W"
        elif my_goals < opp_goals:
            result_letter = "L"
        else:
            result_letter = "D"

        return {
            "fixture_id": fix_id,
            "date": match_date,
            "time": match_time,
            "league_name": league.get("name", "?"),
            "league_country": league.get("country", "?"),
            "league_id": league.get("id", 0),
            "home_team": home_info.get("name", "?"),
            "away_team": away_info.get("name", "?"),
            "home_id": home_id,
            "away_id": away_id,
            "score_home": score_home,
            "score_away": score_away,
            "result": result_letter,
            "is_home": is_home,
            "opponent": opponent_name,
            "opponent_id": opponent_id,
            "total_goals": score_home + score_away,
            "ht_home": ht_home,
            "ht_away": ht_away,
            "ht_total": (ht_home or 0) + (ht_away or 0) if ht_home is not None else None,
            # Odds e lineup serão preenchidos abaixo
            "odds_home": None,
            "odds_draw": None,
            "odds_away": None,
            "was_favorite": None,
            "lineup_type": "N/D",
            "lineup_count": None,
            "stats": {},
            "players": [],
        }

    # Processar todos os fixtures
    all_processed = [_process_fixture(f, team_id) for f in all_fixtures]
    league_processed = [_process_fixture(f, team_id) for f in league_fixtures]

    # ── 4. Buscar dados extras para cada fixture (odds, lineups, stats) ──
    all_fix_ids = set()
    for m in all_processed + league_processed:
        all_fix_ids.add(m["fixture_id"])

    for fix_id in all_fix_ids:
        # Buscar estatísticas do jogo
        stats_data = _api_football_request("fixtures/statistics", {"fixture": fix_id})
        stats_response = stats_data.get("response", [])

        # Buscar lineups do jogo
        lineups_data = _api_football_request("fixtures/lineups", {"fixture": fix_id})
        lineups_response = lineups_data.get("response", [])

        # Buscar odds do jogo (pré-jogo)
        odds_data = _api_football_request("odds", {"fixture": fix_id})
        odds_response = odds_data.get("response", [])

        # Buscar estatísticas de jogadores (finalizações individuais)
        players_data = _api_football_request("fixtures/players", {"fixture": fix_id})
        players_response = players_data.get("response", [])

        # Processar e atribuir aos matches
        for match_list in [all_processed, league_processed]:
            for m in match_list:
                if m["fixture_id"] != fix_id:
                    continue

                # ── Stats ──
                for team_stats in stats_response:
                    sid = team_stats.get("team", {}).get("id", 0)
                    is_my_team = (sid == team_id)
                    prefix = "team" if is_my_team else "opp"

                    stat_list = team_stats.get("statistics", [])
                    stat_dict = {}
                    for s in stat_list:
                        stype = s.get("type", "")
                        sval = s.get("value")
                        stat_dict[stype] = sval

                    m["stats"][f"{prefix}_shots"] = stat_dict.get("Total Shots")
                    m["stats"][f"{prefix}_shots_on_target"] = stat_dict.get("Shots on Goal")
                    m["stats"][f"{prefix}_corners"] = stat_dict.get("Corner Kicks")
                    m["stats"][f"{prefix}_fouls"] = stat_dict.get("Fouls")
                    m["stats"][f"{prefix}_cards_yellow"] = stat_dict.get("Yellow Cards")
                    m["stats"][f"{prefix}_cards_red"] = stat_dict.get("Red Cards")
                    m["stats"][f"{prefix}_possession"] = stat_dict.get("Ball Possession")
                    m["stats"][f"{prefix}_offsides"] = stat_dict.get("Offsides")
                    m["stats"][f"{prefix}_saves"] = stat_dict.get("Goalkeeper Saves")
                    m["stats"][f"{prefix}_passes"] = stat_dict.get("Total passes")
                    m["stats"][f"{prefix}_passes_pct"] = stat_dict.get("Passes %")
                    m["stats"][f"{prefix}_expected_goals"] = stat_dict.get("expected_goals")

                # Total de cartões no jogo
                ty = _safe_int(m["stats"].get("team_cards_yellow"))
                tr = _safe_int(m["stats"].get("team_cards_red"))
                oy = _safe_int(m["stats"].get("opp_cards_yellow"))
                or_ = _safe_int(m["stats"].get("opp_cards_red"))
                if ty is not None and oy is not None:
                    m["stats"]["total_cards"] = (ty or 0) + (tr or 0) + (oy or 0) + (or_ or 0)
                # Total de escanteios
                tc = _safe_int(m["stats"].get("team_corners"))
                oc = _safe_int(m["stats"].get("opp_corners"))
                if tc is not None and oc is not None:
                    m["stats"]["total_corners"] = (tc or 0) + (oc or 0)

                # ── Lineups ──
                for lineup in lineups_response:
                    lid = lineup.get("team", {}).get("id", 0)
                    if lid == team_id:
                        start_xi = lineup.get("startXI", [])
                        subs = lineup.get("substitutes", [])
                        m["lineup_count"] = len(start_xi)
                        # Classificar lineup (heurística simples)
                        # Se tiver 11 titulares, é titular
                        # Na realidade precisaríamos de dados da temporada
                        # Por ora, marcamos apenas que temos dados
                        m["lineup_type"] = "disponivel"
                        m["stats"]["formation"] = lineup.get("formation", "N/D")
                        m["stats"]["coach"] = lineup.get("coach", {}).get("name", "N/D")
                        break

                # ── Odds ──
                if odds_response:
                    bookmakers = odds_response[0].get("bookmakers", [])
                    for bm in bookmakers:
                        bm_name = bm.get("name", "")
                        if bm_name in config.PREFERRED_BOOKMAKERS:
                            for bet in bm.get("bets", []):
                                if bet.get("name") == "Match Winner":
                                    for v in bet.get("values", []):
                                        if v.get("value") == "Home":
                                            m["odds_home"] = float(v.get("odd", 0))
                                        elif v.get("value") == "Draw":
                                            m["odds_draw"] = float(v.get("odd", 0))
                                        elif v.get("value") == "Away":
                                            m["odds_away"] = float(v.get("odd", 0))
                            break  # Usar primeiro bookmaker preferido encontrado

                    # Determinar se era favorito
                    if m["odds_home"] and m["odds_away"]:
                        if m["is_home"]:
                            m["was_favorite"] = m["odds_home"] < m["odds_away"]
                        else:
                            m["was_favorite"] = m["odds_away"] < m["odds_home"]

                # ── Estatísticas de Jogadores (finalizações individuais) ──
                for team_players in players_response:
                    tid = team_players.get("team", {}).get("id", 0)
                    if tid == team_id:
                        plist = []
                        for p in team_players.get("players", []):
                            pi = p.get("player", {})
                            pstats_list = p.get("statistics", [])
                            if not pstats_list:
                                continue
                            ps = pstats_list[0]
                            shots_info = ps.get("shots", {}) or {}
                            goals_info = ps.get("goals", {}) or {}
                            games_info = ps.get("games", {}) or {}
                            minutes_played = games_info.get("minutes", 0) or 0
                            if minutes_played < 1:
                                continue  # Pular jogadores que não entraram
                            plist.append({
                                "id": pi.get("id"),
                                "name": pi.get("name", "?"),
                                "position": games_info.get("position", "?"),
                                "number": games_info.get("number"),
                                "minutes": minutes_played,
                                "rating": games_info.get("rating"),
                                "substitute": games_info.get("substitute", False),
                                "total_shots": (shots_info.get("total") or 0),
                                "shots_on_target": (shots_info.get("on") or 0),
                                "goals": (goals_info.get("total") or 0),
                                "assists": (goals_info.get("assists") or 0),
                            })
                        m["players"] = plist
                        break

    result["all_matches"] = all_processed
    result["league_matches"] = league_processed

    # ── Computar analise EV+ completa ──
    try:
        result["ev_analysis"] = _compute_ev_analysis(all_processed, league_processed)
        print(f"  [EV+] Analise EV+ computada para team {team_id}: "
              f"{len(result['ev_analysis'].get('top_opportunities', []))} oportunidades, "
              f"{len(result['ev_analysis'].get('player_rankings', []))} jogadores ranqueados")
    except Exception as e:
        print(f"  [EV+] Erro ao computar analise: {e}")
        result["ev_analysis"] = {}

    # ── Salvar resultado completo no cache (local + Supabase) ──
    try:
        _save_to_cache("team_history", cache_params, result)
        print(f"  [CACHE] team_history({team_id}) salvo (local + Supabase)")
    except Exception as e:
        print(f"  [CACHE] Erro ao salvar team_history: {e}")

    return result


def fetch_h2h(team1_id: int, team2_id: int, last: int = 10) -> list[dict]:
    """Busca confrontos diretos (head-to-head) entre dois times."""
    data = _api_football_request("fixtures/headtohead", {
        "h2h": f"{team1_id}-{team2_id}", "last": last, "status": "FT-AET-PEN"
    })
    raw = data.get("response", [])
    matches = []
    for fix_raw in raw:
        fixture = fix_raw.get("fixture", {})
        league = fix_raw.get("league", {})
        teams = fix_raw.get("teams", {})
        goals = fix_raw.get("goals", {})

        fix_id = fixture.get("id", 0)
        date_str = fixture.get("date", "")
        try:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            dt_br = dt.astimezone(config.BR_TIMEZONE)
            match_date = dt_br.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            match_date = "N/D"

        home_info = teams.get("home", {})
        away_info = teams.get("away", {})
        score_home = goals.get("home", 0) or 0
        score_away = goals.get("away", 0) or 0

        matches.append({
            "fixture_id": fix_id,
            "date": match_date,
            "league_name": league.get("name", "?"),
            "home_team": home_info.get("name", "?"),
            "away_team": away_info.get("name", "?"),
            "home_id": home_info.get("id", 0),
            "away_id": away_info.get("id", 0),
            "score_home": score_home,
            "score_away": score_away,
            "total_goals": score_home + score_away,
        })
    return matches


def _safe_int(val) -> int | None:
    """Converte valor para int, retorna None se impossível."""
    if val is None:
        return None
    try:
        if isinstance(val, str):
            val = val.replace("%", "")
        return int(float(val))
    except (ValueError, TypeError):
        return None


def enrich_multi_bookmaker(match_id: int) -> dict:
    """
    Carrega as odds brutas do cache da API para um fixture e retorna
    o dict _bookmakers por mercado.  Não faz chamadas reais à API.
    
    Retorna: {market_key: {bk_name: {sel: odd, ...}, ...}, ...}
    """
    raw = _api_football_request("odds", {"fixture": match_id}, cache_only=True)
    if not raw:
        return {}
    
    response = raw.get("response", [])
    if not response:
        return {}
    
    odds_raw = response[0] if isinstance(response, list) else response
    bookmakers = odds_raw.get("bookmakers", [])
    if not bookmakers:
        return {}
    
    result = {}  # {market_key: {bk_name: {sel: odd}}}
    
    for bk in bookmakers:
        current_bk = bk.get("name", "Desconhecido")
        for bet in bk.get("bets", []):
            bet_id = bet.get("id", 0)
            bet_name_raw = bet.get("name", "")
            mk = _bet_to_market_key(bet_id, bet_name_raw)
            if not mk:
                continue
            
            values = bet.get("values", [])
            val_map = {}
            for v in values:
                val_map[str(v.get("value", "")).lower().replace(" ", "_")] = float(v.get("odd", 0))
            if not val_map:
                continue
            
            if mk not in result:
                result[mk] = {}
            result[mk][current_bk] = val_map
    
    return result


def fetch_finished_fixtures(match_ids: list[int]) -> dict:
    """
    Busca resultado FINAL de jogos pelo fixture_id.
    Retorna dict {match_id: {"status": "FT"|..., "home_goals": int, "away_goals": int, "score": "2-1"}}.
    Usa cache — se já tem resultado FT em cache, não busca de novo.
    """
    results = {}
    if not match_ids:
        return results

    for mid in match_ids:
        # Tentar cache primeiro (fixtures retorna dados completos)
        cached = _get_cached_response("fixtures", {"id": mid})
        raw = cached or _api_football_request("fixtures", {"id": mid})
        if not raw:
            continue
        response = raw.get("response", [])
        if not response:
            continue
        fix = response[0]
        fix_data = fix.get("fixture", {})
        goals = fix.get("goals", {})
        status_short = fix_data.get("status", {}).get("short", "?")
        score = fix.get("score", {})

        home_goals = goals.get("home")
        away_goals = goals.get("away")

        finished = status_short in ("FT", "AET", "PEN")

        if finished and home_goals is not None and away_goals is not None:
            results[mid] = {
                "status": status_short,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "score": f"{home_goals}-{away_goals}",
                "ht_home": (score.get("halftime", {}) or {}).get("home"),
                "ht_away": (score.get("halftime", {}) or {}).get("away"),
            }
            # Salvar no cache para não buscar de novo
            if not cached:
                _save_to_cache("fixtures", {"id": mid}, raw)
        elif not finished:
            results[mid] = {"status": status_short, "home_goals": None, "away_goals": None, "score": None}

    return results


## _generate_estimated_odds REMOVIDA — sistema usa apenas odds reais de casas de apostas


# ═══════════════════════════════════════════════════════════════
#  GERADOR SINTÉTICO (FALLBACK) - mantido para modo demo
# ═══════════════════════════════════════════════════════════════

LEAGUES_DB = [
    {"id": 39, "name": "Premier League", "country": "Inglaterra",
     "teams": [
         ("Arsenal", 1.85, 0.85), ("Manchester City", 2.05, 0.75),
         ("Liverpool", 1.95, 0.80), ("Chelsea", 1.55, 1.05),
         ("Manchester United", 1.50, 1.10), ("Tottenham", 1.65, 1.00),
         ("Newcastle", 1.60, 0.90), ("Aston Villa", 1.50, 1.00),
         ("Brighton", 1.45, 0.95), ("West Ham", 1.35, 1.10),
     ], "avg_goals": 2.85, "venue_lat": 51.5, "venue_lon": -0.1},
    {"id": 140, "name": "La Liga", "country": "Espanha",
     "teams": [
         ("Real Madrid", 2.10, 0.70), ("Barcelona", 2.15, 0.75),
         ("Atletico Madrid", 1.55, 0.80), ("Athletic Bilbao", 1.40, 0.90),
         ("Real Sociedad", 1.45, 0.95), ("Villarreal", 1.50, 1.00),
         ("Betis", 1.35, 1.05), ("Girona", 1.40, 1.10),
         ("Sevilla", 1.30, 1.10), ("Mallorca", 1.15, 1.05),
     ], "avg_goals": 2.55, "venue_lat": 40.4, "venue_lon": -3.7},
]


def _generate_form(attack: float, defense: float) -> list[str]:
    strength = attack - defense
    p_win = min(0.65, max(0.15, 0.35 + strength * 0.15))
    p_draw = 0.25
    form = []
    for _ in range(10):
        r = random.random()
        if r < p_win:
            form.append("W")
        elif r < p_win + p_draw:
            form.append("D")
        else:
            form.append("L")
    return form


def generate_synthetic_fixtures(date: str) -> list[MatchAnalysis]:
    """Fallback: gera fixtures sintéticas."""
    from math import exp, factorial
    random.seed(hash(date) % 2**32)
    np.random.seed(hash(date) % 2**32)
    matches = []
    mid = 1000
    for league in LEAGUES_DB:
        teams = league["teams"]
        n = len(teams)
        ng = random.randint(2, min(4, n // 2))
        idx = list(range(n))
        random.shuffle(idx)
        for g in range(ng):
            hi, ai = idx[g*2], idx[g*2+1]
            hd, ad = teams[hi], teams[ai]
            form_h = _generate_form(hd[1], hd[2])
            form_a = _generate_form(ad[1], ad[2])
            fp_h = _form_points(form_h)
            fp_a = _form_points(form_a)
            hs = TeamStats(team_id=hash(hd[0])%100000, team_name=hd[0],
                          attack_strength=hd[1], defense_strength=hd[2],
                          home_goals_scored_avg=round(hd[1]*1.4,2), home_goals_conceded_avg=round(hd[2]*1.0,2),
                          away_goals_scored_avg=round(hd[1]*1.0,2), away_goals_conceded_avg=round(hd[2]*1.3,2),
                          shots_on_target_avg=round(hd[1]*3.5,1), shots_blocked_avg=round(hd[2]*3,1),
                          corners_avg=round(hd[1]*3.8,1), cards_avg=round(1.5+hd[2]*0.8,1),
                          fouls_avg=round(10+hd[2]*3,1), possession_final_third=round(hd[1]*25,1),
                          form_last10=form_h, form_points=fp_h,
                          league_position=hi+1, league_points=int(fp_h*60),
                          games_played=25, games_remaining=13,
                          points_to_title=max(0,60-int(fp_h*60)), points_to_relegation=max(0,int(fp_h*60)-25))
            aws = TeamStats(team_id=hash(ad[0])%100000, team_name=ad[0],
                          attack_strength=ad[1], defense_strength=ad[2],
                          home_goals_scored_avg=round(ad[1]*1.4,2), home_goals_conceded_avg=round(ad[2]*1.0,2),
                          away_goals_scored_avg=round(ad[1]*1.0,2), away_goals_conceded_avg=round(ad[2]*1.3,2),
                          shots_on_target_avg=round(ad[1]*3.5,1), shots_blocked_avg=round(ad[2]*3,1),
                          corners_avg=round(ad[1]*3.8,1), cards_avg=round(1.5+ad[2]*0.8,1),
                          fouls_avg=round(10+ad[2]*3,1), possession_final_third=round(ad[1]*25,1),
                          form_last10=form_a, form_points=fp_a,
                          league_position=ai+1, league_points=int(fp_a*60),
                          games_played=25, games_remaining=13,
                          points_to_title=max(0,60-int(fp_a*60)), points_to_relegation=max(0,int(fp_a*60)-25))
            matches.append(MatchAnalysis(
                match_id=mid, league_id=league["id"], league_name=league["name"],
                league_country=league["country"], match_date=date,
                match_time=random.choice(["15:00","17:00","20:00","21:00"]),
                venue_name=f"Estádio {hd[0]}", home_team=hs, away_team=aws,
                odds=MarketOdds(home_win=round(1/max(.1,fp_h)*1.8,2), draw=3.3,
                               away_win=round(1/max(.1,fp_a)*1.8,2)),
            ))
            mid += 1
    return matches


# ═══════════════════════════════════════════════════════
# INTERFACE PÚBLICA
# ═══════════════════════════════════════════════════════

def ingest_all_fixtures(analysis_dates: list[str] = None) -> list[MatchAnalysis]:
    """
    Pipeline principal de ingestão de dados.
    Escolhe automaticamente entre API real e dados sintéticos.
    Aceita lista de datas customizada; default = config.ANALYSIS_DATES.
    """
    if analysis_dates is None:
        analysis_dates = config.ANALYSIS_DATES

    if config.USE_MOCK_DATA:
        print("[ETL] Modo: DADOS SINTÉTICOS (Demo)")
        all_matches = []
        for date in analysis_dates:
            print(f"[ETL] Gerando dados sintéticos para {date}...")
            matches = generate_synthetic_fixtures(date)
            all_matches.extend(matches)
            print(f"[ETL] {len(matches)} jogos gerados para {date}")
        print(f"[ETL] Total: {len(all_matches)} jogos")
        return all_matches
    else:
        print("[ETL] Modo: API REAL (Produção)")
        print(f"[ETL] Datas solicitadas: {analysis_dates}")
        print(f"[ETL] API Key: {config.API_FOOTBALL_KEY[:8]}...{config.API_FOOTBALL_KEY[-4:]}")
        print(f"[ETL] Weather: {'Configurado' if config.OPENWEATHER_KEY else 'Não configurado'}")
        return _ingest_real_data(analysis_dates=analysis_dates)


if __name__ == "__main__":
    fixtures = ingest_all_fixtures()
    print(f"\n{'='*60}")
    print(f"  Resumo: {len(fixtures)} partidas carregadas")
    print(f"  Ligas: {len(set(m.league_name for m in fixtures))}")
    print(f"{'='*60}")
    for m in fixtures[:10]:
        print(f"  {m.league_name}: {m.home_team.team_name} vs {m.away_team.team_name} "
              f"| Odds: {m.odds.home_win}/{m.odds.draw}/{m.odds.away_win}"
              f"| Pos: {m.home_team.league_position} vs {m.away_team.league_position}")
