"""
═══════════════════════════════════════════════════════════════════════
  ENGINE DE ANÁLISE PREDITIVA ESPORTIVA — SERVIDOR WEB
  Persistência em JSON — dados sobrevivem a restarts do servidor
═══════════════════════════════════════════════════════════════════════
"""

import json
import os
import time
import sys
import io

# ═══ FIX ENCODING WINDOWS (cp1252 → utf-8) ═══
# Força UTF-8 em TODOS os streams de I/O do Python no Windows
# Isso evita UnicodeEncodeError com emojis/acentos em qualquer print()
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

def _force_utf8():
    """Força encoding UTF-8 em stdout e stderr."""
    for stream_name in ('stdout', 'stderr'):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        if hasattr(stream, 'reconfigure'):
            try:
                stream.reconfigure(encoding='utf-8', errors='replace')
                continue
            except Exception:
                pass
        if hasattr(stream, 'buffer'):
            try:
                new_stream = io.TextIOWrapper(
                    stream.buffer, encoding='utf-8', errors='replace',
                    line_buffering=True
                )
                setattr(sys, stream_name, new_stream)
            except Exception:
                pass

_force_utf8()
from datetime import datetime
from flask import Flask, render_template, jsonify

import config
from data_ingestion import ingest_all_fixtures, _check_api_plan, _api_call_count
from models import run_models_batch
from context_engine import apply_context_batch
from value_finder import find_all_value, ValueOpportunity
from data_ingestion import MatchAnalysis
import supabase_client

app = Flask(__name__, template_folder="templates", static_folder="static")

CACHE_FILE = os.path.join(os.path.dirname(__file__), "_cache_data.json")

# Cache global
_cache = {
    "matches": None,
    "opportunities": None,
    "run_time": None,
    "stats": None,
    "last_run_at": None,
    "api_calls_used": 0,
}

# ═══════════════════════════════════════════════
# PERSISTÊNCIA JSON
# ═══════════════════════════════════════════════

def _save_cache_to_disk():
    """Salva resultados serializados em JSON para sobreviver a restarts."""
    try:
        data = {
            "stats": _cache["stats"],
            "last_run_at": _cache["last_run_at"],
            "api_calls_used": _cache["api_calls_used"],
            "opportunities": [serialize_opportunity(o) for o in (_cache["opportunities"] or [])],
            "matches": [serialize_match(m) for m in (_cache["matches"] or [])],
            "leagues": _build_leagues_list(),
        }
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"[CACHE] Dados salvos em {CACHE_FILE}")
    except Exception as e:
        print(f"[CACHE] Erro ao salvar: {e}")


def _load_cache_from_disk() -> bool:
    """Carrega dados do JSON. Se não existir, tenta carregar do Supabase."""
    if not os.path.exists(CACHE_FILE):
        # Tentar carregar do Supabase
        print("[CACHE] Cache local não encontrado. Tentando recuperar do Supabase...")
        if _load_cache_from_supabase():
            return True
        return False
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        _cache["stats"] = data.get("stats")
        _cache["last_run_at"] = data.get("last_run_at")
        _cache["api_calls_used"] = data.get("api_calls_used", 0)
        # Guardar como dicts (já serializados) — aplicar filtros de sanidade
        raw_opps = data.get("opportunities", [])
        raw_matches = data.get("matches", [])

        # Construir mapa de xG total por match_id para filtro de sanidade
        xg_map = {}
        for m in raw_matches:
            mid = m.get("match_id")
            xg_map[mid] = (m.get("home_xg", 0) or 0) + (m.get("away_xg", 0) or 0)

        # Construir mapa de qualidade de dados por match_id
        dq_map = {}
        for m in raw_matches:
            mid = m.get("match_id")
            dq_map[mid] = m.get("data_quality", 0)

        # SEM FILTROS RESTRITIVOS — exibir TODAS as análises para refinamento do algoritmo
        # Apenas filtro mínimo de qualidade de dados (para evitar lixo sem dados reais)
        def _is_opp_sane(o):
            match_dq = dq_map.get(o.get("match_id"), o.get("data_quality", 0))
            if match_dq < 0.30:  # Apenas rejeitar se quase sem dados reais
                return False
            return True

        filtered_opps = [o for o in raw_opps if _is_opp_sane(o)]
        _cache["_serialized_opportunities"] = filtered_opps
        _cache["_serialized_matches"] = raw_matches
        _cache["_serialized_leagues"] = data.get("leagues", [])
        # Atualizar stats para refletir dados
        if _cache["stats"]:
            removed = len(raw_opps) - len(filtered_opps)
            _cache["stats"]["total_opportunities"] = len(filtered_opps)
            _cache["stats"]["high_conf"] = sum(1 for o in filtered_opps if o.get("confidence") == "ALTO")
            _cache["stats"]["med_conf"] = sum(1 for o in filtered_opps if o.get("confidence") == "MÉDIO")
            _cache["stats"]["low_conf"] = sum(1 for o in filtered_opps if o.get("confidence") == "BAIXO")
            if filtered_opps:
                _cache["stats"]["avg_edge"] = round(sum(o.get("edge", 0) for o in filtered_opps) / len(filtered_opps), 2)
                _cache["stats"]["max_edge"] = max(o.get("edge", 0) for o in filtered_opps)
                top = max(filtered_opps, key=lambda o: o.get("edge", 0))
                _cache["stats"]["max_edge_match"] = f"{top.get('home_team')} vs {top.get('away_team')}"
                _cache["stats"]["max_edge_selection"] = top.get("selection", "")
            if removed > 0:
                print(f"[CACHE] Filtrados {removed} oportunidades por qualidade de dados insuficiente")
        # Marcar que tem dados (mas são dicts, não objetos)
        _cache["matches"] = "FROM_DISK"
        _cache["opportunities"] = "FROM_DISK"
        print(f"[CACHE] Dados carregados de {CACHE_FILE} ({len(filtered_opps)} oportunidades)")
        print(f"[CACHE] Última execução: {_cache['last_run_at']}")
        return True
    except Exception as e:
        print(f"[CACHE] Erro ao carregar do disco: {e}")
        # Tentar carregar do Supabase como fallback
        if _load_cache_from_supabase():
            return True
        return False


def _load_cache_from_supabase() -> bool:
    """Tenta carregar a última execução do Supabase. Retorna True se conseguiu."""
    if not supabase_client.is_configured():
        print("[CACHE] Supabase não configurado — não é possível recuperar dados")
        return False
    
    try:
        print("[CACHE] Buscando última execução no Supabase...")
        runs = supabase_client.get_run_history(limit=1)
        if not runs:
            print("[CACHE] Nenhuma execução encontrada no Supabase")
            return False
        
        latest_run = runs[0]
        run_id = latest_run.get("id")
        print(f"[CACHE] Última execução encontrada: {run_id}")
        print(f"[CACHE] Data: {latest_run.get('executed_at', '?')}")
        
        # Carregar oportunidades e partidas desta execução
        opps = supabase_client.get_opportunities_by_run(run_id)
        matches = supabase_client.get_matches_by_run(run_id)
        
        if not opps and not matches:
            print("[CACHE] Execução encontrada mas sem dados (oportunidades/partidas)")
            return False
        
        print(f"[CACHE] Recuperados: {len(opps)} oportunidades, {len(matches)} partidas")
        
        # Construir stats a partir dos dados recuperados
        _cache["stats"] = {
            "total_matches": len(matches),
            "total_leagues": len(set(m.get("league_name", "") for m in matches)),
            "total_opportunities": len(opps),
            "high_conf": sum(1 for o in opps if o.get("confidence") == "ALTO"),
            "med_conf": sum(1 for o in opps if o.get("confidence") == "MÉDIO"),
            "low_conf": sum(1 for o in opps if o.get("confidence") == "BAIXO"),
            "avg_edge": round(sum(o.get("edge", 0) for o in opps) / max(1, len(opps)), 2) if opps else 0,
            "max_edge": max((o.get("edge", 0) for o in opps), default=0),
            "max_edge_match": f"{opps[0].get('home_team', '')} vs {opps[0].get('away_team', '')}" if opps else "",
            "max_edge_selection": opps[0].get("selection", "") if opps else "",
            "run_time": latest_run.get("run_time_seconds", 0),
            "analysis_dates": latest_run.get("analysis_dates", config.ANALYSIS_DATES),
            "mode": latest_run.get("mode", "API Real"),
            "last_run_at": latest_run.get("executed_at", ""),
            "api_calls_this_run": latest_run.get("api_calls_used", 0),
        }
        
        # Construir mapa de qualidade de dados para filtro
        dq_map = {}
        for m in matches:
            dq_map[m.get("match_id")] = m.get("data_quality", 0)
        
        # Construir dict de matches para lookup
        matches_dict = {m.get("match_id"): m for m in matches}
        
        # SEM FILTROS RESTRITIVOS — exibir TODAS as análises para refinamento
        def _is_opp_sane(o):
            match_dq = dq_map.get(o.get("match_id"), o.get("data_quality", 0))
            if match_dq < 0.30:  # Apenas rejeitar se quase sem dados reais
                return False
            return True
        
        filtered_opps = [o for o in opps if _is_opp_sane(o)]
        
        _cache["_serialized_opportunities"] = filtered_opps
        _cache["_serialized_matches"] = matches
        _cache["_serialized_leagues"] = _build_leagues_list_from_matches(matches)
        _cache["last_run_at"] = latest_run.get("executed_at", "")
        _cache["api_calls_used"] = latest_run.get("api_calls_used", 0)
        _cache["matches"] = "FROM_SUPABASE"
        _cache["opportunities"] = "FROM_SUPABASE"
        
        if len(filtered_opps) < len(opps):
            print(f"[CACHE] Filtrados {len(opps) - len(filtered_opps)} oportunidades (sanidade)")
        
        print(f"[CACHE] Dados recuperados do Supabase com sucesso")
        print(f"[CACHE] Última execução: {_cache['last_run_at']}")
        return True
        
    except Exception as e:
        print(f"[CACHE] Erro ao carregar do Supabase: {e}")
        return False


def _build_leagues_list_from_matches(matches: list[dict]) -> list[dict]:
    """Constrói lista de ligas a partir de matches serializados."""
    leagues = {}
    for m in matches:
        key = m.get("league_name", "Desconhecida")
        if key not in leagues:
            leagues[key] = {
                "name": key,
                "country": m.get("league_country", "N/D"),
                "matches": 0,
                "opportunities": 0,
            }
        leagues[key]["matches"] += 1
    return list(leagues.values())


def _build_leagues_list() -> list:
    """Constrói lista de ligas a partir dos dados em memória."""
    if not _cache["matches"] or _cache["matches"] in ("FROM_DISK", "FROM_SUPABASE"):
        return _cache.get("_serialized_leagues", [])
    leagues = {}
    for m in _cache["matches"]:
        key = m.league_name
        if key not in leagues:
            leagues[key] = {
                "name": m.league_name,
                "country": m.league_country,
                "matches": 0,
                "opportunities": 0,
            }
        leagues[key]["matches"] += 1
    if _cache["opportunities"] and _cache["opportunities"] not in ("FROM_DISK", "FROM_SUPABASE"):
        for o in _cache["opportunities"]:
            key = o.league_name
            if key in leagues:
                leagues[key]["opportunities"] += 1
    return list(leagues.values())


# ═══════════════════════════════════════════════
# ENGINE
# ═══════════════════════════════════════════════

def run_engine():
    """Executa o pipeline completo, cacheia e persiste em disco."""
    _force_utf8()  # Garantir UTF-8 no contexto do request Flask
    start = time.time()

    matches = ingest_all_fixtures()
    matches = run_models_batch(matches)
    matches = apply_context_batch(matches)
    opportunities = find_all_value(matches)

    elapsed = round(time.time() - start, 2)
    n_leagues = len(set(m.league_name for m in matches))

    from data_ingestion import _api_call_count
    now = datetime.now()

    _cache["matches"] = matches
    _cache["opportunities"] = opportunities
    _cache["run_time"] = elapsed
    _cache["last_run_at"] = now.strftime("%d/%m/%Y %H:%M:%S")
    _cache["api_calls_used"] = _api_call_count

    _cache["stats"] = {
        "total_matches": len(matches),
        "total_leagues": n_leagues,
        "total_opportunities": len(opportunities),
        "high_conf": sum(1 for o in opportunities if o.confidence == "ALTO"),
        "med_conf": sum(1 for o in opportunities if o.confidence == "MÉDIO"),
        "low_conf": sum(1 for o in opportunities if o.confidence == "BAIXO"),
        "avg_edge": round(sum(o.edge for o in opportunities) / max(1, len(opportunities)) * 100, 2),
        "max_edge": round(opportunities[0].edge * 100, 1) if opportunities else 0,
        "max_edge_match": (f"{opportunities[0].home_team} vs {opportunities[0].away_team}"
                           if opportunities else ""),
        "max_edge_selection": opportunities[0].selection if opportunities else "",
        "run_time": elapsed,
        "analysis_dates": config.ANALYSIS_DATES,
        "mode": "Dados Sintéticos (Demo)" if config.USE_MOCK_DATA else "API Real (PRO)",
        "last_run_at": _cache["last_run_at"],
        "api_calls_this_run": _api_call_count,
    }

    # Persistir em disco (local)
    _save_cache_to_disk()

    # Persistir no Supabase (nuvem) — OBRIGATÓRIO
    print("[APP] Salvando dados no Supabase...")
    try:
        serialized_opps = [serialize_opportunity(o) for o in opportunities]
        serialized_matches = [serialize_match(m) for m in matches]
        supabase_client.save_full_run(_cache["stats"], serialized_opps, serialized_matches)
        print("[APP] Dados salvos no Supabase com sucesso")
    except Exception as e:
        print(f"[APP] ERRO CRITICO ao salvar no Supabase: {e}")
        print("[APP] AVISO: Dados salvos apenas localmente. Verifique configuração do Supabase.")

    return _cache


# ═══════════════════════════════════════════════
# SERIALIZAÇÃO
# ═══════════════════════════════════════════════

def serialize_opportunity(o: ValueOpportunity) -> dict:
    return {
        "match_id": o.match_id,
        "league_name": o.league_name,
        "league_country": o.league_country,
        "match_date": o.match_date,
        "match_time": o.match_time,
        "home_team": o.home_team,
        "away_team": o.away_team,
        "market": o.market,
        "selection": o.selection,
        "market_odd": o.market_odd,
        "fair_odd": o.fair_odd,
        "model_prob": round(o.model_prob * 100, 1),
        "implied_prob": round(o.implied_prob * 100, 1),
        "edge": round(o.edge * 100, 1),
        "edge_pct": o.edge_pct,
        "kelly_bet_pct": o.kelly_bet_pct,
        "confidence": o.confidence,
        "reasoning": o.reasoning,
        "home_xg": o.home_xg,
        "away_xg": o.away_xg,
        "weather_note": o.weather_note,
        "fatigue_note": o.fatigue_note,
        "urgency_home": o.urgency_home,
        "urgency_away": o.urgency_away,
        "bookmaker": o.bookmaker,
        "data_quality": o.data_quality,
    }


def serialize_match(m: MatchAnalysis) -> dict:
    h = m.home_team
    a = m.away_team
    return {
        "match_id": m.match_id,
        "league_id": m.league_id,
        "league_name": m.league_name,
        "league_country": m.league_country,
        "match_date": m.match_date,
        "match_time": m.match_time,
        "home_team": h.team_name,
        "home_team_id": h.team_id,
        "away_team_id": a.team_id,
        "away_team": a.team_name,
        "home_xg": round(m.model_home_xg, 2),
        "away_xg": round(m.model_away_xg, 2),
        "prob_home": round(m.model_prob_home * 100, 1),
        "prob_draw": round(m.model_prob_draw * 100, 1),
        "prob_away": round(m.model_prob_away * 100, 1),
        "prob_over25": round(m.model_prob_over25 * 100, 1),
        "prob_btts": round(m.model_prob_btts * 100, 1),
        "corners_expected": round(m.model_corners_expected, 1),
        "cards_expected": round(m.model_cards_expected, 1),
        "odds_home": m.odds.home_win,
        "odds_draw": m.odds.draw,
        "odds_away": m.odds.away_win,
        "weather_temp": m.weather.temperature_c,
        "weather_wind": m.weather.wind_speed_kmh,
        "weather_rain": m.weather.rain_mm,
        "weather_desc": m.weather.description,
        "home_fatigue": m.home_fatigue,
        "away_fatigue": m.away_fatigue,
        "urgency_home": m.league_urgency_home,
        "urgency_away": m.league_urgency_away,
        "injuries_home": m.injuries_home,
        "injuries_away": m.injuries_away,
        "referee": m.referee.name,
        "referee_cards_avg": m.referee.cards_per_game_avg,
        "referee_fouls_avg": m.referee.fouls_per_game_avg,
        "home_form": h.form_last10,
        "away_form": a.form_last10,
        "venue": m.venue_name,
        "bookmaker": m.odds.bookmaker,
        # Dados de força dos times para análise detalhada
        "home_attack": round(h.attack_strength, 2),
        "home_defense": round(h.defense_strength, 2),
        "away_attack": round(a.attack_strength, 2),
        "away_defense": round(a.defense_strength, 2),
        "home_goals_scored_avg": round(h.home_goals_scored_avg, 2),
        "home_goals_conceded_avg": round(h.home_goals_conceded_avg, 2),
        "away_goals_scored_avg": round(a.away_goals_scored_avg, 2),
        "away_goals_conceded_avg": round(a.away_goals_conceded_avg, 2),
        "home_form_points": round(h.form_points, 2),
        "away_form_points": round(a.form_points, 2),
        "home_league_pos": h.league_position,
        "away_league_pos": a.league_position,
        "home_league_pts": h.league_points,
        "away_league_pts": a.league_points,
        # Estatísticas detalhadas dos times
        "home_shots_on_target_avg": round(h.shots_on_target_avg, 1),
        "away_shots_on_target_avg": round(a.shots_on_target_avg, 1),
        "home_shots_blocked_avg": round(h.shots_blocked_avg, 1),
        "away_shots_blocked_avg": round(a.shots_blocked_avg, 1),
        "home_corners_avg": round(h.corners_avg, 1),
        "away_corners_avg": round(a.corners_avg, 1),
        "home_cards_avg": round(h.cards_avg, 1),
        "away_cards_avg": round(a.cards_avg, 1),
        "home_fouls_avg": round(h.fouls_avg, 1),
        "away_fouls_avg": round(a.fouls_avg, 1),
        "home_possession": round(h.possession_final_third, 1),
        "away_possession": round(a.possession_final_third, 1),
        "home_games_played": h.games_played,
        "away_games_played": a.games_played,
        "home_games_remaining": h.games_remaining,
        "away_games_remaining": a.games_remaining,
        "home_points_to_title": h.points_to_title,
        "away_points_to_title": a.points_to_title,
        "home_points_to_relegation": h.points_to_relegation,
        "away_points_to_relegation": a.points_to_relegation,
        # Odds extras
        "odds_over25": m.odds.over_25,
        "odds_under25": m.odds.under_25,
        "odds_btts_yes": m.odds.btts_yes,
        "odds_btts_no": m.odds.btts_no,
        "odds_corners_over": m.odds.over_95_corners,
        "odds_corners_under": m.odds.under_95_corners,
        "odds_cards_over": m.odds.over_35_cards,
        "odds_cards_under": m.odds.under_35_cards,
        "odds_ah_line": m.odds.asian_handicap_line,
        "odds_ah_home": m.odds.asian_handicap_home,
        "odds_ah_away": m.odds.asian_handicap_away,
        "odds_1x": m.odds.double_chance_1x,
        "odds_x2": m.odds.double_chance_x2,
        # Qualidade dos dados
        "data_quality": m.data_quality_score,
        "has_real_odds": m.has_real_odds,
        "has_real_standings": m.has_real_standings,
        "has_real_weather": m.has_real_weather,
        "home_has_real_data": h.has_real_data,
        "away_has_real_data": a.has_real_data,
    }


# ═══════════════════════════════════════════════
# ROTAS
# ═══════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/run", methods=["POST"])
def api_run():
    """Executa o engine e retorna os resultados."""
    try:
        run_engine()
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

    return jsonify({
        "ok": True,
        "last_run_at": _cache["last_run_at"],
        "api_calls_this_run": _cache["api_calls_used"],
    })


@app.route("/api/status")
def api_status():
    """Retorna status da API (requests usadas/restantes). Consome 1 request."""
    has_data = _cache["matches"] is not None
    try:
        plan_info = _check_api_plan()
        return jsonify({
            "ok": True,
            "plan": plan_info["plan"],
            "limit": plan_info["limit"],
            "used": plan_info["used"],
            "available": plan_info["available"],
            "last_run_at": _cache.get("last_run_at"),
            "api_calls_last_run": _cache.get("api_calls_used", 0),
            "has_data": has_data,
            "analysis_dates": config.ANALYSIS_DATES,
        })
    except Exception as e:
        return jsonify({
            "ok": False,
            "error": str(e),
            "has_data": has_data,
            "last_run_at": _cache.get("last_run_at"),
            "analysis_dates": config.ANALYSIS_DATES,
        })


@app.route("/api/stats")
def api_stats():
    if not _cache["stats"]:
        return jsonify({"ok": False, "msg": "Engine não executado ainda"}), 200
    return jsonify(_cache["stats"])


@app.route("/api/opportunities")
def api_opportunities():
    # Se carregado do disco, retornar dicts direto
    if _cache.get("opportunities") in ("FROM_DISK", "FROM_SUPABASE"):
        return jsonify(_cache.get("_serialized_opportunities", []))
    if not _cache["opportunities"]:
        return jsonify([])
    return jsonify([serialize_opportunity(o) for o in _cache["opportunities"]])


@app.route("/api/matches")
def api_matches():
    if _cache.get("matches") in ("FROM_DISK", "FROM_SUPABASE"):
        return jsonify(_cache.get("_serialized_matches", []))
    if not _cache["matches"]:
        return jsonify([])
    return jsonify([serialize_match(m) for m in _cache["matches"]])


@app.route("/api/leagues")
def api_leagues():
    if _cache.get("matches") in ("FROM_DISK", "FROM_SUPABASE"):
        return jsonify(_cache.get("_serialized_leagues", []))
    if not _cache["matches"]:
        return jsonify([])
    return jsonify(_build_leagues_list())


@app.route("/api/team-history/<int:team_id>")
def api_team_history(team_id):
    """Busca histórico completo de um time (sob demanda).
    Usa cache local para não gastar créditos desnecessariamente.
    Query params: league_id (opcional), last (default 10)"""
    from flask import request as flask_request
    from data_ingestion import fetch_team_history

    league_id = flask_request.args.get("league_id", None, type=int)
    last = flask_request.args.get("last", 10, type=int)
    last = min(last, 15)  # Limitar a 15 para economia

    try:
        history = fetch_team_history(team_id, league_id=league_id, last=last)
        return jsonify(history)
    except Exception as e:
        print(f"[API] Erro ao buscar historico do time {team_id}: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "team_id": team_id, "all_matches": [], "league_matches": []}), 500


@app.route("/api/history")
def api_history():
    """Retorna histórico de execuções (do Supabase)."""
    runs = supabase_client.get_run_history(limit=20)
    return jsonify(runs)


@app.route("/api/history/<run_id>/opportunities")
def api_history_opportunities(run_id):
    """Retorna oportunidades de uma execução passada."""
    opps = supabase_client.get_opportunities_by_run(run_id)
    return jsonify(opps)


@app.route("/api/opportunity/<opp_id>/result", methods=["POST"])
def api_update_result(opp_id):
    """Atualiza resultado de uma oportunidade (GREEN/RED/VOID)."""
    from flask import request
    data = request.get_json()
    status = data.get("status", "")
    score = data.get("score", "")
    if status not in ("GREEN", "RED", "VOID", "POSTPONED"):
        return jsonify({"ok": False, "error": "Status inválido"}), 400
    ok = supabase_client.update_opportunity_result(opp_id, status, score)
    return jsonify({"ok": ok})


@app.route("/api/opportunity/<opp_id>/bet", methods=["POST"])
def api_register_bet(opp_id):
    """Registra uma aposta feita pelo usuário."""
    from flask import request
    data = request.get_json()
    amount = data.get("amount", 0)
    bet_return = data.get("return")
    notes = data.get("notes", "")
    ok = supabase_client.update_bet_info(opp_id, amount, bet_return, notes)
    return jsonify({"ok": ok})


if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  ApostasIA Engine — Servidor Web")
    print("=" * 55)
    mode_label = "API REAL (PRO)" if not config.USE_MOCK_DATA else "DADOS SINTÉTICOS"
    print(f"  Modo: {mode_label}")
    print(f"  Datas de análise: {config.ANALYSIS_DATES}")
    print(f"  Pipeline: MANUAL (não roda ao iniciar)")

    # Tentar carregar dados anteriores do disco
    if _load_cache_from_disk():
        stats = _cache.get("stats", {})
        print(f"  Dados anteriores carregados!")
        print(f"     Última execução: {_cache.get('last_run_at', '?')}")
        print(f"     {stats.get('total_matches', 0)} jogos | {stats.get('total_opportunities', 0)} oportunidades")
    else:
        print(f"  Sem dados anteriores. Use o botão para executar.")

    print("=" * 55)
    print(f"  Acesse: http://localhost:5000")
    print("=" * 55)
    print()

    app.run(host="0.0.0.0", port=5000, debug=False)
