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
from datetime import datetime, timezone, timedelta
from flask import Flask, render_template, jsonify

import config
from data_ingestion import (
    ingest_all_fixtures, _check_api_plan, _api_call_count,
    MatchAnalysis, TeamStats, WeatherData, RefereeStats, MarketOdds,
    _get_cached_response, _parse_odds_response,
)
from models import run_models_batch
from context_engine import apply_context_batch
from value_finder import find_all_value, ValueOpportunity
import supabase_client
import numpy as np

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["TEMPLATES_AUTO_RELOAD"] = True

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
# CONVERSÃO DE FUSO HORÁRIO (UTC → Brasília)
# ═══════════════════════════════════════════════

_UTC = timezone.utc

def _convert_utc_to_br(date_str: str, time_str: str) -> tuple:
    """Converte date/time strings de UTC para fuso de Brasília (UTC-3).
    Retorna (new_date, new_time). Se falhar, retorna os originais."""
    try:
        dt_utc = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
        dt_utc = dt_utc.replace(tzinfo=_UTC)
        dt_br = dt_utc.astimezone(config.BR_TIMEZONE)
        return dt_br.strftime("%Y-%m-%d"), dt_br.strftime("%H:%M")
    except Exception:
        return date_str, time_str


def _convert_cached_data_timezone(matches: list[dict], opps: list[dict]):
    """Converte horários de UTC para Brasília em dados cacheados (dicts).
    Modifica as listas in-place."""
    converted = 0
    for m in matches:
        d = m.get("match_date", "")
        t = m.get("match_time", "")
        if d and t and t != "N/D":
            new_d, new_t = _convert_utc_to_br(d, t)
            m["match_date"] = new_d
            m["match_time"] = new_t
            if new_t != t:
                converted += 1
    for o in opps:
        d = o.get("match_date", "")
        t = o.get("match_time", "")
        if d and t and t != "N/D":
            new_d, new_t = _convert_utc_to_br(d, t)
            o["match_date"] = new_d
            o["match_time"] = new_t
    if converted > 0:
        print(f"[TZ] Convertidos {converted} horários de UTC → Brasília (UTC-3)")


# ═══════════════════════════════════════════════
# PERSISTÊNCIA JSON
# ═══════════════════════════════════════════════

def _save_results_to_disk_cache(updates: list[dict]):
    """Atualiza APENAS os result_status/result_score no cache de disco, sem sobrescrever tudo."""
    if not os.path.exists(CACHE_FILE) or not updates:
        return
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        opps = data.get("opportunities", [])
        if not opps:
            return

        # Mapear updates por id e match_id
        update_by_id = {u["id"]: u for u in updates}
        update_by_match = {}
        for u in updates:
            mid = u.get("match_id")
            if mid is not None:
                update_by_match.setdefault(mid, []).append(u)

        count = 0
        for opp in opps:
            opp_id = opp.get("id")
            mid = opp.get("match_id")

            # Match por UUID
            if opp_id and opp_id in update_by_id:
                u = update_by_id[opp_id]
                opp["result_status"] = u["result_status"]
                opp["result_score"] = u["result_score"]
                opp["result_ht_score"] = u.get("result_ht_score", "")
                opp["result_corners"] = u.get("result_corners", "")
                opp["result_cards"] = u.get("result_cards", "")
                opp["result_shots"] = u.get("result_shots", "")
                count += 1
                continue

            # Fallback por match_id
            if mid in update_by_match:
                for u in update_by_match[mid]:
                    opp["result_status"] = u["result_status"]
                    opp["result_score"] = u["result_score"]
                    opp["result_ht_score"] = u.get("result_ht_score", "")
                    opp["result_corners"] = u.get("result_corners", "")
                    opp["result_cards"] = u.get("result_cards", "")
                    opp["result_shots"] = u.get("result_shots", "")
                    count += 1
                    break

        if count > 0:
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            print(f"[CACHE] Resultados salvos no disco: {count} oportunidades atualizadas")
    except Exception as e:
        print(f"[CACHE] Erro ao salvar resultados no disco: {e}")


def _save_cache_to_disk():
    """Salva resultados serializados em JSON para sobreviver a restarts."""
    try:
        data = {
            "_timezone": "America/Sao_Paulo",
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

        # ── Conversão de fuso horário: dados antigos em UTC → Brasília ──
        if data.get("_timezone") != "America/Sao_Paulo":
            print("[CACHE] Dados em UTC detectados — convertendo para horário de Brasília...")
            _convert_cached_data_timezone(raw_matches, raw_opps)

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

        # ── Mesclar resultados do Supabase (fonte de verdade para GREEN/RED) ──
        _merge_results_from_supabase(filtered_opps)

        return True
    except Exception as e:
        print(f"[CACHE] Erro ao carregar do disco: {e}")
        # Tentar carregar do Supabase como fallback
        if _load_cache_from_supabase():
            return True
        return False


def _preserve_existing_results(opportunities):
    """
    Preserva result_status/result_score de oportunidades já resolvidas no Supabase.
    Chamado após recalculate ou run_engine para não perder GREEN/RED/VOID.
    Funciona com objetos ValueOpportunity.
    """
    if not supabase_client.is_configured():
        return
    try:
        sb = supabase_client.get_client()
        if not sb:
            return

        try:
            result = (
                sb.table("opportunities")
                .select("match_id, market, selection, result_status, result_score, result_ht_score, result_corners, result_cards, result_shots")
                .neq("result_status", "PENDENTE")
                .execute()
            )
        except Exception:
            result = (
                sb.table("opportunities")
                .select("match_id, market, selection, result_status, result_score")
                .neq("result_status", "PENDENTE")
                .execute()
            )
        resolved = result.data or []
        if not resolved:
            return

        # Mapa por (match_id, market_lower, selection_lower)
        resolved_map = {}
        for r in resolved:
            key = (r.get("match_id"), (r.get("market") or "").lower(), (r.get("selection") or "").lower())
            resolved_map[key] = r

        count = 0
        for opp in opportunities:
            key = (opp.match_id, opp.market.lower(), opp.selection.lower())
            if key in resolved_map:
                r = resolved_map[key]
                opp.result_status = r["result_status"]
                opp.result_score = r.get("result_score", "")
                opp.result_ht_score = r.get("result_ht_score", "")
                opp.result_corners = r.get("result_corners", "")
                opp.result_cards = r.get("result_cards", "")
                opp.result_shots = r.get("result_shots", "")
                count += 1

        if count > 0:
            print(f"[RECALC] ✅ Preservados {count} resultados existentes (GREEN/RED/VOID) do Supabase")
    except Exception as e:
        print(f"[RECALC] Aviso: Não foi possível preservar resultados: {e}")


def _merge_results_from_supabase(opps: list[dict]):
    """
    Busca result_status/result_score do Supabase e mescla nas oportunidades locais.
    O Supabase é a fonte de verdade para resultados de apostas.
    """
    if not supabase_client.is_configured():
        return
    try:
        sb = supabase_client.get_client()
        if not sb:
            return

        # Buscar APENAS oportunidades que já foram resolvidas (não PENDENTE)
        # Tentar com colunas detalhadas, fallback para sem elas (caso migração ainda não foi executada)
        try:
            result = (
                sb.table("opportunities")
                .select("id, match_id, market, selection, result_status, result_score, result_ht_score, result_corners, result_cards, result_shots")
                .neq("result_status", "PENDENTE")
                .execute()
            )
        except Exception:
            result = (
                sb.table("opportunities")
                .select("id, match_id, market, selection, result_status, result_score")
                .neq("result_status", "PENDENTE")
                .execute()
            )
        resolved = result.data or []
        if not resolved:
            print("[CACHE] Nenhum resultado resolvido no Supabase")
            return

        # Criar mapa por id e por (match_id, market, selection)
        resolved_by_id = {r["id"]: r for r in resolved}
        resolved_by_key = {}
        for r in resolved:
            key = (r.get("match_id"), (r.get("market") or "").lower(), (r.get("selection") or "").lower())
            resolved_by_key[key] = r

        count = 0
        for opp in opps:
            # Se já tem resultado, pular
            if opp.get("result_status") and opp["result_status"] != "PENDENTE":
                continue

            opp_id = opp.get("id")
            mid = opp.get("match_id")

            def _apply_result(opp, r):
                opp["result_status"] = r["result_status"]
                opp["result_score"] = r.get("result_score", "")
                opp["result_ht_score"] = r.get("result_ht_score", "")
                opp["result_corners"] = r.get("result_corners", "")
                opp["result_cards"] = r.get("result_cards", "")
                opp["result_shots"] = r.get("result_shots", "")

            # Match por UUID
            if opp_id and opp_id in resolved_by_id:
                _apply_result(opp, resolved_by_id[opp_id])
                count += 1
                continue

            # Fallback por (match_id, market, selection)
            key = (mid, (opp.get("market") or "").lower(), (opp.get("selection") or "").lower())
            if key in resolved_by_key:
                _apply_result(opp, resolved_by_key[key])
                count += 1

        if count > 0:
            print(f"[CACHE] ✅ Mesclados {count} resultados do Supabase (GREEN/RED/VOID)")
        else:
            print("[CACHE] Sem resultados novos para mesclar do Supabase")

    except Exception as e:
        print(f"[CACHE] Aviso: Não foi possível mesclar resultados do Supabase: {e}")


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
        
        # ── Conversão de fuso horário: dados do Supabase em UTC → Brasília ──
        print("[CACHE] Convertendo horários do Supabase para fuso de Brasília...")
        _convert_cached_data_timezone(matches, filtered_opps)
        
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

def run_engine(analysis_dates: list[str] = None):
    """Executa o pipeline completo, cacheia e persiste em disco.
    Aceita lista customizada de datas; default = config.ANALYSIS_DATES."""
    _force_utf8()  # Garantir UTF-8 no contexto do request Flask
    start = time.time()

    if analysis_dates is None:
        analysis_dates = config.get_default_dates()

    matches = ingest_all_fixtures(analysis_dates=analysis_dates)
    matches = run_models_batch(matches)
    matches = apply_context_batch(matches)
    opportunities = find_all_value(matches)

    # Preservar resultados já resolvidos no Supabase
    _preserve_existing_results(opportunities)

    elapsed = round(time.time() - start, 2)
    n_leagues = len(set(m.league_name for m in matches))

    from data_ingestion import _api_call_count
    now = datetime.now(config.BR_TIMEZONE)

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
        "analysis_dates": analysis_dates,
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
# RECALCULAR (sem API calls — usa dados em cache)
# ═══════════════════════════════════════════════

def deserialize_match(d: dict) -> MatchAnalysis:
    """Reconstrói um MatchAnalysis a partir de um dict serializado."""
    home = TeamStats(
        team_id=d.get("home_team_id", 0),
        team_name=d.get("home_team", "Casa"),
        attack_strength=d.get("home_attack", 1.0),
        defense_strength=d.get("home_defense", 1.0),
        home_goals_scored_avg=d.get("home_goals_scored_avg", 1.3),
        home_goals_conceded_avg=d.get("home_goals_conceded_avg", 1.1),
        away_goals_scored_avg=d.get("away_goals_scored_avg", 1.0),
        away_goals_conceded_avg=d.get("away_goals_conceded_avg", 1.3),
        shots_total_avg=d.get("home_shots_total_avg", d.get("home_shots_on_target_avg", 4.0) * 2.8),
        shots_on_target_avg=d.get("home_shots_on_target_avg", 4.0),
        shots_blocked_avg=d.get("home_shots_blocked_avg", 3.0),
        corners_avg=d.get("home_corners_avg", 5.0),
        cards_avg=d.get("home_cards_avg", 2.0),
        fouls_avg=d.get("home_fouls_avg", 12.0),
        possession_final_third=d.get("home_possession", 30.0),
        form_last10=d.get("home_form", []),
        form_points=d.get("home_form_points", 0.0),
        league_position=d.get("home_league_pos", 0),
        league_points=d.get("home_league_pts", 0),
        games_played=d.get("home_games_played", 0),
        games_remaining=d.get("home_games_remaining", 0),
        points_to_title=d.get("home_points_to_title", 99),
        points_to_relegation=d.get("home_points_to_relegation", 99),
        has_real_data=d.get("home_has_real_data", False),
    )
    away = TeamStats(
        team_id=d.get("away_team_id", 0),
        team_name=d.get("away_team", "Fora"),
        attack_strength=d.get("away_attack", 1.0),
        defense_strength=d.get("away_defense", 1.0),
        home_goals_scored_avg=d.get("home_goals_scored_avg", 1.3),
        home_goals_conceded_avg=d.get("home_goals_conceded_avg", 1.1),
        away_goals_scored_avg=d.get("away_goals_scored_avg", 1.0),
        away_goals_conceded_avg=d.get("away_goals_conceded_avg", 1.3),
        shots_total_avg=d.get("away_shots_total_avg", d.get("away_shots_on_target_avg", 4.0) * 2.8),
        shots_on_target_avg=d.get("away_shots_on_target_avg", 4.0),
        shots_blocked_avg=d.get("away_shots_blocked_avg", 3.0),
        corners_avg=d.get("away_corners_avg", 5.0),
        cards_avg=d.get("away_cards_avg", 2.0),
        fouls_avg=d.get("away_fouls_avg", 12.0),
        possession_final_third=d.get("away_possession", 30.0),
        form_last10=d.get("away_form", []),
        form_points=d.get("away_form_points", 0.0),
        league_position=d.get("away_league_pos", 0),
        league_points=d.get("away_league_pts", 0),
        games_played=d.get("away_games_played", 0),
        games_remaining=d.get("away_games_remaining", 0),
        points_to_title=d.get("away_points_to_title", 99),
        points_to_relegation=d.get("away_points_to_relegation", 99),
        has_real_data=d.get("away_has_real_data", False),
    )
    weather = WeatherData(
        temperature_c=d.get("weather_temp", 20.0),
        wind_speed_kmh=d.get("weather_wind", 5.0),
        rain_mm=d.get("weather_rain", 0.0),
        humidity_pct=50.0,
        description=d.get("weather_desc", "N/D"),
    )
    ref = RefereeStats(
        name=d.get("referee", "Desconhecido"),
        cards_per_game_avg=d.get("referee_cards_avg", 4.0),
        fouls_per_game_avg=d.get("referee_fouls_avg", 25.0),
    )
    odds = MarketOdds(
        home_win=d.get("odds_home", 2.0),
        draw=d.get("odds_draw", 3.3),
        away_win=d.get("odds_away", 3.5),
        over_25=d.get("odds_over25", 1.85) if "odds_over25" in d else 1.85,
        under_25=d.get("odds_under25", 1.95),
        btts_yes=d.get("odds_btts_yes", 1.80),
        btts_no=d.get("odds_btts_no", 2.00),
        over_95_corners=d.get("odds_corners_over", 1.90),
        under_95_corners=d.get("odds_corners_under", 1.90),
        over_35_cards=d.get("odds_cards_over", 1.85),
        under_35_cards=d.get("odds_cards_under", 1.95),
        double_chance_1x=d.get("odds_1x", 0.0),
        double_chance_x2=d.get("odds_x2", 0.0),
        asian_handicap_line=d.get("odds_ah_line", -0.5),
        asian_handicap_home=d.get("odds_ah_home", 1.90),
        asian_handicap_away=d.get("odds_ah_away", 1.90),
        bookmaker=d.get("bookmaker", "N/D"),
        all_markets=d.get("all_markets", {}),
    )

    match = MatchAnalysis(
        match_id=d.get("match_id", 0),
        league_id=d.get("league_id", 0),
        league_name=d.get("league_name", "?"),
        league_country=d.get("league_country", "?"),
        match_date=d.get("match_date", ""),
        match_time=d.get("match_time", ""),
        venue_name=d.get("venue", ""),
        home_team=home,
        away_team=away,
        weather=weather,
        referee=ref,
        odds=odds,
        league_urgency_home=d.get("urgency_home", 0.5),
        league_urgency_away=d.get("urgency_away", 0.5),
        home_fatigue=d.get("home_fatigue", False),
        away_fatigue=d.get("away_fatigue", False),
        injuries_home=d.get("injuries_home", []),
        injuries_away=d.get("injuries_away", []),
        has_real_odds=d.get("has_real_odds", False),
        has_real_standings=d.get("has_real_standings", False),
        has_real_weather=d.get("has_real_weather", False),
        data_quality_score=d.get("data_quality", 0.0),
        odds_home_away_suspect=d.get("odds_home_away_suspect", False),
        league_avg_goals=d.get("league_avg_goals", 2.7),
        model_alpha_h=d.get("model_alpha_h", 1.0),
        model_beta_h=d.get("model_beta_h", 1.0),
        model_alpha_a=d.get("model_alpha_a", 1.0),
        model_beta_a=d.get("model_beta_a", 1.0),
        h2h_avg_goals=d.get("h2h_avg_goals"),
        h2h_count=d.get("h2h_count", 0),
    )
    return match


def recalculate_engine():
    """
    Recalcula TUDO (modelo + scanner de valor) usando dados JÁ em cache.
    ZERO requisições à API — apenas reprocessa com o código atualizado.
    """
    _force_utf8()
    start = time.time()

    # ── 1. Obter matches serializados do cache ──
    serialized_matches = None

    # Prioridade: cache local > Supabase
    if _cache.get("matches") in ("FROM_DISK", "FROM_SUPABASE"):
        serialized_matches = _cache.get("_serialized_matches", [])
    elif _cache.get("matches") and isinstance(_cache["matches"], list):
        # Já são objetos MatchAnalysis — serializar primeiro
        serialized_matches = [serialize_match(m) for m in _cache["matches"]]

    if not serialized_matches:
        # Tentar carregar do disco
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            serialized_matches = data.get("matches", [])
            # Converter fuso se dados antigos (UTC)
            if data.get("_timezone") != "America/Sao_Paulo":
                print("[RECALC] Dados em UTC detectados — convertendo para Brasília...")
                _convert_cached_data_timezone(serialized_matches, [])

    if not serialized_matches:
        raise ValueError("Nenhum dado em cache para recalcular. Execute o pipeline primeiro.")

    print(f"[RECALC] Reconstruindo {len(serialized_matches)} partidas do cache...")

    # ── 2. Deserializar → objetos MatchAnalysis ──
    matches = []
    for d in serialized_matches:
        try:
            match = deserialize_match(d)
            matches.append(match)
        except Exception as e:
            print(f"[RECALC] Erro ao deserializar match {d.get('match_id', '?')}: {e}")

    print(f"[RECALC] {len(matches)} partidas reconstruidas")

    # ── 2b. Re-parsear odds a partir do cache BRUTO da API ──
    # Corrige bugs de parsing anteriores (ex: BTTS 1o Tempo sobrescrevendo BTTS)
    # usando a versão corrigida de _parse_odds_response, SEM nenhuma API call.
    odds_refreshed = 0
    for match in matches:
        if not match.has_real_odds:
            continue
        # Buscar dados brutos de odds no cache local/Supabase (0 API calls)
        raw = _get_cached_response("odds", {"fixture": match.match_id})
        if not raw:
            continue
        response = raw.get("response", [])
        if not response:
            continue
        odds_raw = response[0]
        # Re-parsear com o parser corrigido
        new_odds = _parse_odds_response(odds_raw)
        if new_odds.bookmaker not in ("N/D", "Modelo (Estimado)", ""):
            match.odds = new_odds
            odds_refreshed += 1

    print(f"[RECALC] Odds re-parseadas do cache bruto: {odds_refreshed}/{len(matches)} partidas")

    # ── 2c. Carregar dados H2H do cache (0 API calls) ──
    # Se o usuário já visualizou o detalhe do jogo, o H2H estará em cache.
    # Caso contrário, não faz API call — apenas pula o ajuste H2H.
    h2h_loaded = 0
    for match in matches:
        home_id = match.home_team.team_id
        away_id = match.away_team.team_id
        if not home_id or not away_id:
            continue
        try:
            h2h_raw = _get_cached_response("fixtures/headtohead", {
                "h2h": f"{home_id}-{away_id}", "last": 10, "status": "FT-AET-PEN"
            })
            if not h2h_raw:
                # Tentar ordem inversa
                h2h_raw = _get_cached_response("fixtures/headtohead", {
                    "h2h": f"{away_id}-{home_id}", "last": 10, "status": "FT-AET-PEN"
                })
            if h2h_raw:
                h2h_fixtures = h2h_raw.get("response", [])
                if h2h_fixtures:
                    total_goals = []
                    for fix in h2h_fixtures:
                        g = fix.get("goals", {})
                        gh = g.get("home", 0) or 0
                        ga = g.get("away", 0) or 0
                        total_goals.append(gh + ga)
                    if total_goals:
                        match.h2h_avg_goals = round(sum(total_goals) / len(total_goals), 2)
                        match.h2h_count = len(total_goals)
                        h2h_loaded += 1
        except Exception:
            pass

    if h2h_loaded:
        print(f"[RECALC] H2H carregado do cache para {h2h_loaded} partidas")

    # ── 3. Re-executar MODELOS (inclui novos mercados de finalizações) ──
    print("[RECALC] Executando modelos estatisticos (Dixon-Coles + NB Shots)...")
    matches = run_models_batch(matches)
    matches = apply_context_batch(matches)

    # ── 4. Re-executar SCANNER DE VALOR (inclui novos mercados) ──
    print("[RECALC] Escaneando oportunidades +EV...")
    opportunities = find_all_value(matches)

    # Reordenar por edge (maior primeiro)
    opportunities.sort(key=lambda o: o.edge, reverse=True)

    # ── 4b. Preservar resultados existentes do Supabase ──
    _preserve_existing_results(opportunities)

    elapsed = round(time.time() - start, 2)
    n_leagues = len(set(m.league_name for m in matches))

    print(f"[RECALC] Concluido em {elapsed}s | {len(matches)} jogos | {len(opportunities)} oportunidades")

    # ── 5. Atualizar cache ──
    now = datetime.now(config.BR_TIMEZONE)
    _cache["matches"] = matches
    _cache["opportunities"] = opportunities
    _cache["run_time"] = elapsed
    _cache["last_run_at"] = now.strftime("%d/%m/%Y %H:%M:%S")
    # Manter api_calls_used anterior (não fez nenhuma call nova)

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
        "mode": "Recalculo (sem API calls)",
        "last_run_at": _cache["last_run_at"],
        "api_calls_this_run": 0,
    }

    # Persistir em disco
    _save_cache_to_disk()

    # Persistir no Supabase
    print("[RECALC] Salvando no Supabase...")
    try:
        serialized_opps = [serialize_opportunity(o) for o in opportunities]
        serialized_matches_new = [serialize_match(m) for m in matches]
        supabase_client.save_full_run(_cache["stats"], serialized_opps, serialized_matches_new)
        print("[RECALC] Dados salvos no Supabase com sucesso")
    except Exception as e:
        print(f"[RECALC] Erro ao salvar no Supabase: {e}")

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
        "kelly_bet_pct": o.kelly_bet_pct if o.kelly_bet_pct != "N/A" else "0.00%",
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
        "odds_suspect": getattr(o, 'odds_suspect', False),
        "result_status": getattr(o, 'result_status', 'PENDENTE'),
        "result_score": getattr(o, 'result_score', ''),
        "result_ht_score": getattr(o, 'result_ht_score', ''),
        "result_corners": getattr(o, 'result_corners', ''),
        "result_cards": getattr(o, 'result_cards', ''),
        "result_shots": getattr(o, 'result_shots', ''),
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
        # α/β REAIS usados no cálculo de xG (podem diferir de attack/defense_strength)
        "model_alpha_h": round(getattr(m, 'model_alpha_h', h.attack_strength), 4),
        "model_beta_h": round(getattr(m, 'model_beta_h', h.defense_strength), 4),
        "model_alpha_a": round(getattr(m, 'model_alpha_a', a.attack_strength), 4),
        "model_beta_a": round(getattr(m, 'model_beta_a', a.defense_strength), 4),
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
        "home_shots_total_avg": round(getattr(h, 'shots_total_avg', h.shots_on_target_avg * 2.8), 1),
        "away_shots_total_avg": round(getattr(a, 'shots_total_avg', a.shots_on_target_avg * 2.8), 1),
        "home_shots_on_target_avg": round(h.shots_on_target_avg, 1),
        "away_shots_on_target_avg": round(a.shots_on_target_avg, 1),
        "home_shots_blocked_avg": round(h.shots_blocked_avg, 1),
        "away_shots_blocked_avg": round(a.shots_blocked_avg, 1),
        # Finalizações esperadas (modelo)
        "model_home_shots": round(getattr(m, 'model_home_shots_expected', 0), 1),
        "model_away_shots": round(getattr(m, 'model_away_shots_expected', 0), 1),
        "model_total_shots": round(getattr(m, 'model_total_shots_expected', 0), 1),
        "model_home_sot": round(getattr(m, 'model_home_sot_expected', 0), 1),
        "model_away_sot": round(getattr(m, 'model_away_sot_expected', 0), 1),
        "model_total_sot": round(getattr(m, 'model_total_sot_expected', 0), 1),
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
        # TODOS os mercados de odds da API
        "all_markets": getattr(m.odds, 'all_markets', {}) or {},
        # Probabilidades do modelo para TODOS os mercados
        "model_probs": getattr(m, 'model_probs', {}) or {},
        # Qualidade dos dados
        "data_quality": m.data_quality_score,
        "has_real_odds": m.has_real_odds,
        "has_real_standings": m.has_real_standings,
        "has_real_weather": m.has_real_weather,
        "home_has_real_data": h.has_real_data,
        "away_has_real_data": a.has_real_data,
        "odds_home_away_suspect": getattr(m, 'odds_home_away_suspect', False),
        "league_avg_goals": getattr(m, 'league_avg_goals', 2.7),
        "h2h_avg_goals": getattr(m, 'h2h_avg_goals', None),
        "h2h_count": getattr(m, 'h2h_count', 0),
    }


# ═══════════════════════════════════════════════
# ROTAS
# ═══════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/run", methods=["POST"])
def api_run():
    """Executa o engine e retorna os resultados.
    Aceita JSON com date_from e date_to para datas customizadas."""
    from flask import request
    try:
        data = request.get_json(silent=True) or {}
        date_from = data.get("date_from")
        date_to = data.get("date_to")

        if date_from and date_to:
            analysis_dates = config.build_date_range(date_from, date_to)
        elif date_from:
            analysis_dates = [date_from]
        else:
            analysis_dates = None  # usa default

        run_engine(analysis_dates=analysis_dates)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500

    return jsonify({
        "ok": True,
        "last_run_at": _cache["last_run_at"],
        "api_calls_this_run": _cache["api_calls_used"],
        "analysis_dates": _cache["stats"].get("analysis_dates", []),
    })


@app.route("/api/recalculate", methods=["POST"])
def api_recalculate():
    """
    Recalcula modelos e scanner usando dados JÁ em cache.
    ZERO requisicoes a API — apenas reprocessa com codigo atualizado.
    Ideal para aplicar novos mercados/modelos sem gastar creditos.
    """
    try:
        recalculate_engine()
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500

    return jsonify({
        "ok": True,
        "last_run_at": _cache["last_run_at"],
        "api_calls_this_run": 0,
        "total_opportunities": _cache["stats"]["total_opportunities"],
        "mode": "Recalculo (0 API calls)",
    })


@app.route("/api/status")
def api_status():
    """Retorna status da API (requests usadas/restantes). Consome 1 request."""
    has_data = _cache["matches"] is not None
    # Usar as datas do cache (última análise) ou default
    cached_dates = _cache.get("stats", {}).get("analysis_dates") if _cache.get("stats") else None
    analysis_dates = cached_dates or config.get_default_dates()
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
            "analysis_dates": analysis_dates,
        })
    except Exception as e:
        return jsonify({
            "ok": False,
            "error": str(e),
            "has_data": has_data,
            "last_run_at": _cache.get("last_run_at"),
            "analysis_dates": analysis_dates,
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
    """Busca historico completo de um time (sob demanda)."""
    from flask import request as flask_request
    from data_ingestion import fetch_team_history

    league_id = flask_request.args.get("league_id", None, type=int)
    last = flask_request.args.get("last", 10, type=int)
    last = min(last, 15)

    try:
        history = fetch_team_history(team_id, league_id=league_id, last=last)
        return jsonify(history)
    except Exception as e:
        print(f"[API] Erro ao buscar historico do time {team_id}: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "team_id": team_id, "all_matches": [], "league_matches": []}), 500


@app.route("/api/h2h/<int:team1_id>/<int:team2_id>")
def api_h2h(team1_id, team2_id):
    """Busca confrontos diretos entre dois times."""
    from data_ingestion import fetch_h2h

    try:
        matches = fetch_h2h(team1_id, team2_id, last=10)
        return jsonify(matches)
    except Exception as e:
        print(f"[API] Erro ao buscar H2H {team1_id} vs {team2_id}: {e}")
        return jsonify([]), 500


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


# ═══════════════════════════════════════════════
# CHECK-RESULTS — Buscar resultados de jogos finalizados
# ═══════════════════════════════════════════════

@app.route("/api/check-results", methods=["POST"])
def api_check_results():
    """
    Busca resultados de jogos onde oportunidades foram mapeadas e o jogo já terminou.
    Só verifica jogos cujo horário + 120min < agora (status 'Encerrado').
    Resolve GREEN/RED/VOID automaticamente, salva no Supabase E atualiza o cache local.

    IMPORTANTE:
    - match_time no Supabase está em BRASÍLIA (convertido na ingestão).
    - Não usa get_resolved_match_ids (causava falso-positivo entre runs).
    - Agrupa por match_id direto das oportunidades pendentes elegíveis.
    - Usa skip_cache=True para buscar resultado atualizado da API (não o cache antigo NS).
    """
    from data_ingestion import fetch_finished_fixtures

    try:
        # 1. Buscar oportunidades pendentes com match_date <= hoje
        pending = supabase_client.get_pending_opportunities()
        if not pending:
            return jsonify({"ok": True, "msg": "Nenhuma oportunidade pendente",
                            "checked": 0, "finished": 0, "resolved": 0,
                            "green": 0, "red": 0, "void": 0,
                            "total_pending": 0, "skipped_not_finished": 0})

        # 2. Filtrar APENAS jogos cuja hora prevista + 120min < agora (encerrados)
        #    match_time no Supabase está em BRASÍLIA (convertido durante ingestão)
        now = datetime.now(config.BR_TIMEZONE)
        eligible = []
        skipped_not_finished = 0
        for opp in pending:
            md = opp.get("match_date", "")
            mt = opp.get("match_time", "00:00")
            try:
                dt_parts = md.split("-")
                tm_parts = (mt or "00:00").split(":")
                # match_time já está em horário de Brasília
                match_dt_br = datetime(
                    int(dt_parts[0]), int(dt_parts[1]), int(dt_parts[2]),
                    int(tm_parts[0]), int(tm_parts[1]), 0,
                    tzinfo=config.BR_TIMEZONE,
                )
                elapsed_min = (now - match_dt_br).total_seconds() / 60
                if elapsed_min >= 120:
                    eligible.append(opp)
                else:
                    skipped_not_finished += 1
            except Exception:
                eligible.append(opp)  # Em caso de erro no parse, incluir mesmo assim

        if not eligible:
            return jsonify({"ok": True, "msg": f"Nenhum jogo encerrado (>120min). {skipped_not_finished} ainda em andamento/futuro.",
                            "checked": 0, "finished": 0, "resolved": 0,
                            "green": 0, "red": 0, "void": 0,
                            "total_pending": len(pending), "skipped_not_finished": skipped_not_finished})

        # 3. Agrupar por match_id (evitar buscar o mesmo jogo múltiplas vezes)
        #    NÃO excluir match_ids "já resolvidos" — podem ter oportunidades PENDENTES
        #    de uma run mais recente (recalcular cria novas oportunidades)
        match_ids_to_check = set()
        for opp in eligible:
            mid = opp.get("match_id")
            if mid:
                match_ids_to_check.add(mid)

        if not match_ids_to_check:
            return jsonify({"ok": True, "msg": "Nenhum match_id encontrado.",
                            "checked": 0, "finished": 0, "resolved": 0,
                            "green": 0, "red": 0, "void": 0,
                            "total_pending": len(pending), "skipped_not_finished": skipped_not_finished})

        print(f"[CHECK-RESULTS] {len(eligible)} oportunidades elegíveis | {len(match_ids_to_check)} jogos únicos a verificar | {skipped_not_finished} ignorados (ainda não encerrados)")

        # 4. Buscar resultados via API (1 call por jogo, skip_cache=True para dados frescos)
        results = fetch_finished_fixtures(list(match_ids_to_check))
        finished_count = sum(1 for r in results.values() if r.get("score"))

        if not finished_count:
            print(f"[CHECK-RESULTS] Nenhum jogo FT encontrado. Status retornados: {set(r.get('status','?') for r in results.values())}")

        # 5. Resolver cada oportunidade
        updates = []
        for opp in eligible:
            mid = opp.get("match_id")
            if mid not in results:
                continue
            result = results[mid]
            if not result.get("score"):
                continue  # API retornou mas jogo ainda não terminou

            score = result["score"]
            hg = result["home_goals"]
            ag = result["away_goals"]

            status = _resolve_opportunity(opp, hg, ag, result)
            if status:
                # Montar detalhes do resultado (HT, corners, cards, shots)
                result_detail = {
                    "ht_home": result.get("ht_home"),
                    "ht_away": result.get("ht_away"),
                    "corners_home": result.get("corners_home"),
                    "corners_away": result.get("corners_away"),
                    "cards_home": result.get("cards_home"),
                    "cards_away": result.get("cards_away"),
                    "shots_home": result.get("shots_home"),
                    "shots_away": result.get("shots_away"),
                    "shots_on_home": result.get("shots_on_home"),
                    "shots_on_away": result.get("shots_on_away"),
                }
                # Formatar strings legíveis
                ht_h, ht_a = result.get("ht_home"), result.get("ht_away")
                result_ht_score = f"{ht_h}-{ht_a}" if ht_h is not None and ht_a is not None else ""
                c_h, c_a = result.get("corners_home"), result.get("corners_away")
                result_corners = f"{c_h}-{c_a}" if c_h is not None and c_a is not None else ""
                cd_h, cd_a = result.get("cards_home"), result.get("cards_away")
                result_cards = f"{cd_h}-{cd_a}" if cd_h is not None and cd_a is not None else ""
                sh_h, sh_a = result.get("shots_home"), result.get("shots_away")
                sot_h, sot_a = result.get("shots_on_home"), result.get("shots_on_away")
                shots_parts = []
                if sh_h is not None and sh_a is not None:
                    shots_parts.append(f"{sh_h}-{sh_a}")
                if sot_h is not None and sot_a is not None:
                    shots_parts.append(f"({sot_h}-{sot_a} gol)")
                result_shots = " ".join(shots_parts)

                updates.append({
                    "id": opp["id"],
                    "match_id": mid,
                    "result_status": status,
                    "result_score": score,
                    "result_ht_score": result_ht_score,
                    "result_corners": result_corners,
                    "result_cards": result_cards,
                    "result_shots": result_shots,
                    "result_detail": result_detail,
                    "market_odd": opp.get("market_odd", 0),
                })

        # 6. Salvar resultados no Supabase
        saved = 0
        n_green = sum(1 for u in updates if u["result_status"] == "GREEN")
        n_red = sum(1 for u in updates if u["result_status"] == "RED")
        n_void = sum(1 for u in updates if u["result_status"] == "VOID")

        if updates:
            saved = supabase_client.batch_update_results(updates)
            print(f"[CHECK-RESULTS] {saved} oportunidades atualizadas ({n_green} GREEN, {n_red} RED, {n_void} VOID)")

            # 7. ATUALIZAR O CACHE EM MEMÓRIA para que /api/opportunities retorne dados atualizados
            _update_cache_with_results(updates)

        return jsonify({
            "ok": True,
            "checked": len(match_ids_to_check),
            "finished": finished_count,
            "resolved": saved,
            "green": n_green,
            "red": n_red,
            "void": n_void,
            "total_pending": len(pending),
            "skipped_not_finished": skipped_not_finished,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


def _update_cache_with_results(updates: list[dict]):
    """
    Atualiza o cache em memória com os resultados do check-results.
    Garante que /api/opportunities retorne dados atualizados sem precisar reload do Supabase.
    """
    if not updates:
        return

    # Criar mapa por opp_id (UUID do Supabase) E por match_id
    update_by_id = {u["id"]: u for u in updates}
    update_by_match = {}  # match_id -> [updates]
    for u in updates:
        mid = u.get("match_id")
        if mid is not None:
            update_by_match.setdefault(mid, []).append(u)

    updated_count = 0

    # 1) Atualizar _serialized_opportunities (quando veio do disco/supabase)
    ser_opps = _cache.get("_serialized_opportunities")
    if ser_opps and isinstance(ser_opps, list):
        for opp in ser_opps:
            opp_id = opp.get("id")
            mid = opp.get("match_id")

            # Primeiro: match por UUID do Supabase
            if opp_id and opp_id in update_by_id:
                u = update_by_id[opp_id]
                opp["result_status"] = u["result_status"]
                opp["result_score"] = u["result_score"]
                opp["result_ht_score"] = u.get("result_ht_score", "")
                opp["result_corners"] = u.get("result_corners", "")
                opp["result_cards"] = u.get("result_cards", "")
                opp["result_shots"] = u.get("result_shots", "")
                updated_count += 1
                continue

            # Fallback: match por match_id + market + selection
            if mid in update_by_match:
                opp_market = (opp.get("market") or "").lower()
                opp_sel = (opp.get("selection") or "").lower()
                for u in update_by_match[mid]:
                    opp["result_status"] = u["result_status"]
                    opp["result_score"] = u["result_score"]
                    opp["result_ht_score"] = u.get("result_ht_score", "")
                    opp["result_corners"] = u.get("result_corners", "")
                    opp["result_cards"] = u.get("result_cards", "")
                    opp["result_shots"] = u.get("result_shots", "")
                    updated_count += 1
                    break

    # 2) Atualizar objetos em memória (quando rodou direto do engine)
    opps_in_memory = _cache.get("opportunities")
    if opps_in_memory and isinstance(opps_in_memory, list) and opps_in_memory not in ("FROM_DISK", "FROM_SUPABASE"):
        for opp in opps_in_memory:
            if hasattr(opp, 'match_id') and opp.match_id in update_by_match:
                for u in update_by_match[opp.match_id]:
                    opp.result_status = u["result_status"]
                    opp.result_score = u["result_score"]
                    if hasattr(opp, 'result_ht_score'):
                        opp.result_ht_score = u.get("result_ht_score", "")
                    if hasattr(opp, 'result_corners'):
                        opp.result_corners = u.get("result_corners", "")
                    if hasattr(opp, 'result_cards'):
                        opp.result_cards = u.get("result_cards", "")
                    if hasattr(opp, 'result_shots'):
                        opp.result_shots = u.get("result_shots", "")
                    updated_count += 1
                    break

    print(f"[CHECK-RESULTS] Cache em memória atualizado: {updated_count} oportunidades")

    # Persistir cache atualizado no disco para sobreviver a restarts
    _save_results_to_disk_cache(updates)


def _resolve_opportunity(opp: dict, home_goals: int, away_goals: int, result: dict) -> str:
    """
    Determina se uma oportunidade foi GREEN, RED ou VOID com base no placar final.
    Retorna 'GREEN', 'RED', 'VOID' ou None se não puder determinar.
    """
    market = (opp.get("market") or "").lower()
    selection = (opp.get("selection") or "").lower()
    total_goals = home_goals + away_goals
    ht_hg = result.get("ht_home")
    ht_ag = result.get("ht_away")

    try:
        # ── 1x2 ──
        if market == "1x2" or market == "resultado":
            if "casa" in selection or "home" in selection:
                return "GREEN" if home_goals > away_goals else "RED"
            elif "empate" in selection or "draw" in selection:
                return "GREEN" if home_goals == away_goals else "RED"
            elif "fora" in selection or "away" in selection:
                return "GREEN" if away_goals > home_goals else "RED"

        # ── Dupla Chance ──
        if "dupla chance" in market:
            if "1x" in selection or "casa ou empate" in selection:
                return "GREEN" if home_goals >= away_goals else "RED"
            elif "x2" in selection or "fora ou empate" in selection:
                return "GREEN" if away_goals >= home_goals else "RED"
            elif "12" in selection or "casa ou fora" in selection:
                return "GREEN" if home_goals != away_goals else "RED"

        # ── Gols Over/Under ──
        if ("gols" in market or "goals" in market) and ("o/u" in market or "over" in market or "under" in market):
            import re
            line_match = re.search(r'(\d+\.?\d*)', selection)
            if line_match:
                line = float(line_match.group(1))
                if "over" in selection or "acima" in selection:
                    return "GREEN" if total_goals > line else "RED"
                elif "under" in selection or "abaixo" in selection:
                    return "GREEN" if total_goals < line else "RED"

        # ── BTTS ──
        if "btts" in market or "ambas" in market:
            both_scored = (home_goals > 0 and away_goals > 0)
            if "sim" in selection or "yes" in selection:
                return "GREEN" if both_scored else "RED"
            elif "não" in selection or "no" in selection:
                return "GREEN" if not both_scored else "RED"

        # ── Clean Sheet ──
        if "clean sheet" in market:
            if "casa" in market or "home" in market:
                cs = (away_goals == 0)
                if "sim" in selection or "yes" in selection:
                    return "GREEN" if cs else "RED"
                else:
                    return "GREEN" if not cs else "RED"
            elif "fora" in market or "away" in market:
                cs = (home_goals == 0)
                if "sim" in selection or "yes" in selection:
                    return "GREEN" if cs else "RED"
                else:
                    return "GREEN" if not cs else "RED"

        # ── Vitória sem Sofrer ──
        if "sofrer" in market or "win to nil" in market:
            if "casa" in market or "home" in market:
                wtn = (home_goals > 0 and away_goals == 0)
            else:
                wtn = (away_goals > 0 and home_goals == 0)
            if "sim" in selection or "yes" in selection:
                return "GREEN" if wtn else "RED"
            else:
                return "GREEN" if not wtn else "RED"

        # ── Par/Impar ──
        if "par" in market and "impar" in market:
            is_odd = total_goals % 2 == 1
            if "impar" in selection or "odd" in selection:
                return "GREEN" if is_odd else "RED"
            else:
                return "GREEN" if not is_odd else "RED"

        # ── 1° Tempo (HT) ──
        if "1o tempo" in market or "1° tempo" in market or "ht" in market:
            if ht_hg is not None and ht_ag is not None:
                ht_total = ht_hg + ht_ag
                if "resultado" in market or "winner" in market:
                    if "casa" in selection or "home" in selection:
                        return "GREEN" if ht_hg > ht_ag else "RED"
                    elif "empate" in selection or "draw" in selection:
                        return "GREEN" if ht_hg == ht_ag else "RED"
                    elif "fora" in selection or "away" in selection:
                        return "GREEN" if ht_ag > ht_hg else "RED"
                elif "o/u" in market or "over" in market or "under" in market:
                    import re
                    line_m = re.search(r'(\d+\.?\d*)', selection)
                    if line_m:
                        line = float(line_m.group(1))
                        if "over" in selection or "acima" in selection:
                            return "GREEN" if ht_total > line else "RED"
                        elif "under" in selection or "abaixo" in selection:
                            return "GREEN" if ht_total < line else "RED"

        # ── Gols Casa O/U ──
        if "gols casa" in market or "home goals" in market:
            import re
            line_m = re.search(r'(\d+\.?\d*)', selection)
            if line_m:
                line = float(line_m.group(1))
                if "over" in selection or "acima" in selection:
                    return "GREEN" if home_goals > line else "RED"
                elif "under" in selection or "abaixo" in selection:
                    return "GREEN" if home_goals < line else "RED"

        # ── Gols Fora O/U ──
        if "gols fora" in market or "away goals" in market:
            import re
            line_m = re.search(r'(\d+\.?\d*)', selection)
            if line_m:
                line = float(line_m.group(1))
                if "over" in selection or "acima" in selection:
                    return "GREEN" if away_goals > line else "RED"
                elif "under" in selection or "abaixo" in selection:
                    return "GREEN" if away_goals < line else "RED"

        # ── Genérico: Over/Under com linha numérica ──
        import re
        line_m = re.search(r'(?:over|under|acima|abaixo)\s*(\d+\.?\d*)', selection)
        if line_m:
            line = float(line_m.group(1))
            if "over" in selection or "acima" in selection:
                return "GREEN" if total_goals > line else "RED"
            elif "under" in selection or "abaixo" in selection:
                return "GREEN" if total_goals < line else "RED"

    except Exception as e:
        print(f"[RESOLVE] Erro ao resolver opp {opp.get('id', '?')}: {e}")

    return None  # Não foi possível determinar (ex: escanteios, cartões — precisam de dados extras)


# ═══════════════════════════════════════════════
# DASHBOARD DE PERFORMANCE
# ═══════════════════════════════════════════════

@app.route("/api/dashboard")
def api_dashboard():
    """
    Dashboard completo de performance.
    Retorna dados BRUTOS (resolved + pending_count) para o frontend fazer 
    toda a agregação e filtragem client-side, igual à tela principal.
    """
    try:
        dashboard_data = supabase_client.get_all_opportunities_for_dashboard()
        resolved = dashboard_data.get("resolved", [])
        pending_count = dashboard_data.get("pending_count", 0)

        if not resolved and pending_count == 0:
            return jsonify({"ok": True, "data": {"resolved": [], "pending_count": 0}, "msg": "Nenhuma oportunidade no banco"})

        # Desduplicar resolvidas: manter apenas 1 registro por (match_id, market, selection)
        seen_keys = set()
        unique_resolved = []
        for o in resolved:
            key = (o.get("match_id"), (o.get("market") or "").lower(), (o.get("selection") or "").lower())
            if key not in seen_keys:
                seen_keys.add(key)
                unique_resolved.append(o)
        
        if len(unique_resolved) < len(resolved):
            print(f"[DASHBOARD] Dedup: {len(resolved)} → {len(unique_resolved)} oportunidades únicas")

        # Serializar campos necessários para cada oportunidade
        clean = []
        for o in unique_resolved:
            clean.append({
                "match_id": o.get("match_id"),
                "match_date": o.get("match_date", ""),
                "match_time": o.get("match_time", ""),
                "home_team": o.get("home_team", ""),
                "away_team": o.get("away_team", ""),
                "league_name": o.get("league_name", ""),
                "league_country": o.get("league_country", ""),
                "market": o.get("market", ""),
                "selection": o.get("selection", ""),
                "market_odd": o.get("market_odd", 0),
                "fair_odd": o.get("fair_odd", 0),
                "edge": round((o.get("edge") or 0) * 100, 2),
                "model_prob": round((o.get("model_prob") or 0) * 100, 2),
                "confidence": o.get("confidence", ""),
                "bookmaker": o.get("bookmaker", ""),
                "result_status": o.get("result_status", "PENDENTE"),
                "result_score": o.get("result_score", ""),
                "result_ht_score": o.get("result_ht_score", ""),
                "result_corners": o.get("result_corners", ""),
                "result_cards": o.get("result_cards", ""),
                "result_shots": o.get("result_shots", ""),
            })

        return jsonify({"ok": True, "data": {"resolved": clean, "pending_count": pending_count}})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/run-dates")
def api_run_dates():
    """Retorna histórico de datas já analisadas."""
    runs = supabase_client.get_run_dates_history()
    return jsonify(runs)


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
