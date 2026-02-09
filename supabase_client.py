"""
═══════════════════════════════════════════════════════════════════════
  SUPABASE CLIENT — Persistência na Nuvem para ApostasIA
  Salva execuções, oportunidades e partidas no banco Supabase.
═══════════════════════════════════════════════════════════════════════
"""

import config

# Tentar importar supabase; se não instalado, desativar
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False

_client: "Client | None" = None


def get_client() -> "Client | None":
    """Retorna instância singleton do Supabase client."""
    global _client
    if _client is not None:
        return _client
    if not SUPABASE_AVAILABLE:
        print("[SUPABASE] SDK não instalado (pip install supabase)")
        return None
    if not config.SUPABASE_URL or not config.SUPABASE_SERVICE_KEY:
        print("[SUPABASE] Chaves não configuradas no .env")
        return None
    try:
        _client = create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_KEY)
        print("[SUPABASE] Conectado com sucesso")
        return _client
    except Exception as e:
        print(f"[SUPABASE] Erro ao conectar: {e}")
        return None


def is_configured() -> bool:
    """Verifica se o Supabase está configurado e acessível."""
    return get_client() is not None


def save_pipeline_run(stats: dict) -> str | None:
    """
    Salva uma execução do pipeline no banco.
    Retorna o UUID do run criado, ou None se falhar.
    """
    sb = get_client()
    if not sb:
        return None

    try:
        data = {
            "analysis_dates": stats.get("analysis_dates", []),
            "total_matches": stats.get("total_matches", 0),
            "total_leagues": stats.get("total_leagues", 0),
            "total_opportunities": stats.get("total_opportunities", 0),
            "high_conf": stats.get("high_conf", 0),
            "med_conf": stats.get("med_conf", 0),
            "low_conf": stats.get("low_conf", 0),
            "avg_edge": stats.get("avg_edge", 0),
            "max_edge": stats.get("max_edge", 0),
            "run_time_seconds": stats.get("run_time", 0),
            "api_calls_used": stats.get("api_calls_this_run", 0),
            "mode": stats.get("mode", "API Real"),
        }

        result = sb.table("pipeline_runs").insert(data).execute()
        if result.data:
            run_id = result.data[0]["id"]
            print(f"[SUPABASE] Pipeline run salvo: {run_id}")
            return run_id
        return None
    except Exception as e:
        print(f"[SUPABASE] Erro ao salvar pipeline_run: {e}")
        return None


def save_opportunities(run_id: str, opportunities: list[dict]) -> int:
    """
    Salva todas as oportunidades de uma execução.
    Retorna número de oportunidades salvas.
    """
    sb = get_client()
    if not sb or not run_id or not opportunities:
        return 0

    try:
        rows = []
        for o in opportunities:
            rows.append({
                "run_id": run_id,
                "match_id": o.get("match_id", 0),
                "league_name": o.get("league_name", ""),
                "league_country": o.get("league_country", ""),
                "match_date": o.get("match_date", ""),
                "match_time": o.get("match_time", ""),
                "home_team": o.get("home_team", ""),
                "away_team": o.get("away_team", ""),
                "market": o.get("market", ""),
                "selection": o.get("selection", ""),
                "bookmaker": o.get("bookmaker", "N/D"),
                "market_odd": o.get("market_odd", 0),
                "fair_odd": o.get("fair_odd", 0),
                "model_prob": o.get("model_prob", 0) / 100.0 if o.get("model_prob", 0) > 1 else o.get("model_prob", 0),
                "implied_prob": o.get("implied_prob", 0) / 100.0 if o.get("implied_prob", 0) > 1 else o.get("implied_prob", 0),
                "edge": o.get("edge", 0) / 100.0 if o.get("edge", 0) > 1 else o.get("edge", 0),
                "edge_pct": o.get("edge_pct", ""),
                "kelly_fraction": float(o.get("kelly_bet_pct", "0%").replace("%", "")) / 100.0 if o.get("kelly_bet_pct") else 0,
                "kelly_bet_pct": o.get("kelly_bet_pct", ""),
                "confidence": o.get("confidence", ""),
                "reasoning": o.get("reasoning", ""),
                "home_xg": o.get("home_xg", 0),
                "away_xg": o.get("away_xg", 0),
                "weather_note": o.get("weather_note", ""),
                "fatigue_note": o.get("fatigue_note", ""),
                "urgency_home": o.get("urgency_home", 0.5),
                "urgency_away": o.get("urgency_away", 0.5),
            })

        # Inserir em lotes de 100 (limite do Supabase)
        saved = 0
        batch_size = 100
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            result = sb.table("opportunities").insert(batch).execute()
            if result.data:
                saved += len(result.data)

        print(f"[SUPABASE] {saved} oportunidades salvas")
        return saved
    except Exception as e:
        print(f"[SUPABASE] Erro ao salvar oportunidades: {e}")
        return 0


def save_matches(run_id: str, matches: list[dict]) -> int:
    """
    Salva todas as partidas de uma execução.
    Retorna número de partidas salvas.
    """
    sb = get_client()
    if not sb or not run_id or not matches:
        return 0

    try:
        rows = []
        for m in matches:
            rows.append({
                "run_id": run_id,
                "match_id": m.get("match_id", 0),
                "league_name": m.get("league_name", ""),
                "league_country": m.get("league_country", ""),
                "match_date": m.get("match_date", ""),
                "match_time": m.get("match_time", ""),
                "home_team": m.get("home_team", ""),
                "away_team": m.get("away_team", ""),
                "home_xg": m.get("home_xg", 0),
                "away_xg": m.get("away_xg", 0),
                "prob_home": m.get("prob_home", 0),
                "prob_draw": m.get("prob_draw", 0),
                "prob_away": m.get("prob_away", 0),
                "prob_over25": m.get("prob_over25", 0),
                "prob_btts": m.get("prob_btts", 0),
                "corners_expected": m.get("corners_expected", 0),
                "cards_expected": m.get("cards_expected", 0),
                "odds_home": m.get("odds_home", 0),
                "odds_draw": m.get("odds_draw", 0),
                "odds_away": m.get("odds_away", 0),
                "bookmaker": m.get("bookmaker", "N/D"),
                "weather_temp": m.get("weather_temp", 0),
                "weather_wind": m.get("weather_wind", 0),
                "weather_desc": m.get("weather_desc", ""),
                "venue": m.get("venue", ""),
                "referee": m.get("referee", ""),
            })

        saved = 0
        batch_size = 100
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            result = sb.table("matches").insert(batch).execute()
            if result.data:
                saved += len(result.data)

        print(f"[SUPABASE] {saved} partidas salvas")
        return saved
    except Exception as e:
        print(f"[SUPABASE] Erro ao salvar partidas: {e}")
        return 0


def save_full_run(stats: dict, opportunities: list[dict], matches: list[dict]):
    """
    Salva uma execução completa: run + oportunidades + partidas.
    Chamado automaticamente após cada pipeline.
    """
    if not is_configured():
        print("[SUPABASE] Não configurado — dados salvos apenas localmente")
        return

    print("[SUPABASE] Salvando execução no banco de dados...")
    run_id = save_pipeline_run(stats)
    if not run_id:
        print("[SUPABASE] Falha ao criar pipeline_run — abortando")
        return

    save_opportunities(run_id, opportunities)
    save_matches(run_id, matches)
    print(f"[SUPABASE] Execução salva com sucesso (run_id: {run_id})")


def get_run_history(limit: int = 10) -> list[dict]:
    """Retorna as últimas N execuções do pipeline."""
    sb = get_client()
    if not sb:
        return []
    try:
        result = (
            sb.table("pipeline_runs")
            .select("*")
            .order("executed_at", desc=True)
            .limit(limit)
            .execute()
        )
        return result.data or []
    except Exception as e:
        print(f"[SUPABASE] Erro ao buscar histórico: {e}")
        return []


def get_opportunities_by_run(run_id: str) -> list[dict]:
    """Retorna oportunidades de uma execução específica."""
    sb = get_client()
    if not sb:
        return []
    try:
        result = (
            sb.table("opportunities")
            .select("*")
            .eq("run_id", run_id)
            .order("edge", desc=True)
            .execute()
        )
        return result.data or []
    except Exception as e:
        print(f"[SUPABASE] Erro: {e}")
        return []


def get_matches_by_run(run_id: str) -> list[dict]:
    """Retorna partidas de uma execução específica."""
    sb = get_client()
    if not sb:
        return []
    try:
        result = (
            sb.table("matches")
            .select("*")
            .eq("run_id", run_id)
            .execute()
        )
        return result.data or []
    except Exception as e:
        print(f"[SUPABASE] Erro ao buscar partidas: {e}")
        return []


def update_opportunity_result(opp_id: str, status: str, score: str = None) -> bool:
    """
    Atualiza o resultado de uma oportunidade.
    status: 'GREEN', 'RED', 'VOID', 'POSTPONED'
    """
    sb = get_client()
    if not sb:
        return False
    try:
        from datetime import datetime
        data = {
            "result_status": status,
            "result_updated_at": datetime.now().isoformat(),
        }
        if score:
            data["result_score"] = score

        result = sb.table("opportunities").update(data).eq("id", opp_id).execute()
        return bool(result.data)
    except Exception as e:
        print(f"[SUPABASE] Erro ao atualizar resultado: {e}")
        return False


def update_bet_info(opp_id: str, amount: float, bet_return: float = None, notes: str = None) -> bool:
    """
    Registra informações de aposta feita pelo usuário.
    """
    sb = get_client()
    if not sb:
        return False
    try:
        data = {
            "bet_placed": True,
            "bet_amount": amount,
        }
        if bet_return is not None:
            data["bet_return"] = bet_return
            data["bet_profit"] = bet_return - amount
        if notes:
            data["bet_notes"] = notes

        result = sb.table("opportunities").update(data).eq("id", opp_id).execute()
        return bool(result.data)
    except Exception as e:
        print(f"[SUPABASE] Erro ao registrar aposta: {e}")
        return False
