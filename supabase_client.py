"""
═══════════════════════════════════════════════════════════════════════
  SUPABASE CLIENT — Persistencia na Nuvem para ApostasIA
  Salva execucoes, oportunidades, partidas E respostas brutas da API.
  Garante que NENHUM dado consultado seja perdido.
═══════════════════════════════════════════════════════════════════════
"""

import json
import hashlib
from datetime import datetime, timedelta

import config

# Tentar importar supabase; se nao instalado, desativar
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False

_client: "Client | None" = None
_client_checked = False  # Evitar logs duplicados


def get_client() -> "Client | None":
    """Retorna instancia singleton do Supabase client."""
    global _client, _client_checked
    if _client is not None:
        return _client
    if _client_checked:
        return None  # Ja tentou e falhou, nao logar de novo
    _client_checked = True
    if not SUPABASE_AVAILABLE:
        print("[SUPABASE] SDK nao instalado (pip install supabase)")
        return None
    if not config.SUPABASE_URL or not config.SUPABASE_SERVICE_KEY:
        print("[SUPABASE] Chaves nao configuradas no .env")
        return None
    try:
        _client = create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_KEY)
        print("[SUPABASE] Conectado com sucesso")
        return _client
    except Exception as e:
        print(f"[SUPABASE] Erro ao conectar: {e}")
        return None


def is_configured() -> bool:
    """Verifica se o Supabase esta configurado e acessivel."""
    return get_client() is not None


# ═══════════════════════════════════════════════════════════════
#  CACHE DE RESPOSTAS BRUTAS DA API — Persistencia permanente
#  Tabela: api_responses
#  Garante que TODA resposta da API seja salva na nuvem
# ═══════════════════════════════════════════════════════════════

# TTL em horas por tipo de endpoint (para decidir se busca do cache)
_CACHE_TTL_HOURS = {
    "fixtures":             3,     # fixtures mudam com frequencia
    "standings":           12,     # standings mudam 1x por dia
    "odds":                 4,     # odds mudam moderadamente
    "injuries":             6,     # lesoes mudam moderadamente
    "fixtures/lineups":   720,     # dados historicos, nao mudam (30 dias)
    "fixtures/statistics": 720,    # dados historicos, nao mudam (30 dias)
    "weather":             3,      # clima muda rapido
    "status":              1,      # status da conta
}


def _make_cache_key(endpoint: str, params: dict) -> str:
    """Gera chave unica para cache: endpoint + hash dos params."""
    params_str = json.dumps(params, sort_keys=True)
    h = hashlib.md5(f"{endpoint}_{params_str}".encode()).hexdigest()[:16]
    return f"{endpoint.replace('/', '_')}_{h}"


def save_api_response(endpoint: str, params: dict, response_data: dict) -> bool:
    """
    Salva resposta bruta da API no Supabase (tabela api_responses).
    Usa upsert para atualizar se a mesma chave ja existir.
    """
    sb = get_client()
    if not sb:
        return False

    cache_key = _make_cache_key(endpoint, params)
    ttl = _CACHE_TTL_HOURS.get(endpoint, 4)

    try:
        row = {
            "cache_key": cache_key,
            "endpoint": endpoint,
            "params": params,
            "response_data": response_data,
            "ttl_hours": ttl,
            "fetched_at": datetime.now().isoformat(),
        }
        sb.table("api_responses").upsert(row, on_conflict="cache_key").execute()
        return True
    except Exception as e:
        # Nao logar erro para cada request (seria muito verbose)
        # Apenas na primeira vez
        if not hasattr(save_api_response, '_error_logged'):
            print(f"[SUPABASE] Erro ao salvar api_response ({endpoint}): {e}")
            print(f"[SUPABASE] DICA: Crie a tabela 'api_responses' no Supabase. Veja instrucoes no console.")
            _print_create_table_sql()
            save_api_response._error_logged = True
        return False


def get_api_response(endpoint: str, params: dict) -> dict | None:
    """
    Busca resposta cacheada no Supabase.
    Retorna None se nao encontrada ou se expirada pelo TTL.
    """
    sb = get_client()
    if not sb:
        return None

    cache_key = _make_cache_key(endpoint, params)

    try:
        result = (
            sb.table("api_responses")
            .select("response_data, fetched_at, ttl_hours")
            .eq("cache_key", cache_key)
            .limit(1)
            .execute()
        )
        if not result.data:
            return None

        row = result.data[0]
        fetched_at = datetime.fromisoformat(row["fetched_at"].replace("Z", "+00:00").replace("+00:00", ""))
        ttl_hours = row.get("ttl_hours", 4)
        age_hours = (datetime.now() - fetched_at).total_seconds() / 3600

        if age_hours < ttl_hours:
            return row["response_data"]
        else:
            return None  # Expirado
    except Exception:
        return None


def get_api_response_ignore_ttl(endpoint: str, params: dict) -> dict | None:
    """
    Busca resposta cacheada no Supabase IGNORANDO TTL.
    Util para dados historicos que queremos sempre ter disponivel.
    """
    sb = get_client()
    if not sb:
        return None

    cache_key = _make_cache_key(endpoint, params)

    try:
        result = (
            sb.table("api_responses")
            .select("response_data, fetched_at")
            .eq("cache_key", cache_key)
            .limit(1)
            .execute()
        )
        if not result.data:
            return None
        return result.data[0]["response_data"]
    except Exception:
        return None


def get_api_cache_stats_supabase() -> dict:
    """Retorna estatisticas do cache no Supabase."""
    sb = get_client()
    if not sb:
        return {"total": 0, "endpoints": {}}

    try:
        result = sb.table("api_responses").select("endpoint", count="exact").execute()
        total = result.count if hasattr(result, 'count') else len(result.data or [])

        # Contar por endpoint
        endpoints = {}
        if result.data:
            for row in result.data:
                ep = row.get("endpoint", "unknown")
                endpoints[ep] = endpoints.get(ep, 0) + 1

        return {"total": total, "endpoints": endpoints}
    except Exception:
        return {"total": 0, "endpoints": {}}


def _print_create_table_sql():
    """Imprime o SQL para criar a tabela api_responses no Supabase."""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║  CRIAR TABELA api_responses NO SUPABASE                      ║
║  Execute este SQL no SQL Editor do Supabase Dashboard:        ║
╠═══════════════════════════════════════════════════════════════╣

CREATE TABLE IF NOT EXISTS api_responses (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    cache_key TEXT UNIQUE NOT NULL,
    endpoint TEXT NOT NULL,
    params JSONB NOT NULL DEFAULT '{}'::jsonb,
    response_data JSONB NOT NULL DEFAULT '{}'::jsonb,
    ttl_hours INTEGER NOT NULL DEFAULT 4,
    fetched_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Indice para buscas rapidas por cache_key
CREATE INDEX IF NOT EXISTS idx_api_responses_cache_key
    ON api_responses (cache_key);

-- Indice para buscas por endpoint
CREATE INDEX IF NOT EXISTS idx_api_responses_endpoint
    ON api_responses (endpoint);

-- Indice para limpeza por data
CREATE INDEX IF NOT EXISTS idx_api_responses_fetched_at
    ON api_responses (fetched_at);

-- Habilitar RLS (Row Level Security) - necessario no Supabase
ALTER TABLE api_responses ENABLE ROW LEVEL SECURITY;

-- Politica para permitir acesso com service_key
CREATE POLICY "Allow all for service role"
    ON api_responses
    FOR ALL
    USING (true)
    WITH CHECK (true);

╚═══════════════════════════════════════════════════════════════╝
""")


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


# ═══════════════════════════════════════════════════════════════
#  CONSULTAS DE RESULTADOS — Jogos terminados
# ═══════════════════════════════════════════════════════════════

def get_pending_opportunities() -> list[dict]:
    """
    Retorna oportunidades com result_status='PENDENTE' cujos jogos
    já devem ter terminado (match_date <= hoje).
    Inclui match_id, selection, market para resolver o resultado.
    """
    sb = get_client()
    if not sb:
        return []
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        result = (
            sb.table("opportunities")
            .select("id, match_id, market, selection, match_date, match_time, home_team, away_team, market_odd, model_prob, edge, confidence, league_name, league_country, bookmaker")
            .eq("result_status", "PENDENTE")
            .lte("match_date", today)
            .order("match_date", desc=True)
            .limit(500)
            .execute()
        )
        return result.data or []
    except Exception as e:
        print(f"[SUPABASE] Erro ao buscar oportunidades pendentes: {e}")
        return []


def get_resolved_match_ids() -> set:
    """
    Retorna set de match_ids que JÁ tiveram resultado resolvido.
    Evita buscar dados de jogos que já foram processados.
    """
    sb = get_client()
    if not sb:
        return set()
    try:
        result = (
            sb.table("opportunities")
            .select("match_id")
            .neq("result_status", "PENDENTE")
            .execute()
        )
        return {r["match_id"] for r in (result.data or [])}
    except Exception:
        return set()


def batch_update_results(updates: list[dict]) -> int:
    """
    Atualiza resultado de múltiplas oportunidades de uma vez.
    updates: [{"id": uuid, "result_status": "GREEN"|"RED"|"VOID", "result_score": "2-1"}, ...]
    Retorna número de atualizações bem sucedidas.
    """
    sb = get_client()
    if not sb:
        return 0
    count = 0
    now = datetime.now().isoformat()
    for u in updates:
        try:
            data = {
                "result_status": u["result_status"],
                "result_score": u.get("result_score", ""),
                "result_updated_at": now,
            }
            # Calcular retorno assumindo 1 unidade apostada
            if u["result_status"] == "GREEN":
                data["bet_amount"] = 1.0
                data["bet_return"] = u.get("market_odd", 0)
                data["bet_profit"] = data["bet_return"] - 1.0
            elif u["result_status"] == "RED":
                data["bet_amount"] = 1.0
                data["bet_return"] = 0.0
                data["bet_profit"] = -1.0
            elif u["result_status"] == "VOID":
                data["bet_amount"] = 1.0
                data["bet_return"] = 1.0
                data["bet_profit"] = 0.0

            result = sb.table("opportunities").update(data).eq("id", u["id"]).execute()
            if result.data:
                count += 1
        except Exception as e:
            print(f"[SUPABASE] Erro ao atualizar {u.get('id', '?')}: {e}")
    return count


# ═══════════════════════════════════════════════════════════════
#  DASHBOARD — Consultas de performance completas
# ═══════════════════════════════════════════════════════════════

def get_all_resolved_opportunities() -> list[dict]:
    """
    Retorna TODAS as oportunidades já resolvidas (GREEN/RED/VOID),
    incluindo todos os campos necessários para o dashboard.
    """
    sb = get_client()
    if not sb:
        return []
    try:
        result = (
            sb.table("opportunities")
            .select("*")
            .in_("result_status", ["GREEN", "RED", "VOID"])
            .order("match_date", desc=True)
            .limit(5000)
            .execute()
        )
        return result.data or []
    except Exception as e:
        print(f"[SUPABASE] Erro ao buscar oportunidades resolvidas: {e}")
        return []


def get_run_dates_history() -> list[dict]:
    """
    Retorna todas as datas de análise já executadas, com stats resumidos.
    Permite ao frontend mostrar quais datas já foram rodadas.
    """
    sb = get_client()
    if not sb:
        return []
    try:
        result = (
            sb.table("pipeline_runs")
            .select("id, executed_at, analysis_dates, total_matches, total_opportunities, mode")
            .order("executed_at", desc=True)
            .limit(50)
            .execute()
        )
        return result.data or []
    except Exception as e:
        print(f"[SUPABASE] Erro ao buscar datas de execução: {e}")
        return []


def get_all_opportunities_for_dashboard(limit: int = 10000) -> list[dict]:
    """
    Retorna TODAS as oportunidades (PENDENTE + GREEN + RED + VOID)
    para o dashboard completo.
    """
    sb = get_client()
    if not sb:
        return []
    try:
        result = (
            sb.table("opportunities")
            .select("id, match_id, market, selection, match_date, match_time, "
                     "home_team, away_team, league_name, league_country, bookmaker, "
                     "market_odd, fair_odd, model_prob, implied_prob, edge, "
                     "kelly_fraction, confidence, result_status, result_score, "
                     "bet_amount, bet_return, bet_profit")
            .order("match_date", desc=True)
            .limit(limit)
            .execute()
        )
        return result.data or []
    except Exception as e:
        print(f"[SUPABASE] Erro ao buscar dashboard: {e}")
        return []
