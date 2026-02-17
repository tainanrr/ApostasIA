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
                "confidence_score": o.get("confidence_score", 0.0),
                "analysis_type": o.get("analysis_type", "PRE_JOGO"),
            })

        # Inserir em lotes de 100 (limite do Supabase)
        saved = 0
        batch_size = 100
        _new_columns = {"confidence_score", "analysis_type"}
        _retry_without_new = False

        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            if _retry_without_new:
                batch = [{k: v for k, v in r.items() if k not in _new_columns} for r in batch]
            try:
                result = sb.table("opportunities").insert(batch).execute()
                if result.data:
                    saved += len(result.data)
            except Exception as batch_err:
                err_msg = str(batch_err)
                if any(col in err_msg for col in _new_columns) and not _retry_without_new:
                    print(f"[SUPABASE] ⚠️  Colunas novas ausentes, tentando sem elas...")
                    _retry_without_new = True
                    batch = [{k: v for k, v in r.items() if k not in _new_columns} for r in batch]
                    try:
                        result = sb.table("opportunities").insert(batch).execute()
                        if result.data:
                            saved += len(result.data)
                    except Exception as retry_err:
                        print(f"[SUPABASE] Erro ao salvar lote (retry): {retry_err}")
                else:
                    print(f"[SUPABASE] Erro ao salvar lote: {batch_err}")

        print(f"[SUPABASE] {saved} oportunidades salvas")
        return saved
    except Exception as e:
        print(f"[SUPABASE] Erro ao salvar oportunidades: {e}")
        return 0


def save_matches(run_id: str, matches: list[dict]) -> int:
    """
    Salva todas as partidas de uma execução.
    Retorna número de partidas salvas.
    Salva TODOS os campos detalhados para permitir visualização completa ao carregar do Supabase.
    """
    sb = get_client()
    if not sb or not run_id or not matches:
        return 0

    try:
        rows = []
        for m in matches:
            row = {
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
            }
            # ── Campos detalhados extras (salvos se existirem no schema) ──
            _extra_fields = {
                "league_id": m.get("league_id", 0),
                "home_team_id": m.get("home_team_id", 0),
                "away_team_id": m.get("away_team_id", 0),
                "home_fatigue": float(m.get("home_fatigue") or 0),
                "away_fatigue": float(m.get("away_fatigue") or 0),
                "urgency_home": m.get("urgency_home", 0.5),
                "urgency_away": m.get("urgency_away", 0.5),
                "injuries_home": m.get("injuries_home"),
                "injuries_away": m.get("injuries_away"),
                "referee_cards_avg": m.get("referee_cards_avg", 0),
                "referee_fouls_avg": m.get("referee_fouls_avg", 0),
                "home_form": m.get("home_form"),
                "away_form": m.get("away_form"),
                "home_attack": m.get("home_attack", 0),
                "home_defense": m.get("home_defense", 0),
                "away_attack": m.get("away_attack", 0),
                "away_defense": m.get("away_defense", 0),
                "model_alpha_h": m.get("model_alpha_h", 0),
                "model_beta_h": m.get("model_beta_h", 0),
                "model_alpha_a": m.get("model_alpha_a", 0),
                "model_beta_a": m.get("model_beta_a", 0),
                "home_goals_scored_avg": m.get("home_goals_scored_avg", 0),
                "home_goals_conceded_avg": m.get("home_goals_conceded_avg", 0),
                "away_goals_scored_avg": m.get("away_goals_scored_avg", 0),
                "away_goals_conceded_avg": m.get("away_goals_conceded_avg", 0),
                "home_form_points": m.get("home_form_points", 0),
                "away_form_points": m.get("away_form_points", 0),
                "home_league_pos": m.get("home_league_pos", 0),
                "away_league_pos": m.get("away_league_pos", 0),
                "home_league_pts": m.get("home_league_pts", 0),
                "away_league_pts": m.get("away_league_pts", 0),
                "home_games_played": m.get("home_games_played", 0),
                "away_games_played": m.get("away_games_played", 0),
                "home_games_remaining": m.get("home_games_remaining", 0),
                "away_games_remaining": m.get("away_games_remaining", 0),
                "home_points_to_title": m.get("home_points_to_title", 99),
                "away_points_to_title": m.get("away_points_to_title", 99),
                "home_points_to_relegation": m.get("home_points_to_relegation", 99),
                "away_points_to_relegation": m.get("away_points_to_relegation", 99),
                "home_shots_total_avg": m.get("home_shots_total_avg", 0),
                "away_shots_total_avg": m.get("away_shots_total_avg", 0),
                "home_shots_on_target_avg": m.get("home_shots_on_target_avg", 0),
                "away_shots_on_target_avg": m.get("away_shots_on_target_avg", 0),
                "home_shots_blocked_avg": m.get("home_shots_blocked_avg", 0),
                "away_shots_blocked_avg": m.get("away_shots_blocked_avg", 0),
                "home_corners_avg": m.get("home_corners_avg", 0),
                "away_corners_avg": m.get("away_corners_avg", 0),
                "home_cards_avg": m.get("home_cards_avg", 0),
                "away_cards_avg": m.get("away_cards_avg", 0),
                "home_fouls_avg": m.get("home_fouls_avg", 0),
                "away_fouls_avg": m.get("away_fouls_avg", 0),
                "home_possession": m.get("home_possession", 0),
                "away_possession": m.get("away_possession", 0),
                "weather_rain": m.get("weather_rain", 0),
                "data_quality": m.get("data_quality", 0),
                "has_real_odds": 1 if m.get("has_real_odds") else 0,
                "has_real_standings": 1 if m.get("has_real_standings") else 0,
                "has_real_weather": 1 if m.get("has_real_weather") else 0,
                "home_has_real_data": 1 if m.get("home_has_real_data") else 0,
                "away_has_real_data": 1 if m.get("away_has_real_data") else 0,
                "odds_home_away_suspect": 1 if m.get("odds_home_away_suspect") else 0,
                "league_avg_goals": m.get("league_avg_goals", 2.7),
                "model_home_shots": m.get("model_home_shots", 0),
                "model_away_shots": m.get("model_away_shots", 0),
                "model_total_shots": m.get("model_total_shots", 0),
                "model_home_sot": m.get("model_home_sot", 0),
                "model_away_sot": m.get("model_away_sot", 0),
                "model_total_sot": m.get("model_total_sot", 0),
                "odds_over25": m.get("odds_over25", 0),
                "odds_under25": m.get("odds_under25", 0),
                "odds_btts_yes": m.get("odds_btts_yes", 0),
                "odds_btts_no": m.get("odds_btts_no", 0),
                "odds_corners_over": m.get("odds_corners_over", 0),
                "odds_corners_under": m.get("odds_corners_under", 0),
                "odds_cards_over": m.get("odds_cards_over", 0),
                "odds_cards_under": m.get("odds_cards_under", 0),
                "odds_ah_line": m.get("odds_ah_line", 0),
                "odds_ah_home": m.get("odds_ah_home", 0),
                "odds_ah_away": m.get("odds_ah_away", 0),
                "odds_1x": m.get("odds_1x", 0),
                "odds_x2": m.get("odds_x2", 0),
            }
            # Só adicionar campos extras que não sejam None (para compatibilidade com schemas antigos)
            # Converter booleans → int para colunas numeric no Supabase
            for k, v in _extra_fields.items():
                if v is not None:
                    row[k] = int(v) if isinstance(v, bool) else v
            
            # JSONB fields - serializar
            import json
            if m.get("all_markets"):
                row["all_markets"] = json.dumps(m["all_markets"])
            if m.get("model_probs"):
                row["model_probs"] = json.dumps(m["model_probs"])
            
            rows.append(row)

        # Campos básicos que SEMPRE existem no schema
        _basic_keys = {
            "run_id", "match_id", "league_name", "league_country", "match_date",
            "match_time", "home_team", "away_team", "home_xg", "away_xg",
            "prob_home", "prob_draw", "prob_away", "prob_over25", "prob_btts",
            "corners_expected", "cards_expected", "odds_home", "odds_draw", "odds_away",
            "bookmaker", "weather_temp", "weather_wind", "weather_desc", "venue", "referee",
        }

        saved = 0
        batch_size = 100
        use_full_schema = True  # Tentar com todos os campos primeiro
        
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            
            if not use_full_schema:
                # Fallback: usar apenas campos básicos
                batch = [{k: v for k, v in row.items() if k in _basic_keys} for row in batch]
            
            try:
                result = sb.table("matches").insert(batch).execute()
                if result.data:
                    saved += len(result.data)
            except Exception as batch_err:
                err_msg = str(batch_err).lower()
                if "column" in err_msg and ("does not exist" in err_msg or "not found" in err_msg):
                    if use_full_schema:
                        print(f"[SUPABASE] ⚠️  Schema matches incompleto — salvando apenas campos básicos")
                        print(f"[SUPABASE]    Execute o SQL de migração para habilitar todos os campos detalhados")
                        use_full_schema = False
                        # Retry este batch com campos básicos
                        basic_batch = [{k: v for k, v in row.items() if k in _basic_keys} for row in batch]
                        try:
                            result = sb.table("matches").insert(basic_batch).execute()
                            if result.data:
                                saved += len(result.data)
                        except Exception as retry_err:
                            print(f"[SUPABASE] Erro ao salvar batch (retry básico): {retry_err}")
                else:
                    print(f"[SUPABASE] Erro ao salvar batch de partidas: {batch_err}")

        print(f"[SUPABASE] {saved} partidas salvas" + (" (campos básicos apenas)" if not use_full_schema else " (completo)"))
        return saved
    except Exception as e:
        print(f"[SUPABASE] Erro ao salvar partidas: {e}")
        return 0


def save_full_run(stats: dict, opportunities: list[dict], matches: list[dict]):
    """
    Salva uma execução completa: run + oportunidades + partidas.
    Estratégia UPSERT (preserva dados anteriores):
      - Oportunidades com mesmo (match_id, market, selection): SOBRESCRITAS pela nova run
      - Oportunidades novas (sem equivalente anterior): INSERIDAS
      - Oportunidades anteriores sem equivalente na nova run: PRESERVADAS
      - Oportunidades já resolvidas (GREEN/RED/VOID): NUNCA tocadas
    """
    if not is_configured():
        print("[SUPABASE] Não configurado — dados salvos apenas localmente")
        return

    print("[SUPABASE] Salvando execução no banco de dados...")

    run_id = save_pipeline_run(stats)
    if not run_id:
        print("[SUPABASE] Falha ao criar pipeline_run — abortando")
        return

    _upsert_opportunities(run_id, opportunities)
    save_matches(run_id, matches)
    _cleanup_orphan_runs(run_id)
    print(f"[SUPABASE] Execução salva com sucesso (run_id: {run_id})")


def _upsert_opportunities(run_id: str, new_opportunities: list[dict]):
    """
    Estratégia de UPSERT inteligente para oportunidades:
      1. Busca PENDENTES existentes no mesmo range de datas
      2. Se já existe PENDENTE com mesmo (match_id, market, selection): DELETA a antiga
      3. Insere TODAS as novas oportunidades
      4. Oportunidades anteriores sem equivalente: PRESERVADAS (nunca tocadas)
      5. Oportunidades já resolvidas (GREEN/RED/VOID): NUNCA tocadas
    """
    sb = get_client()
    if not sb or not run_id or not new_opportunities:
        return

    try:
        # 1. Determinar range de datas da run atual
        analysis_dates = set()
        for o in new_opportunities:
            d = o.get("match_date", "")
            if d:
                analysis_dates.add(d)

        if not analysis_dates:
            print("[SUPABASE] Sem datas nas oportunidades — inserindo tudo como novo")
            save_opportunities(run_id, new_opportunities)
            return

        date_from = min(analysis_dates)
        date_to = max(analysis_dates)

        # 2. Buscar PENDENTES existentes no range de datas
        existing_pendentes = []
        page_size = 1000
        offset = 0
        while True:
            result = (
                sb.table("opportunities")
                .select("id, match_id, market, selection")
                .eq("result_status", "PENDENTE")
                .gte("match_date", date_from)
                .lte("match_date", date_to)
                .range(offset, offset + page_size - 1)
                .execute()
            )
            batch = result.data or []
            existing_pendentes.extend(batch)
            if len(batch) < page_size:
                break
            offset += page_size

        print(f"[SUPABASE] {len(existing_pendentes)} pendentes existentes no range {date_from}→{date_to}")

        # 3. Mapear existentes por (match_id, market_lower, selection_lower)
        #    Pode haver múltiplas (de runs diferentes) → acumular IDs
        existing_map = {}
        for e in existing_pendentes:
            key = (e.get("match_id"), (e.get("market") or "").lower(), (e.get("selection") or "").lower())
            if key not in existing_map:
                existing_map[key] = []
            existing_map[key].append(e["id"])

        # 4. Identificar quais pendentes existentes serão substituídas pelas novas
        ids_to_delete = []
        for o in new_opportunities:
            key = (o.get("match_id"), (o.get("market") or "").lower(), (o.get("selection") or "").lower())
            if key in existing_map:
                ids_to_delete.extend(existing_map[key])
                del existing_map[key]  # Consumido — não deletar de novo

        # 5. Deletar duplicatas em lotes
        if ids_to_delete:
            batch_size = 100
            total_deleted = 0
            for i in range(0, len(ids_to_delete), batch_size):
                batch_ids = ids_to_delete[i:i + batch_size]
                try:
                    result = sb.table("opportunities").delete().in_("id", batch_ids).execute()
                    total_deleted += len(result.data) if result.data else 0
                except Exception as e:
                    print(f"[SUPABASE] Erro ao deletar lote de duplicatas: {e}")
            print(f"[SUPABASE] {total_deleted} pendentes substituídas (mesma partida/mercado/seleção)")

        # 6. Inserir TODAS as novas oportunidades
        save_opportunities(run_id, new_opportunities)

        # 7. Log de preservadas (existentes que não tiveram equivalente na nova run)
        preserved_count = sum(len(ids) for ids in existing_map.values())
        if preserved_count > 0:
            print(f"[SUPABASE] {preserved_count} oportunidades anteriores preservadas (sem equivalente na nova run)")

    except Exception as e:
        print(f"[SUPABASE] Erro no upsert: {e}")
        import traceback
        traceback.print_exc()
        print("[SUPABASE] Fallback: inserindo todas como novas (sem dedup)")
        save_opportunities(run_id, new_opportunities)


def _cleanup_orphan_runs(current_run_id: str):
    """
    Remove pipeline_runs que ficaram sem nenhuma oportunidade (exceto a run atual).
    Também remove matches órfãos dessas runs.
    """
    sb = get_client()
    if not sb:
        return
    try:
        all_runs = (
            sb.table("pipeline_runs")
            .select("id")
            .neq("id", current_run_id)
            .execute()
        )

        cleaned = 0
        for run in (all_runs.data or []):
            r = sb.table("opportunities").select("id", count="exact").eq("run_id", run["id"]).execute()
            if (r.count or 0) == 0:
                sb.table("matches").delete().eq("run_id", run["id"]).execute()
                sb.table("pipeline_runs").delete().eq("id", run["id"]).execute()
                cleaned += 1

        if cleaned > 0:
            print(f"[SUPABASE] Limpeza: {cleaned} runs órfãs removidas")
    except Exception as e:
        print(f"[SUPABASE] Aviso na limpeza de runs órfãs: {e}")


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
        import pytz
        br_tz = pytz.timezone("America/Sao_Paulo")
        today = datetime.now(br_tz).strftime("%Y-%m-%d")
        result = (
            sb.table("opportunities")
            .select("id, match_id, market, selection, match_date, match_time, home_team, away_team, market_odd, model_prob, edge, confidence, league_name, league_country, bookmaker")
            .eq("result_status", "PENDENTE")
            .lte("match_date", today)
            .order("match_date", desc=True)
            .limit(2000)
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
    updates: [{"id": uuid, "result_status": "GREEN"|"RED"|"VOID", "result_score": "2-1",
               "result_detail": {...}}, ...]
    Retorna número de atualizações bem sucedidas.
    Detecta automaticamente se as colunas extras existem no Supabase.
    """
    sb = get_client()
    if not sb:
        return 0
    count = 0
    now = datetime.now().isoformat()

    # Detectar colunas extras disponíveis no Supabase (testar uma vez)
    _extra_cols_available = None  # None = não testado, True/False = resultado

    for i, u in enumerate(updates):
        try:
            data = {
                "result_status": u["result_status"],
                "result_score": u.get("result_score", ""),
                "result_updated_at": now,
            }

            # Calcular retorno assumindo 1 unidade apostada
            if u["result_status"] == "GREEN":
                data["bet_amount"] = 1.0
                data["bet_return"] = float(u.get("market_odd", 0) or 0)
                data["bet_profit"] = data["bet_return"] - 1.0
            elif u["result_status"] == "RED":
                data["bet_amount"] = 1.0
                data["bet_return"] = 0.0
                data["bet_profit"] = -1.0
            elif u["result_status"] == "VOID":
                data["bet_amount"] = 1.0
                data["bet_return"] = 1.0
                data["bet_profit"] = 0.0

            # Dados detalhados do resultado (HT, corners, cards, shots)
            # Só tenta adicionar se as colunas existem (testado na primeira tentativa)
            if _extra_cols_available is not False:
                detail_fields = {}
                if u.get("result_ht_score"):
                    detail_fields["result_ht_score"] = u["result_ht_score"]
                if u.get("result_corners"):
                    detail_fields["result_corners"] = u["result_corners"]
                if u.get("result_cards"):
                    detail_fields["result_cards"] = u["result_cards"]
                if u.get("result_shots"):
                    detail_fields["result_shots"] = u["result_shots"]
                if u.get("result_detail"):
                    import json as _json
                    detail_fields["result_detail"] = _json.dumps(u["result_detail"])
                
                if detail_fields:
                    data_with_details = {**data, **detail_fields}
                    try:
                        result = sb.table("opportunities").update(data_with_details).eq("id", u["id"]).execute()
                        if result.data:
                            count += 1
                            if _extra_cols_available is None:
                                _extra_cols_available = True
                                print("[SUPABASE] ✅ Colunas extras (ht_score, corners, cards, shots, detail) disponíveis")
                            continue  # Sucesso com detalhes, próximo
                    except Exception as detail_err:
                        err_str = str(detail_err)
                        if "could not find" in err_str.lower() or "PGRST204" in err_str:
                            _extra_cols_available = False
                            print("[SUPABASE] ⚠️  Colunas extras não encontradas no Supabase — salvando sem detalhes")
                            print("[SUPABASE] 💡 Execute o SQL de migração no Supabase para habilitar: result_ht_score, result_corners, result_cards, result_shots, result_detail")
                        else:
                            raise detail_err

            # Fallback: salvar sem campos extras
            result = sb.table("opportunities").update(data).eq("id", u["id"]).execute()
            if result.data:
                count += 1
        except Exception as e:
            print(f"[SUPABASE] Erro ao atualizar {u.get('id', '?')}: {e}")

        # Progresso a cada 50 registros
        if (i + 1) % 50 == 0:
            print(f"[SUPABASE] Progresso: {i+1}/{len(updates)} atualizações ({count} ok)")

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


def get_all_opportunities_for_dashboard() -> dict:
    """
    Retorna dados para o dashboard:
    - 'resolved': TODAS as oportunidades com resultado (GREEN/RED/VOID)
    - 'pending_count': contagem de oportunidades ainda PENDENTE
    Busca resolvidas separadamente para não serem diluídas por pendentes.
    """
    sb = get_client()
    if not sb:
        return {"resolved": [], "pending_count": 0}
    try:
        select_cols = (
            "id, match_id, market, selection, match_date, match_time, "
            "home_team, away_team, league_name, league_country, bookmaker, "
            "market_odd, fair_odd, model_prob, implied_prob, edge, "
            "kelly_fraction, confidence, result_status, result_score, "
            "bet_amount, bet_return, bet_profit"
        )

        # 1. Buscar TODAS as resolvidas (sem limit — geralmente poucas centenas)
        resolved_result = (
            sb.table("opportunities")
            .select(select_cols)
            .neq("result_status", "PENDENTE")
            .order("match_date", desc=True)
            .limit(10000)
            .execute()
        )

        # 2. Contar pendentes (sem trazer os dados)
        pending_result = (
            sb.table("opportunities")
            .select("id", count="exact")
            .eq("result_status", "PENDENTE")
            .execute()
        )

        resolved = resolved_result.data or []
        pending_count = pending_result.count or 0

        print(f"[SUPABASE] Dashboard: {len(resolved)} resolvidas, {pending_count} pendentes")
        return {"resolved": resolved, "pending_count": pending_count}
    except Exception as e:
        print(f"[SUPABASE] Erro ao buscar dashboard: {e}")
        return {"resolved": [], "pending_count": 0}


def get_opportunities_by_dates(date_from: str, date_to: str) -> list[dict]:
    """
    Retorna TODAS as oportunidades cujo match_date está no intervalo [date_from, date_to].
    Inclui todas as colunas necessárias para a tabela principal.
    Busca tanto pendentes quanto resolvidas.
    """
    sb = get_client()
    if not sb:
        return []
    try:
        all_data = []
        page_size = 1000
        offset = 0

        while True:
            result = (
                sb.table("opportunities")
                .select("*")
                .gte("match_date", date_from)
                .lte("match_date", date_to)
                .order("match_date", desc=False)
                .order("match_time", desc=False)
                .range(offset, offset + page_size - 1)
                .execute()
            )
            batch = result.data or []
            all_data.extend(batch)
            if len(batch) < page_size:
                break
            offset += page_size

        print(f"[SUPABASE] Oportunidades {date_from}→{date_to}: {len(all_data)} encontradas")
        return all_data
    except Exception as e:
        print(f"[SUPABASE] Erro ao buscar oportunidades por data: {e}")
        return []


def get_matches_by_dates(date_from: str, date_to: str) -> list[dict]:
    """
    Retorna TODAS as partidas cujo match_date está no intervalo [date_from, date_to].
    """
    sb = get_client()
    if not sb:
        return []
    try:
        all_data = []
        page_size = 1000
        offset = 0

        while True:
            result = (
                sb.table("matches")
                .select("*")
                .gte("match_date", date_from)
                .lte("match_date", date_to)
                .order("match_date", desc=False)
                .order("match_time", desc=False)
                .range(offset, offset + page_size - 1)
                .execute()
            )
            batch = result.data or []
            all_data.extend(batch)
            if len(batch) < page_size:
                break
            offset += page_size

        print(f"[SUPABASE] Matches {date_from}→{date_to}: {len(all_data)} encontradas")
        return all_data
    except Exception as e:
        print(f"[SUPABASE] Erro ao buscar partidas por data: {e}")
        return []
