"""
Runner standalone para GitHub Actions.
Verifica config no Supabase e executa pipeline/resultados conforme agendamento.
Pode ser chamado com argumentos diretos para execução manual.

Uso:
    python runner.py --check-schedule      # Verifica se há algo agendado para agora
    python runner.py --pipeline [--days N] # Executa pipeline (N dias à frente, default 2)
    python runner.py --results [--days N]  # Verifica resultados (N dias para trás, default 2)
"""

import sys
import os
import argparse
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import config
from datetime import datetime, timedelta

import supabase_client


def run_pipeline(days_range: int = 2, trigger: str = "manual", filters: dict = None):
    """Executa o pipeline de análise para os próximos N dias.
    Para períodos grandes (>7 dias), processa em batches de 7 dias.
    filters: dict opcional com league_ids, countries, tiers para filtrar fixtures."""
    from data_ingestion import ingest_all_fixtures
    from models import run_models_batch
    from context_engine import apply_context_batch
    from value_finder import find_all_value
    from app import serialize_opportunity, serialize_match

    exec_id = supabase_client.log_execution_start("pipeline", trigger)
    now = datetime.now(config.BR_TIMEZONE)
    dates = [(now + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days_range)]

    filter_desc = ""
    if filters:
        parts = []
        if filters.get("league_ids"):
            parts.append(f"ligas={filters['league_ids']}")
        if filters.get("countries"):
            parts.append(f"países={filters['countries']}")
        if filters.get("tiers"):
            parts.append(f"tiers={filters['tiers']}")
        filter_desc = f" | filtros: {', '.join(parts)}"

    print(f"[RUNNER] Pipeline: {dates[0]}..{dates[-1]} ({days_range} dias) | trigger={trigger}{filter_desc}")
    start = time.time()

    BATCH_SIZE = 7
    all_matches = []
    all_opportunities = []

    try:
        for batch_start in range(0, len(dates), BATCH_SIZE):
            batch_dates = dates[batch_start:batch_start + BATCH_SIZE]
            batch_num = batch_start // BATCH_SIZE + 1
            total_batches = (len(dates) + BATCH_SIZE - 1) // BATCH_SIZE

            if total_batches > 1:
                print(f"[RUNNER] ═══ Batch {batch_num}/{total_batches}: {batch_dates[0]} → {batch_dates[-1]} ({len(batch_dates)} dias) ═══")

            matches = ingest_all_fixtures(analysis_dates=batch_dates, filters=filters)
            if not matches:
                print(f"[RUNNER] Batch {batch_num}: nenhuma partida encontrada.")
                continue

            matches = run_models_batch(matches)
            matches = apply_context_batch(matches)
            opportunities = find_all_value(matches)

            all_matches.extend(matches)
            all_opportunities.extend(opportunities)

            if total_batches > 1:
                print(f"[RUNNER] Batch {batch_num}: {len(matches)} jogos, {len(opportunities)} oportunidades")

        if not all_matches:
            supabase_client.log_execution_end(exec_id, "success", {"matches": 0, "opportunities": 0, "dates": dates})
            print("[RUNNER] Nenhuma partida encontrada.")
            return

        n_leagues = len(set(m.league_name for m in all_matches))
        elapsed = round(time.time() - start, 2)

        from data_ingestion import _api_call_count
        details = {
            "matches": len(all_matches),
            "opportunities": len(all_opportunities),
            "leagues": n_leagues,
            "api_calls": _api_call_count,
            "elapsed_seconds": elapsed,
            "dates": dates,
        }
        if filters:
            details["filters"] = filters

        if all_opportunities:
            serialized_opps = [serialize_opportunity(o) for o in all_opportunities]
            serialized_matches = [serialize_match(m) for m in all_matches]
            stats = {
                "analysis_dates": dates,
                "total_matches": len(all_matches),
                "total_leagues": n_leagues,
                "total_opportunities": len(all_opportunities),
                "high_conf": sum(1 for o in all_opportunities if o.confidence == "ALTO"),
                "med_conf": sum(1 for o in all_opportunities if o.confidence == "MÉDIO"),
                "low_conf": sum(1 for o in all_opportunities if o.confidence == "BAIXO"),
                "avg_edge": round(sum(o.edge for o in all_opportunities) / len(all_opportunities) * 100, 2) if all_opportunities else 0,
                "max_edge": round(max(o.edge for o in all_opportunities) * 100, 2) if all_opportunities else 0,
                "run_time": elapsed,
                "api_calls_this_run": _api_call_count,
                "mode": "API Real",
            }
            supabase_client.save_full_run(stats, serialized_opps, serialized_matches)
            details["saved_to_supabase"] = len(serialized_opps)
            print(f"[RUNNER] Salvas {len(serialized_opps)} oportunidades no Supabase")

        supabase_client.log_execution_end(exec_id, "success", details)
        print(f"[RUNNER] Pipeline concluído: {len(all_matches)} jogos, {len(all_opportunities)} oportunidades em {elapsed}s")

    except Exception as e:
        supabase_client.log_execution_end(exec_id, "error", error_message=str(e))
        print(f"[RUNNER] Erro no pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def run_results(days_range: int = 2, trigger: str = "manual"):
    """Verifica resultados de jogos dos últimos N dias."""
    from data_ingestion import fetch_finished_fixtures

    exec_id = supabase_client.log_execution_start("results", trigger)
    print(f"[RUNNER] Resultados: últimos {days_range} dias | trigger={trigger}")
    start = time.time()

    try:
        pending = supabase_client.get_pending_opportunities()
        if not pending:
            supabase_client.log_execution_end(exec_id, "success", {"total_pending": 0, "resolved": 0})
            print("[RUNNER] Nenhuma oportunidade pendente.")
            return

        now = datetime.now(config.BR_TIMEZONE)
        cutoff = now - timedelta(days=days_range)
        eligible = []
        skipped = 0

        for opp in pending:
            md = opp.get("match_date", "")
            mt = opp.get("match_time", "00:00")
            try:
                dt_parts = md.split("-")
                tm_parts = (mt or "00:00").split(":")
                match_dt = datetime(
                    int(dt_parts[0]), int(dt_parts[1]), int(dt_parts[2]),
                    int(tm_parts[0]), int(tm_parts[1]), 0,
                    tzinfo=config.BR_TIMEZONE,
                )
                if match_dt < cutoff:
                    continue
                elapsed_min = (now - match_dt).total_seconds() / 60
                if elapsed_min >= 120:
                    eligible.append(opp)
                else:
                    skipped += 1
            except Exception:
                eligible.append(opp)

        if not eligible:
            supabase_client.log_execution_end(exec_id, "success", {
                "total_pending": len(pending), "eligible": 0, "resolved": 0
            })
            print(f"[RUNNER] Nenhum jogo elegível. {skipped} ainda em andamento.")
            return

        match_ids = {opp.get("match_id") for opp in eligible if opp.get("match_id")}
        print(f"[RUNNER] {len(eligible)} oportunidades elegíveis, {len(match_ids)} jogos a verificar")

        results = fetch_finished_fixtures(list(match_ids))
        finished = sum(1 for r in results.values() if r.get("score"))

        from app import _resolve_opportunity
        updates = []
        for opp in eligible:
            mid = opp.get("match_id")
            result = results.get(str(mid)) or results.get(mid)
            if not result or not result.get("score"):
                continue
            hg = result["home_goals"]
            ag = result["away_goals"]
            status = _resolve_opportunity(opp, hg, ag, result)
            if status:
                updates.append({
                    "id": opp["id"],
                    "result_status": status,
                    "result_score": f"{hg}-{ag}",
                })

        saved = 0
        if updates:
            saved = supabase_client.batch_update_results(updates)

        n_green = sum(1 for u in updates if u["result_status"] == "GREEN")
        n_red = sum(1 for u in updates if u["result_status"] == "RED")
        n_void = sum(1 for u in updates if u["result_status"] == "VOID")
        elapsed_s = round(time.time() - start, 2)

        details = {
            "total_pending": len(pending),
            "eligible": len(eligible),
            "checked_matches": len(match_ids),
            "finished_matches": finished,
            "resolved": saved,
            "green": n_green,
            "red": n_red,
            "void": n_void,
            "elapsed_seconds": elapsed_s,
        }
        supabase_client.log_execution_end(exec_id, "success", details)
        print(f"[RUNNER] Resultados: {saved} resolvidos ({n_green}G/{n_red}R/{n_void}V) em {elapsed_s}s")

    except Exception as e:
        supabase_client.log_execution_end(exec_id, "error", error_message=str(e))
        print(f"[RUNNER] Erro nos resultados: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def check_schedule():
    """Verifica slots agendados e executa os que já passaram e não rodaram hoje.

    Estratégia robusta contra delays do GitHub Actions cron:
    - Para cada slot cujo horário já passou (com margem de 90min para catch-up),
      verifica se já houve execução bem-sucedida desde 30min antes do slot.
    - Se não houve, executa agora (catch-up).
    - Executa no máximo 1 slot por tipo (pipeline/results) por invocação.
    """
    configs = supabase_client.get_scheduler_config()
    if not configs:
        print("[RUNNER] Nenhuma configuração de scheduler encontrada.")
        return

    now = datetime.now(config.BR_TIMEZONE)
    current_hhmm = now.strftime("%H:%M")
    current_minutes = now.hour * 60 + now.minute
    today_str = now.strftime("%Y-%m-%d")
    ran_something = False

    for cfg in configs:
        if not cfg.get("enabled"):
            continue
        hours = cfg.get("hours", [])
        days_range = cfg.get("days_range", 2)
        cfg_type = cfg["id"]

        # Ordena slots do mais recente para o mais antigo (prioriza o mais próximo)
        parsed_slots = []
        for h in hours:
            try:
                parts = h.split(":")
                slot_min = int(parts[0]) * 60 + int(parts[1])
                parsed_slots.append((slot_min, h))
            except (ValueError, IndexError):
                continue
        parsed_slots.sort(reverse=True)

        for slot_min, h in parsed_slots:
            diff = current_minutes - slot_min  # positivo = já passou

            # Slot deve estar entre 0 e 90 min no passado
            if diff < -5 or diff > 90:
                continue

            # Verifica dedup: checa se já rodou desde 30min antes do slot
            dedup_minutes = max(0, slot_min - 30)
            dedup_dt = now.replace(
                hour=dedup_minutes // 60,
                minute=dedup_minutes % 60,
                second=0, microsecond=0,
            )
            since_iso = dedup_dt.isoformat()

            if supabase_client.was_executed_today(cfg_type, since_iso):
                print(f"[RUNNER] Slot {cfg_type}@{h} já executado hoje, skip.")
                continue

            print(f"[RUNNER] Catch-up: {cfg_type} agendado para {h}, agora {current_hhmm} (atraso {diff}min)")
            if cfg_type == "pipeline":
                run_pipeline(days_range=days_range, trigger="scheduled")
            elif cfg_type == "results":
                run_results(days_range=days_range, trigger="scheduled")
            ran_something = True
            break  # 1 execução por tipo por invocação

    if not ran_something:
        print(f"[RUNNER] Nenhuma execução pendente para {current_hhmm}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ApostasIA Runner")
    parser.add_argument("--check-schedule", action="store_true", help="Verifica agenda e executa se necessário")
    parser.add_argument("--pipeline", action="store_true", help="Executa pipeline de análise")
    parser.add_argument("--results", action="store_true", help="Verifica resultados")
    parser.add_argument("--days", type=int, default=None, help="Range de dias (1-90)")
    parser.add_argument("--trigger", type=str, default="github_actions", help="Origem da execução")
    parser.add_argument("--leagues", type=str, default=None, help="IDs de ligas separados por vírgula (ex: 39,140,135)")
    parser.add_argument("--countries", type=str, default=None, help="Países separados por vírgula (ex: England,Spain)")
    parser.add_argument("--tiers", type=str, default=None, help="Tiers de liga separados por vírgula (ex: S,A)")
    parser.add_argument("--filter", type=str, default=None, help="Nome de filtro salvo no Supabase para aplicar")
    args = parser.parse_args()

    # Montar filtros a partir dos argumentos
    run_filters = None
    if args.leagues or args.countries or args.tiers or args.filter:
        run_filters = {}
        if args.filter:
            saved = supabase_client.get_filter_views()
            found = [v for v in saved if v.get("name", "").lower() == args.filter.lower()]
            if found:
                run_filters = found[0].get("state", {})
                print(f"[RUNNER] Filtro carregado: '{args.filter}' → {run_filters}")
            else:
                print(f"[RUNNER] ⚠️  Filtro '{args.filter}' não encontrado. Filtros disponíveis:")
                for v in saved:
                    print(f"    - {v.get('name')}")
        if args.leagues:
            run_filters["league_ids"] = [int(x.strip()) for x in args.leagues.split(",") if x.strip().isdigit()]
        if args.countries:
            run_filters["countries"] = [x.strip() for x in args.countries.split(",")]
        if args.tiers:
            run_filters["tiers"] = [x.strip().upper() for x in args.tiers.split(",")]

    if args.check_schedule:
        check_schedule()
    elif args.pipeline:
        run_pipeline(days_range=args.days or 2, trigger=args.trigger, filters=run_filters)
    elif args.results:
        run_results(days_range=args.days or 2, trigger=args.trigger)
    else:
        parser.print_help()
