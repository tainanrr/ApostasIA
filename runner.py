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


def run_pipeline(days_range: int = 2, trigger: str = "manual"):
    """Executa o pipeline de análise para os próximos N dias."""
    from data_ingestion import ingest_all_fixtures
    from models import run_models_batch
    from context_engine import apply_context_batch
    from value_finder import find_all_value

    exec_id = supabase_client.log_execution_start("pipeline", trigger)
    now = datetime.now(config.BR_TIMEZONE)
    dates = [(now + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days_range)]

    print(f"[RUNNER] Pipeline: {dates} ({days_range} dias) | trigger={trigger}")
    start = time.time()

    try:
        matches = ingest_all_fixtures(analysis_dates=dates)
        if not matches:
            supabase_client.log_execution_end(exec_id, "success", {"matches": 0, "opportunities": 0, "dates": dates})
            print("[RUNNER] Nenhuma partida encontrada.")
            return

        matches = run_models_batch(matches)
        matches = apply_context_batch(matches)
        opportunities = find_all_value(matches)

        n_leagues = len(set(m.league_name for m in matches))
        elapsed = round(time.time() - start, 2)

        from data_ingestion import _api_call_count
        details = {
            "matches": len(matches),
            "opportunities": len(opportunities),
            "leagues": n_leagues,
            "api_calls": _api_call_count,
            "elapsed_seconds": elapsed,
            "dates": dates,
        }

        if opportunities:
            saved = supabase_client.save_opportunities(opportunities, matches)
            details["saved_to_supabase"] = saved
            print(f"[RUNNER] Salvas {saved} oportunidades no Supabase")

        supabase_client.log_execution_end(exec_id, "success", details)
        print(f"[RUNNER] Pipeline concluído: {len(matches)} jogos, {len(opportunities)} oportunidades em {elapsed}s")

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
    """Verifica Supabase se há algo agendado para o horário atual (+/- 15min)."""
    configs = supabase_client.get_scheduler_config()
    if not configs:
        print("[RUNNER] Nenhuma configuração de scheduler encontrada.")
        return

    now = datetime.now(config.BR_TIMEZONE)
    current_hhmm = now.strftime("%H:%M")
    current_minutes = now.hour * 60 + now.minute
    ran_something = False

    for cfg in configs:
        if not cfg.get("enabled"):
            continue
        hours = cfg.get("hours", [])
        days_range = cfg.get("days_range", 2)
        cfg_type = cfg["id"]

        for h in hours:
            try:
                parts = h.split(":")
                sched_minutes = int(parts[0]) * 60 + int(parts[1])
            except (ValueError, IndexError):
                continue

            diff = abs(current_minutes - sched_minutes)
            if diff <= 15:
                print(f"[RUNNER] Horário compatível: {cfg_type} agendado para {h}, agora são {current_hhmm}")
                if cfg_type == "pipeline":
                    run_pipeline(days_range=days_range, trigger="scheduled")
                elif cfg_type == "results":
                    run_results(days_range=days_range, trigger="scheduled")
                ran_something = True
                break

    if not ran_something:
        print(f"[RUNNER] Nenhuma execução agendada para {current_hhmm}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ApostasIA Runner")
    parser.add_argument("--check-schedule", action="store_true", help="Verifica agenda e executa se necessário")
    parser.add_argument("--pipeline", action="store_true", help="Executa pipeline de análise")
    parser.add_argument("--results", action="store_true", help="Verifica resultados")
    parser.add_argument("--days", type=int, default=None, help="Range de dias (1-5)")
    parser.add_argument("--trigger", type=str, default="github_actions", help="Origem da execução")
    args = parser.parse_args()

    if args.check_schedule:
        check_schedule()
    elif args.pipeline:
        run_pipeline(days_range=args.days or 2, trigger=args.trigger)
    elif args.results:
        run_results(days_range=args.days or 2, trigger=args.trigger)
    else:
        parser.print_help()
