"""
Retroativo por seleção específica.

Percorre todas as oportunidades pendentes (result_status='PENDENTE') cuja
`selection` bate com o filtro (default: "Under 1.5 1T"), força um
re-fetch da API-Football (ignorando cache antigo que pode não ter HT),
resolve GREEN/RED/VOID e atualiza o Supabase + o cache local em disco.

Uso:
    python run_retro_selection.py
    python run_retro_selection.py --selection "Under 1.5 1T"
    python run_retro_selection.py --selection "Over 2.5 Gols" --dry-run
    python run_retro_selection.py --selection "Under 1.5 1T" --no-refetch
"""

import sys
import os
import argparse
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import config
import supabase_client
from datetime import datetime, timedelta


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def get_pending_by_selection(selection_filter: str) -> list[dict]:
    """
    Busca TODAS as oportunidades pendentes cuja selection bate com o filtro.
    Faz a filtragem no próprio Supabase (ilike) e pagina via range() para
    contornar o limite default de 1000 linhas da API REST.
    """
    sb = supabase_client.get_client()
    if not sb:
        return []
    import pytz
    br_tz = pytz.timezone("America/Sao_Paulo")
    today = datetime.now(br_tz).strftime("%Y-%m-%d")

    cols = ("id, match_id, market, selection, match_date, match_time, "
            "home_team, away_team, market_odd, model_prob, edge, confidence, "
            "league_name, league_country, bookmaker")
    pattern = f"%{selection_filter}%"
    page_size = 1000
    offset = 0
    all_rows: list[dict] = []
    while True:
        resp = (
            sb.table("opportunities")
            .select(cols)
            .eq("result_status", "PENDENTE")
            .lte("match_date", today)
            .ilike("selection", pattern)
            .order("match_date", desc=False)
            .range(offset, offset + page_size - 1)
            .execute()
        )
        rows = resp.data or []
        if not rows:
            break
        all_rows.extend(rows)
        if len(rows) < page_size:
            break
        offset += page_size
    return all_rows


def fetch_one_fresh(mid: int) -> dict | None:
    """
    Busca UM jogo da API-Football forçando skip_cache=True (garante HT atualizado).
    Retorna dict no mesmo formato de fetch_finished_fixtures, ou None se não finalizado.
    """
    from data_ingestion import (
        _api_football_request,
        _save_to_cache,
        _extract_fixture_stats,
    )

    raw = _api_football_request("fixtures", {"id": mid}, skip_cache=True)
    if not raw:
        return None
    response = raw.get("response", []) or []
    if not response:
        return None

    fix = response[0]
    fix_data = fix.get("fixture", {}) or {}
    goals = fix.get("goals", {}) or {}
    score = fix.get("score", {}) or {}
    status_short = (fix_data.get("status", {}) or {}).get("short", "?")

    hg = goals.get("home")
    ag = goals.get("away")
    finished = status_short in ("FT", "AET", "PEN")

    if not finished or hg is None or ag is None:
        return {
            "status": status_short,
            "home_goals": None,
            "away_goals": None,
            "score": None,
        }

    ht = score.get("halftime", {}) or {}
    entry = {
        "status": status_short,
        "home_goals": hg,
        "away_goals": ag,
        "score": f"{hg}-{ag}",
        "ht_home": ht.get("home"),
        "ht_away": ht.get("away"),
    }

    try:
        stats = _extract_fixture_stats(mid)
        entry.update(stats)
    except Exception as e:
        print(f"    [AVISO] falha ao buscar stats do jogo {mid}: {e}")

    _save_to_cache("fixtures", {"id": mid}, raw)
    return entry


def run_retro_for_selection(
    selection_filter: str = "Under 1.5 1T",
    dry_run: bool = False,
    force_refetch: bool = True,
    max_age_days: int | None = None,
):
    """
    Resolve retroativamente todas as oportunidades pendentes cuja selection
    corresponde ao filtro informado.
    """
    from app import _resolve_opportunity, _update_cache_with_results

    exec_id = supabase_client.log_execution_start("results", f"manual_retro:{selection_filter}")
    start = time.time()
    now = datetime.now(config.BR_TIMEZONE)
    target = _norm(selection_filter)

    try:
        filtered = get_pending_by_selection(selection_filter)
        pending_count = len(filtered)
        print(f"[RETRO] Pendentes no Supabase com selection ~ '{selection_filter}': {pending_count}")
        if not filtered:
            supabase_client.log_execution_end(
                exec_id, "success",
                {"filtered": 0, "resolved": 0, "selection_filter": selection_filter},
            )
            return

        eligible = []
        skipped_recent = 0
        skipped_old = 0
        cutoff_old = now - timedelta(days=max_age_days) if max_age_days else None

        for opp in filtered:
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
                if cutoff_old and match_dt < cutoff_old:
                    skipped_old += 1
                    continue
                elapsed_min = (now - match_dt).total_seconds() / 60
                if elapsed_min < 120:
                    skipped_recent += 1
                    continue
            except Exception:
                pass
            eligible.append(opp)

        print(f"[RETRO] Elegíveis (encerradas >120min): {len(eligible)} "
              f"| ignoradas em andamento: {skipped_recent} | muito antigas: {skipped_old}")

        match_ids = {opp.get("match_id") for opp in eligible if opp.get("match_id")}
        print(f"[RETRO] Jogos únicos a buscar na API: {len(match_ids)}")

        results: dict = {}
        for i, mid in enumerate(sorted(match_ids), 1):
            if force_refetch:
                entry = fetch_one_fresh(mid)
            else:
                from data_ingestion import fetch_finished_fixtures
                entry = (fetch_finished_fixtures([mid]) or {}).get(mid)

            if entry:
                results[mid] = entry
                ht_info = ""
                if entry.get("score"):
                    htm = entry.get("ht_home")
                    hta = entry.get("ht_away")
                    ht_info = f" | HT {htm}-{hta}" if htm is not None and hta is not None else " | HT: sem dados"
                    print(f"  [{i}/{len(match_ids)}] fixture={mid} FT {entry['score']}{ht_info}")
                else:
                    print(f"  [{i}/{len(match_ids)}] fixture={mid} ainda não finalizado (status={entry.get('status')})")

        finished_count = sum(1 for r in results.values() if r.get("score"))
        print(f"[RETRO] Jogos com resultado final: {finished_count}/{len(match_ids)}")

        updates = []
        unresolved_no_ht = 0
        for opp in eligible:
            mid = opp.get("match_id")
            result = results.get(mid)
            if not result or not result.get("score"):
                continue
            hg = result["home_goals"]
            ag = result["away_goals"]
            status = _resolve_opportunity(opp, hg, ag, result)
            if not status:
                unresolved_no_ht += 1
                print(f"  [SKIP] {opp.get('home_team')} vs {opp.get('away_team')} | "
                      f"match_id={mid} | sem HT na API para resolver '{opp.get('selection')}'")
                continue

            ht_h = result.get("ht_home")
            ht_a = result.get("ht_away")
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
                "market": opp.get("market", ""),
                "selection": opp.get("selection", ""),
                "result_status": status,
                "result_score": result["score"],
                "result_ht_score": result_ht_score,
                "result_corners": result_corners,
                "result_cards": result_cards,
                "result_shots": result_shots,
                "result_detail": {
                    "ht_home": ht_h, "ht_away": ht_a,
                    "corners_home": c_h, "corners_away": c_a,
                    "cards_home": cd_h, "cards_away": cd_a,
                    "shots_home": sh_h, "shots_away": sh_a,
                    "shots_on_home": sot_h, "shots_on_away": sot_a,
                },
                "market_odd": opp.get("market_odd", 0),
            })

        n_green = sum(1 for u in updates if u["result_status"] == "GREEN")
        n_red = sum(1 for u in updates if u["result_status"] == "RED")
        n_void = sum(1 for u in updates if u["result_status"] == "VOID")
        print(f"[RETRO] A resolver: {len(updates)} | GREEN={n_green} | RED={n_red} | VOID={n_void} "
              f"| sem HT: {unresolved_no_ht}")

        if dry_run:
            print("[RETRO] DRY-RUN: nada foi salvo no Supabase.")
            for u in updates[:20]:
                print(f"   - {u['result_status']} | {u['selection']} | fixture {u['match_id']} "
                      f"| FT {u['result_score']} | HT {u['result_ht_score']}")
            if len(updates) > 20:
                print(f"   ... e mais {len(updates) - 20} linhas.")
            supabase_client.log_execution_end(
                exec_id, "success",
                {"dry_run": True,
                 "filtered": len(filtered), "eligible": len(eligible),
                 "would_resolve": len(updates),
                 "green": n_green, "red": n_red, "void": n_void,
                 "selection_filter": selection_filter},
            )
            return

        saved = 0
        if updates:
            saved = supabase_client.batch_update_results(updates)
            print(f"[RETRO] Supabase atualizado: {saved} oportunidades.")
            try:
                _update_cache_with_results(updates)
            except Exception as e:
                print(f"[RETRO] (aviso) não foi possível atualizar cache local: {e}")

        elapsed = round(time.time() - start, 2)
        supabase_client.log_execution_end(
            exec_id, "success",
            {
                "selection_filter": selection_filter,
                "filtered": len(filtered),
                "eligible": len(eligible),
                "checked_matches": len(match_ids),
                "finished_matches": finished_count,
                "resolved": saved,
                "green": n_green, "red": n_red, "void": n_void,
                "unresolved_no_ht": unresolved_no_ht,
                "elapsed_seconds": elapsed,
            },
        )
        print(f"[RETRO] Concluído em {elapsed}s. Resolvidas: {saved} | "
              f"{n_green}G / {n_red}R / {n_void}V | sem HT: {unresolved_no_ht}")

    except Exception as e:
        supabase_client.log_execution_end(exec_id, "error", error_message=str(e))
        print(f"[RETRO] Erro: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retroativo por seleção")
    parser.add_argument("--selection", type=str, default="Under 1.5 1T",
                        help="Texto da seleção (default: 'Under 1.5 1T')")
    parser.add_argument("--dry-run", action="store_true",
                        help="Apenas lista o que seria resolvido, sem gravar.")
    parser.add_argument("--no-refetch", action="store_true",
                        help="Usa cache local ao invés de forçar chamada fresca na API.")
    parser.add_argument("--max-age-days", type=int, default=None,
                        help="Opcional: ignora jogos mais antigos que N dias.")
    args = parser.parse_args()

    run_retro_for_selection(
        selection_filter=args.selection,
        dry_run=args.dry_run,
        force_refetch=not args.no_refetch,
        max_age_days=args.max_age_days,
    )
