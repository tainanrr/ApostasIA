"""
═══════════════════════════════════════════════════════════════════════
     ENGINE DE ANÁLISE PREDITIVA ESPORTIVA v1.0
     Sistema Autônomo de Trading Quantitativo
═══════════════════════════════════════════════════════════════════════

Orquestrador Principal — Pipeline de Execução:

    FASE 1: Aquisição de Dados (ETL)
    FASE 2: Modelagem Estatística (Dixon-Coles + NB + Monte Carlo)
    FASE 3: Inteligência Contextual (Clima, Fadiga, Urgência)
    FASE 4: Identificação de Valor (+EV) e Relatório

═══════════════════════════════════════════════════════════════════════
"""

import time
import sys

import config
from data_ingestion import ingest_all_fixtures
from models import run_models_batch
from context_engine import apply_context_batch
from value_finder import find_all_value
from report_generator import generate_report, save_report


def print_banner():
    """Exibe o banner do sistema."""
    banner = """
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   ⚽  ENGINE DE ANÁLISE PREDITIVA ESPORTIVA  ⚽                  ║
║       Sistema Autônomo de Trading Quantitativo                   ║
║                                                                  ║
║   Modelos: Dixon-Coles │ Binomial Negativa │ Monte Carlo         ║
║   Mercados: 1x2 │ O/U │ BTTS │ Corners │ Cartões                ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def main():
    """Pipeline principal de execução."""
    print_banner()
    start = time.time()

    print(f"📅 Período de análise: {config.TODAY} (T) → {config.TOMORROW} (T+1)")
    print(f"🔧 Modo: {'DADOS SINTÉTICOS (Demo)' if config.USE_MOCK_DATA else 'API REAL (Produção)'}")
    print(f"🎯 Edge mínimo: {config.MIN_EDGE_THRESHOLD*100:.0f}%")
    print(f"🎲 Monte Carlo: {config.MONTE_CARLO_SIMULATIONS:,} simulações/jogo")
    print()

    # ═══════════════════════════════════════════════
    # FASE 1: AQUISIÇÃO DE DADOS (ETL)
    # ═══════════════════════════════════════════════
    print("=" * 60)
    print("  FASE 1: AQUISIÇÃO E ENGENHARIA DE DADOS (ETL)")
    print("=" * 60)
    t1 = time.time()
    matches = ingest_all_fixtures()
    print(f"⏱️  Fase 1 concluída em {time.time()-t1:.2f}s")
    print()

    if not matches:
        print("❌ Nenhuma partida encontrada. Abortando.")
        sys.exit(1)

    # ═══════════════════════════════════════════════
    # FASE 2: MODELAGEM ESTATÍSTICA
    # ═══════════════════════════════════════════════
    print("=" * 60)
    print("  FASE 2: MODELAGEM ESTATÍSTICA AVANÇADA")
    print("  Dixon-Coles + Binomial Negativa + Monte Carlo")
    print("=" * 60)
    t2 = time.time()
    matches = run_models_batch(matches)
    print(f"⏱️  Fase 2 concluída em {time.time()-t2:.2f}s")
    print()

    # ═══════════════════════════════════════════════
    # FASE 3: INTELIGÊNCIA CONTEXTUAL
    # ═══════════════════════════════════════════════
    print("=" * 60)
    print("  FASE 3: INTELIGÊNCIA CONTEXTUAL (AJUSTE FINO)")
    print("  Clima + Fadiga + Urgência + Lesões")
    print("=" * 60)
    t3 = time.time()
    matches = apply_context_batch(matches)
    print(f"⏱️  Fase 3 concluída em {time.time()-t3:.2f}s")
    print()

    # ═══════════════════════════════════════════════
    # FASE 4: IDENTIFICAÇÃO DE VALOR E RELATÓRIO
    # ═══════════════════════════════════════════════
    print("=" * 60)
    print("  FASE 4: ANÁLISE DE VALOR (+EV) E RELATÓRIO")
    print("=" * 60)
    t4 = time.time()

    # Encontrar oportunidades
    opportunities = find_all_value(matches)

    # Gerar relatório
    report_content = generate_report(matches, opportunities)
    report_path = save_report(report_content)

    print(f"⏱️  Fase 4 concluída em {time.time()-t4:.2f}s")
    print()

    # ═══════════════════════════════════════════════
    # RESUMO FINAL
    # ═══════════════════════════════════════════════
    total_time = time.time() - start
    n_leagues = len(set(m.league_name for m in matches))

    print("=" * 60)
    print("  ✅ EXECUÇÃO CONCLUÍDA")
    print("=" * 60)
    print(f"  📊 Ligas analisadas:     {n_leagues}")
    print(f"  ⚽ Partidas processadas: {len(matches)}")
    print(f"  🎯 Oportunidades +EV:    {len(opportunities)}")

    if opportunities:
        high = sum(1 for o in opportunities if o.confidence == "ALTO")
        med = sum(1 for o in opportunities if o.confidence == "MÉDIO")
        print(f"  🟢 Alta confiança:       {high}")
        print(f"  🟡 Média confiança:      {med}")
        print(f"  📈 Maior Edge:           {opportunities[0].edge_pct}")
        print(f"     → {opportunities[0].home_team} vs {opportunities[0].away_team}")
        print(f"       {opportunities[0].market}: {opportunities[0].selection}")

    print(f"  📄 Relatório:            {report_path}")
    print(f"  ⏱️  Tempo total:          {total_time:.2f}s")
    print("=" * 60)

    # Preview das top 3 oportunidades
    if opportunities:
        print()
        print("🔝 TOP 3 OPORTUNIDADES:")
        print("-" * 60)
        for i, opp in enumerate(opportunities[:3], 1):
            print(f"  {i}. {opp.home_team} vs {opp.away_team} ({opp.league_name})")
            print(f"     {opp.market}: {opp.selection}")
            print(f"     Odd: {opp.market_odd:.2f} → Justa: {opp.fair_odd:.2f}")
            print(f"     Edge: {opp.edge_pct} | Kelly: {opp.kelly_bet_pct}")
            print(f"     {opp.reasoning}")
            print()

    return matches, opportunities, report_content


if __name__ == "__main__":
    main()
