"""
═══════════════════════════════════════════════════════════════════════
MÓDULO DE GERAÇÃO DE RELATÓRIO
Engine de Análise Preditiva - Camada de Apresentação
═══════════════════════════════════════════════════════════════════════

Gera relatório Markdown completo organizado por liga.
"""

from collections import defaultdict
from datetime import datetime

import config
from data_ingestion import MatchAnalysis
from value_finder import ValueOpportunity


def generate_report(matches: list[MatchAnalysis],
                     opportunities: list[ValueOpportunity]) -> str:
    """
    Gera o relatório diário completo em Markdown.
    """
    lines = []
    now = datetime.now(config.BR_TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")

    # ═══════════════════════════════════════════════
    # CABEÇALHO
    # ═══════════════════════════════════════════════
    lines.append("# 🏟️ RELATÓRIO DIÁRIO DE ANÁLISE QUANTITATIVA ESPORTIVA")
    lines.append("")
    lines.append(f"**Gerado em:** {now}")
    lines.append(f"**Período de Análise:** {config.TODAY} (T) → {config.TOMORROW} (T+1) → {config.DAY_AFTER_TOMORROW} (T+2)")
    lines.append(f"**Modo:** {'Dados Sintéticos (Demo)' if config.USE_MOCK_DATA else 'API Real (Produção)'}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ═══════════════════════════════════════════════
    # RESUMO EXECUTIVO
    # ═══════════════════════════════════════════════
    lines.append("## 📊 Resumo Executivo")
    lines.append("")

    n_leagues = len(set(m.league_name for m in matches))
    n_matches = len(matches)
    n_opps = len(opportunities)
    n_high = sum(1 for o in opportunities if o.confidence == "ALTO")
    n_med = sum(1 for o in opportunities if o.confidence == "MÉDIO")
    n_low = sum(1 for o in opportunities if o.confidence == "BAIXO")

    lines.append(f"| Métrica | Valor |")
    lines.append(f"|---------|-------|")
    lines.append(f"| Ligas Analisadas | **{n_leagues}** |")
    lines.append(f"| Total de Partidas | **{n_matches}** |")
    lines.append(f"| Oportunidades +EV (≥{config.MIN_EDGE_THRESHOLD*100:.0f}%) | **{n_opps}** |")
    lines.append(f"| Confiança ALTA | **{n_high}** 🟢 |")
    lines.append(f"| Confiança MÉDIA | **{n_med}** 🟡 |")
    lines.append(f"| Confiança BAIXA | **{n_low}** 🔴 |")
    lines.append(f"| Modelos Utilizados | Dixon-Coles, Binomial Negativa, Monte Carlo |")
    lines.append(f"| Simulações por Jogo | {config.MONTE_CARLO_SIMULATIONS:,} |")
    lines.append("")

    if n_opps > 0:
        avg_edge = sum(o.edge for o in opportunities) / n_opps
        max_edge_opp = opportunities[0]
        lines.append(f"**Edge Médio:** {avg_edge*100:.2f}% | "
                     f"**Maior Edge:** {max_edge_opp.edge_pct} "
                     f"({max_edge_opp.home_team} vs {max_edge_opp.away_team} - {max_edge_opp.selection})")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ═══════════════════════════════════════════════
    # TOP PICKS - OPORTUNIDADES DESTAQUE
    # ═══════════════════════════════════════════════
    lines.append("## 🎯 Top Picks — Oportunidades Destaque")
    lines.append("")

    top_picks = [o for o in opportunities if o.confidence in ("ALTO", "MÉDIO")][:15]

    if top_picks:
        lines.append("| # | Jogo | Liga | Mercado | Seleção | Odd Casa | Odd Justa | Edge | Kelly | Confiança |")
        lines.append("|---|------|------|---------|---------|----------|-----------|------|-------|-----------|")

        for i, opp in enumerate(top_picks, 1):
            conf_icon = "🟢" if opp.confidence == "ALTO" else "🟡"
            game_str = f"{opp.home_team} vs {opp.away_team}"
            lines.append(
                f"| {i} | {game_str} | {opp.league_name} | "
                f"{opp.market} | {opp.selection} | "
                f"{opp.market_odd:.2f} | {opp.fair_odd:.2f} | "
                f"**{opp.edge_pct}** | {opp.kelly_bet_pct} | "
                f"{conf_icon} {opp.confidence} |"
            )
        lines.append("")

        # Análise detalhada dos Top 5
        lines.append("### 📝 Análise Detalhada — Top 5")
        lines.append("")

        for i, opp in enumerate(top_picks[:5], 1):
            lines.append(f"**{i}. {opp.home_team} vs {opp.away_team}** "
                        f"({opp.league_name} — {opp.league_country})")
            lines.append(f"- 📅 {opp.match_date} às {opp.match_time}")
            lines.append(f"- 🎲 **Mercado:** {opp.market} → **{opp.selection}**")
            lines.append(f"- 💰 Odd Casa: {opp.market_odd:.2f} | "
                        f"Odd Justa (Modelo): {opp.fair_odd:.2f}")
            lines.append(f"- 📈 **Edge: {opp.edge_pct}** | "
                        f"Prob. Modelo: {opp.model_prob*100:.1f}% vs "
                        f"Prob. Implícita: {opp.implied_prob*100:.1f}%")
            lines.append(f"- 💵 Kelly Sugerido: {opp.kelly_bet_pct} da banca")
            lines.append(f"- ⚽ xG: Casa {opp.home_xg:.2f} — Fora {opp.away_xg:.2f}")
            lines.append(f"- 🌤️ Clima: {opp.weather_note}")
            lines.append(f"- 🔥 Urgência: Casa {opp.urgency_home:.1f} | "
                        f"Fora {opp.urgency_away:.1f}")
            lines.append(f"- 🧠 **Análise:** {opp.reasoning}")
            lines.append("")

    else:
        lines.append("*Nenhuma oportunidade de alta/média confiança encontrada.*")
        lines.append("")

    lines.append("---")
    lines.append("")

    # ═══════════════════════════════════════════════
    # ANÁLISE POR LIGA
    # ═══════════════════════════════════════════════
    lines.append("## 🌍 Análise por Liga")
    lines.append("")

    # Agrupar oportunidades por liga
    opps_by_league = defaultdict(list)
    for opp in opportunities:
        opps_by_league[f"{opp.league_country} — {opp.league_name}"].append(opp)

    # Agrupar matches por liga (para mostrar ligas sem oportunidades também)
    matches_by_league = defaultdict(list)
    for m in matches:
        matches_by_league[f"{m.league_country} — {m.league_name}"].append(m)

    for league_key in sorted(matches_by_league.keys()):
        league_matches = matches_by_league[league_key]
        league_opps = opps_by_league.get(league_key, [])

        lines.append(f"### 🏆 {league_key}")
        lines.append(f"*{len(league_matches)} jogo(s) | "
                     f"{len(league_opps)} oportunidade(s) +EV*")
        lines.append("")

        # Tabela de jogos da liga
        lines.append("| Jogo | Hora | xG Casa | xG Fora | Prob H/D/A | O/U 2.5 | BTTS | Corners | Cartões |")
        lines.append("|------|------|---------|---------|------------|---------|------|---------|---------|")

        for m in league_matches:
            prob_str = (f"{m.model_prob_home*100:.0f}%/"
                       f"{m.model_prob_draw*100:.0f}%/"
                       f"{m.model_prob_away*100:.0f}%")
            game_str = f"{m.home_team.team_name} vs {m.away_team.team_name}"
            lines.append(
                f"| {game_str} | {m.match_time} | "
                f"{m.model_home_xg:.2f} | {m.model_away_xg:.2f} | "
                f"{prob_str} | "
                f"{m.model_prob_over25*100:.0f}% | "
                f"{m.model_prob_btts*100:.0f}% | "
                f"{m.model_corners_expected:.1f} | "
                f"{m.model_cards_expected:.1f} |"
            )
        lines.append("")

        # Oportunidades específicas da liga
        if league_opps:
            lines.append("**Oportunidades Identificadas:**")
            lines.append("")
            for opp in league_opps:
                conf_icon = {"ALTO": "🟢", "MÉDIO": "🟡", "BAIXO": "🔴"}.get(opp.confidence, "⚪")
                lines.append(
                    f"- {conf_icon} **{opp.home_team} vs {opp.away_team}** → "
                    f"{opp.market}: {opp.selection} @ {opp.market_odd:.2f} "
                    f"(Edge: **{opp.edge_pct}**) — {opp.reasoning}"
                )
            lines.append("")

        lines.append("---")
        lines.append("")

    # ═══════════════════════════════════════════════
    # ANÁLISE DE MERCADOS ESPECIAIS
    # ═══════════════════════════════════════════════
    lines.append("## 📋 Análise por Tipo de Mercado")
    lines.append("")

    market_types = defaultdict(list)
    for opp in opportunities:
        market_types[opp.market].append(opp)

    for market_name, market_opps in sorted(market_types.items()):
        lines.append(f"### {market_name}")
        lines.append(f"*{len(market_opps)} oportunidade(s)*")
        lines.append("")

        lines.append("| Jogo | Seleção | Odd | Odd Justa | Edge | Kelly | Confiança |")
        lines.append("|------|---------|-----|-----------|------|-------|-----------|")

        for opp in market_opps[:10]:
            conf_icon = {"ALTO": "🟢", "MÉDIO": "🟡", "BAIXO": "🔴"}.get(opp.confidence, "⚪")
            lines.append(
                f"| {opp.home_team} vs {opp.away_team} | "
                f"{opp.selection} | {opp.market_odd:.2f} | "
                f"{opp.fair_odd:.2f} | **{opp.edge_pct}** | "
                f"{opp.kelly_bet_pct} | {conf_icon} {opp.confidence} |"
            )
        lines.append("")

    lines.append("---")
    lines.append("")

    # ═══════════════════════════════════════════════
    # CONDIÇÕES METEOROLÓGICAS RELEVANTES
    # ═══════════════════════════════════════════════
    lines.append("## 🌦️ Alertas Meteorológicos")
    lines.append("")

    weather_alerts = []
    for m in matches:
        alerts = []
        if m.weather.wind_speed_kmh > config.WIND_SPEED_THRESHOLD_KMH:
            alerts.append(f"💨 Vento: {m.weather.wind_speed_kmh:.0f} km/h")
        if m.weather.rain_mm > config.RAIN_VOLUME_THRESHOLD_MM:
            alerts.append(f"🌧️ Chuva: {m.weather.rain_mm:.1f}mm")
        if m.weather.temperature_c > config.HEAT_THRESHOLD_C:
            alerts.append(f"🌡️ Calor: {m.weather.temperature_c:.0f}°C")
        if alerts:
            weather_alerts.append(
                f"| {m.home_team.team_name} vs {m.away_team.team_name} | "
                f"{m.league_name} | {' '.join(alerts)} | {m.weather.description} |"
            )

    if weather_alerts:
        lines.append("| Jogo | Liga | Alertas | Condição |")
        lines.append("|------|------|---------|----------|")
        for alert in weather_alerts:
            lines.append(alert)
    else:
        lines.append("*Nenhum alerta meteorológico significativo.*")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ═══════════════════════════════════════════════
    # FADIGA / ROTAÇÃO
    # ═══════════════════════════════════════════════
    lines.append("## ⚡ Alertas de Fadiga (< 72h entre jogos)")
    lines.append("")

    fatigue_alerts = []
    for m in matches:
        if m.home_fatigue:
            fatigue_alerts.append(
                f"- ⚠️ **{m.home_team.team_name}** (Casa) jogou recentemente — "
                f"Último jogo: {m.home_team.last_match_date or 'N/D'} | "
                f"Partida: vs {m.away_team.team_name} ({m.league_name})"
            )
        if m.away_fatigue:
            fatigue_alerts.append(
                f"- ⚠️ **{m.away_team.team_name}** (Fora) jogou recentemente — "
                f"Último jogo: {m.away_team.last_match_date or 'N/D'} | "
                f"Partida: vs {m.home_team.team_name} ({m.league_name})"
            )

    if fatigue_alerts:
        for alert in fatigue_alerts:
            lines.append(alert)
    else:
        lines.append("*Nenhum alerta de fadiga identificado.*")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ═══════════════════════════════════════════════
    # METODOLOGIA
    # ═══════════════════════════════════════════════
    lines.append("## 🧪 Metodologia")
    lines.append("")
    lines.append("| Componente | Tecnologia/Método | Justificativa |")
    lines.append("|------------|-------------------|---------------|")
    lines.append("| Linguagem | Python 3.9+ | Ecossistema dominante em Data Science |")
    lines.append("| IDE | Cursor (Agent Mode) | Orquestração autônoma e multi-arquivo |")
    lines.append("| Modelagem Gols | Dixon-Coles (Poisson Bivariada) | Correção de interdependência em placares baixos |")
    lines.append("| Modelagem Props | Regressão Binomial Negativa | Tratamento da sobredispersão em cartões/escanteios |")
    lines.append("| Simulação | Monte Carlo (5.000 iter.) | Distribuição empírica robusta |")
    lines.append("| De-Vigging | Power Method | Remoção precisa da margem (viés favorito-zebra) |")
    lines.append("| Gestão de Risco | Kelly Fracionário (1/4) | Otimização de crescimento com proteção de capital |")
    lines.append("| Contexto | Clima + Lesões + Fadiga + Urgência | Alfa exógeno não capturado por modelos puramente estatísticos |")
    lines.append("")
    lines.append("### Fórmulas Principais")
    lines.append("")
    lines.append("**Dixon-Coles:**")
    lines.append("```")
    lines.append("P(x,y) = τ(x,y,λ,μ,ρ) × Poisson(x;λ) × Poisson(y;μ)")
    lines.append("```")
    lines.append("")
    lines.append("**Valor Esperado (EV):**")
    lines.append("```")
    lines.append("EV = (P_modelo × Odd_decimal) - 1")
    lines.append("```")
    lines.append("")
    lines.append("**Kelly Fracionário:**")
    lines.append("```")
    lines.append("f* = [(p × (b+1) - 1) / b] × 0.25")
    lines.append("```")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## ⚠️ Disclaimer")
    lines.append("")
    lines.append("Este relatório é gerado por um sistema de análise quantitativa automatizada. "
                "As probabilidades e sugestões são baseadas em modelos matemáticos e dados "
                "disponíveis no momento da geração. **Nenhuma previsão é garantia de resultado.** "
                "Gestão de risco e disciplina financeira são essenciais. "
                "Aposte apenas o que pode perder.")
    lines.append("")
    lines.append("---")
    lines.append(f"*Relatório gerado automaticamente pela Engine de Análise Preditiva v1.0 — {now}*")

    return "\n".join(lines)


def save_report(content: str, path: str = None) -> str:
    """
    Salva o relatório em arquivo Markdown.
    """
    if path is None:
        path = config.REPORT_OUTPUT_PATH

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"[REPORT] Relatório salvo em: {path}")
    return path
