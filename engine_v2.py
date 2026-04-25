"""
═══════════════════════════════════════════════════════════════════════
  EDGE ENGINE V2 — ApostasIA Pro
═══════════════════════════════════════════════════════════════════════
  Camada de inteligência avançada que ENRIQUECE os dados produzidos
  pela v1 (matches + opportunities já serializadas em cache/Supabase)
  com métricas de classe mundial calibradas para identificar
  oportunidades de valor com MUITO mais assertividade.

  Não substitui a v1: consome os mesmos dados, sem custar API calls.

  ─── Métricas implementadas ───
  • Market Efficiency Score (MES)         - dispersão entre bookmakers
  • Sharp Disagreement (SD)               - delta vs mediana sharp
  • Devigged Fair Probability             - prob justa via power method
  • Composite Confidence Index 2.0 (CCI)  - 7 dimensões ponderadas
  • EV Ajustado (EVA)                     - edge × calibração × qualidade
  • Kelly Smart                           - Kelly fracional + qualidade
  • Risk-Adjusted Score (RAS)             - EVA / variância estimada
  • Opus Score                            - ranking principal unificado
  • Asian Handicap value scan             - via Skellam aproximada
  • Calibration buckets                   - winrate por faixa de prob
  • Portfolio metrics                     - correlação intra-jogo, hedge
═══════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any


# ════════════════════════════════════════════════════
# UTILITÁRIOS
# ════════════════════════════════════════════════════
def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b else default


def _pct(x: float) -> float:
    """Garante valor em [0, 100] para entradas que podem vir em fração."""
    if x is None:
        return 0.0
    if abs(x) <= 1.5:  # provavelmente fração
        return x * 100
    return x


def _frac(x: float) -> float:
    """Converte para fração (0–1) se vier em %."""
    if x is None:
        return 0.0
    if abs(x) > 1.5:
        return x / 100
    return x


# ════════════════════════════════════════════════════
# 1) DEVIG — POWER METHOD
# ════════════════════════════════════════════════════
def power_devig(odds: list[float], iters: int = 80) -> list[float]:
    """Remove vig via power method. Retorna probs justas.
    Mais robusto que basic devig (1/odd / soma)."""
    odds = [o for o in odds if o and o > 1.01]
    if len(odds) < 2:
        return []
    raws = [1.0 / o for o in odds]
    s = sum(raws)
    if s <= 0:
        return []
    # busca k tal que sum(raw**k) = 1
    lo, hi = 0.5, 1.5
    for _ in range(iters):
        k = (lo + hi) / 2
        total = sum(r ** k for r in raws)
        if total > 1:
            lo = k
        else:
            hi = k
    k = (lo + hi) / 2
    return [r ** k for r in raws]


# ════════════════════════════════════════════════════
# 2) MARKET EFFICIENCY SCORE
# ════════════════════════════════════════════════════
def market_efficiency(book_odds: list[float]) -> dict:
    """Mede eficiência do mercado para a seleção.
    Quanto MAIOR o score, mais 'precificado' está (menos ruidoso)."""
    odds = [float(o) for o in book_odds if o and o > 1.01]
    if not odds:
        return {"median": 0, "best": 0, "mean": 0, "n_books": 0, "cv": 0, "score": 0}
    if len(odds) == 1:
        o = odds[0]
        return {"median": round(o, 3), "best": round(o, 3), "mean": round(o, 3),
                "n_books": 1, "cv": 0, "score": 35}
    med = statistics.median(odds)
    best = max(odds)
    mean = statistics.mean(odds)
    std = statistics.pstdev(odds)
    cv = std / mean if mean else 0
    # eficiência alta ⇨ baixa dispersão. Calibrado: cv 0.005 ≈ 95, 0.05 ≈ 25
    score = _clip(100 - cv * 1500, 5, 100)
    return {
        "median": round(med, 3),
        "best": round(best, 3),
        "mean": round(mean, 3),
        "n_books": len(odds),
        "cv": round(cv, 4),
        "score": round(score, 1),
    }


# ════════════════════════════════════════════════════
# 3) SHARP DISAGREEMENT
# ════════════════════════════════════════════════════
def sharp_disagreement(model_prob: float, median_odd: float) -> dict:
    """Quanto o nosso modelo discorda do mercado mediano.
    Positivo = modelo mais otimista (potencial overlay)."""
    p = _frac(model_prob)
    if median_odd <= 1.01 or p <= 0:
        return {"delta_pp": 0, "ratio": 1.0}
    implied = 1.0 / median_odd
    delta_pp = (p - implied) * 100
    ratio = p / implied if implied else 1.0
    return {"delta_pp": round(delta_pp, 2), "ratio": round(ratio, 3)}


# ════════════════════════════════════════════════════
# 4) KELLY SMART
# ════════════════════════════════════════════════════
def kelly_smart(p: float, odd: float, fraction: float = 0.20,
                cap: float = 0.04, quality: float = 0.7,
                cci: float = 60.0) -> dict:
    """Kelly fracionário com ajuste por qualidade e CCI.
    fraction=0.20 (1/5) é mais conservador que v1 (1/4)."""
    p = _frac(p)
    if odd <= 1.01 or p <= 0:
        return {"raw": 0, "adjusted": 0, "stake_pct": 0, "kelly_full": 0}
    b = odd - 1
    q = 1 - p
    full_kelly = (p * b - q) / b
    raw = max(0.0, full_kelly)
    quality_factor = _clip(quality, 0.3, 1.0)
    cci_factor = _clip(cci / 100.0, 0.4, 1.0)
    adj = raw * fraction * quality_factor * cci_factor
    adj = _clip(adj, 0, cap)
    return {
        "raw": round(raw, 4),
        "adjusted": round(adj, 4),
        "stake_pct": round(adj * 100, 2),
        "kelly_full": round(full_kelly, 4),
    }


# ════════════════════════════════════════════════════
# 5) COMPOSITE CONFIDENCE INDEX 2.0
# ════════════════════════════════════════════════════
def composite_confidence_v2(
    edge: float,
    odd: float,
    model_prob: float,
    market_score: float,
    sharp_delta: float,
    data_quality: float,
    league_tier: str = "C",
    history_hit_rate: float | None = None,
    implied_prob: float | None = None,
) -> dict:
    """Composite Confidence Index 2.0 — 7 dimensões calibradas.
    Aprendizados aplicados:
      • Edges 4-9% têm o melhor ROI histórico → recompensa essa faixa
      • Odds 1.55-2.30 são as mais previsíveis
      • Edge muito alto + odd alta = ARMADILHA (penalidade severa)
      • Mercados eficientes (sharp) com edge sólido > mercados ruidosos
      • Concordância com sharp = sinal forte de overlay legítimo
    """
    edge_f = _frac(edge)
    p = _frac(model_prob)

    # ── Dim 1: Edge (faixa-ouro 4-9%) ──
    if edge_f <= 0:
        edge_s = 0
    elif edge_f < 0.03:
        edge_s = edge_f * 1000  # 0-30
    elif edge_f <= 0.05:
        edge_s = 60 + (edge_f - 0.03) * 500  # 60-70
    elif edge_f <= 0.09:
        edge_s = 70 + (edge_f - 0.05) * 750  # 70-100
    elif edge_f <= 0.15:
        edge_s = 100 - (edge_f - 0.09) * 333  # 100→80
    else:
        edge_s = max(20, 80 - (edge_f - 0.15) * 200)

    # ── Dim 2: Odd (faixa-ouro 1.55-2.30) ──
    if odd <= 1.20:
        odd_s = 35
    elif 1.20 < odd <= 1.55:
        odd_s = 50 + (odd - 1.20) * 100
    elif 1.55 < odd <= 2.30:
        odd_s = 95
    elif 2.30 < odd <= 3.00:
        odd_s = 95 - (odd - 2.30) * 35
    elif 3.00 < odd <= 4.50:
        odd_s = 75 - (odd - 3.00) * 20
    else:
        odd_s = max(15, 45 - (odd - 4.50) * 5)

    # ── Dim 3: Probabilidade do modelo ──
    if 0.45 <= p <= 0.65:
        model_s = 95
    elif 0.30 <= p < 0.45 or 0.65 < p <= 0.78:
        model_s = 80 - abs(p - 0.55) * 100
    elif 0.20 <= p < 0.30 or 0.78 < p <= 0.88:
        model_s = 55
    else:
        model_s = 30

    # ── Dim 4: Sharp disagreement (overlay zone) ──
    if sharp_delta <= 0:
        sharp_s = 25
    elif sharp_delta <= 3:
        sharp_s = 50 + sharp_delta * 8
    elif sharp_delta <= 10:
        sharp_s = 75 + (sharp_delta - 3) * 3.5
    elif sharp_delta <= 18:
        sharp_s = 100 - (sharp_delta - 10) * 2
    else:
        sharp_s = max(35, 84 - (sharp_delta - 18) * 4)

    # ── Dim 5: Market Efficiency ──
    market_s = _clip(market_score, 0, 100)

    # ── Dim 6: Data Quality + Tier ──
    dq = _clip(data_quality if data_quality and data_quality <= 1 else (data_quality or 0) / 100, 0, 1)
    tier_bonus = {"S": 8, "A": 5, "B": 0, "C": -5}.get(league_tier, 0)
    quality_s = _clip(dq * 100 + tier_bonus, 0, 100)

    # ── Dim 7: Histórico de calibração ──
    if history_hit_rate is not None and implied_prob is not None:
        imp = _frac(implied_prob)
        delta = (history_hit_rate - imp) * 100
        history_s = _clip(50 + delta * 4, 0, 100)
        has_hist = True
    else:
        history_s = 50
        has_hist = False

    weights = {
        "edge":    0.18,
        "odd":     0.10,
        "model":   0.13,
        "sharp":   0.22,
        "market":  0.13,
        "quality": 0.12,
        "history": 0.12,
    }
    cci = (
        edge_s * weights["edge"]
        + odd_s * weights["odd"]
        + model_s * weights["model"]
        + sharp_s * weights["sharp"]
        + market_s * weights["market"]
        + quality_s * weights["quality"]
        + history_s * weights["history"]
    )
    cci = _clip(cci, 0, 100)

    # Penalidades de SEGURANÇA — armadilhas detectadas em v1
    if edge_f > 0.20 and odd > 4.0:
        cci *= 0.55  # edge fantasma
    if dq < 0.4:
        cci *= 0.85
    if sharp_delta < -2:
        cci *= 0.75  # mercado discorda fortemente
    if odd > 6.0 and edge_f < 0.05:
        cci *= 0.7

    cci = round(cci, 1)
    label = "ELITE" if cci >= 78 else "ALTO" if cci >= 64 else "MÉDIO" if cci >= 48 else "BAIXO"

    return {
        "cci": cci,
        "label": label,
        "has_history": has_hist,
        "breakdown": {
            "edge":    round(edge_s, 1),
            "odd":     round(odd_s, 1),
            "model":   round(model_s, 1),
            "sharp":   round(sharp_s, 1),
            "market":  round(market_s, 1),
            "quality": round(quality_s, 1),
            "history": round(history_s, 1),
        },
        "weights": weights,
    }


# ════════════════════════════════════════════════════
# 6) EV AJUSTADO + RISK-ADJUSTED + OPUS SCORE
# ════════════════════════════════════════════════════
def ev_adjusted(edge_pct: float, cci: float, quality: float,
                sharp_delta: float, n_books: int) -> float:
    """Edge ajustado por confiança composta, qualidade e profundidade do mercado."""
    e = _frac(edge_pct)
    cci_f = _clip(cci / 100, 0.2, 1.0)
    q = _clip(quality if quality <= 1 else quality / 100, 0.3, 1.0)
    sharp_f = _clip(0.85 + sharp_delta * 0.012, 0.6, 1.25)
    book_f = _clip(0.6 + n_books * 0.08, 0.6, 1.15)
    return round(e * cci_f * q * sharp_f * book_f * 100, 2)


def risk_adjusted_score(eva: float, p: float, odd: float) -> float:
    """RAS = EVA / sqrt(variância de Bernoulli). Sharpe-like."""
    p = _frac(p)
    if odd <= 1 or p <= 0 or p >= 1:
        return 0
    var = p * (odd - 1) ** 2 + (1 - p) * 1
    sd = math.sqrt(max(var, 1e-6))
    return round(eva / sd * 10, 2)


def opus_score(cci: float, eva: float, ras: float, market_score: float,
               league_tier: str) -> float:
    """Ranking unificado FINAL — guia mestre da v2."""
    tier_w = {"S": 1.10, "A": 1.05, "B": 1.0, "C": 0.92}.get(league_tier, 0.95)
    base = (cci * 0.45) + (min(eva, 25) * 1.4) + (min(ras, 30) * 0.5) + (market_score * 0.10)
    return round(_clip(base * tier_w, 0, 100), 1)


# ════════════════════════════════════════════════════
# 7) EXTRAÇÃO DE BOOKMAKER ODDS DO all_markets
# ════════════════════════════════════════════════════
def extract_book_odds_for_selection(all_markets: dict, market: str,
                                    selection: str) -> list[float]:
    """Extrai lista de odds por bookmaker para uma seleção específica."""
    if not all_markets or not isinstance(all_markets, dict):
        return []
    candidates = []
    for mk_key, mk_data in all_markets.items():
        if not isinstance(mk_data, dict):
            continue
        if not _market_matches(mk_key, market):
            continue
        bookmakers = mk_data.get("_bookmakers") or {}
        if isinstance(bookmakers, dict):
            for _book, sel_dict in bookmakers.items():
                if not isinstance(sel_dict, dict):
                    continue
                for sel_name, odd_val in sel_dict.items():
                    if _selection_matches(sel_name, selection):
                        try:
                            v = float(odd_val)
                            if 1.01 < v < 999:
                                candidates.append(v)
                        except (TypeError, ValueError):
                            continue
        else:
            for sel_name, odd_val in mk_data.items():
                if sel_name.startswith("_"):
                    continue
                if _selection_matches(sel_name, selection):
                    try:
                        v = float(odd_val)
                        if 1.01 < v < 999:
                            candidates.append(v)
                    except (TypeError, ValueError):
                        continue
    return candidates


def _norm(s: str) -> str:
    return (s or "").lower().replace(" ", "").replace("-", "").replace("_", "")


def _market_matches(mk_key: str, market_label: str) -> bool:
    a, b = _norm(mk_key), _norm(market_label)
    if not a or not b:
        return False
    if a == b or a in b or b in a:
        return True
    syn = {
        "matchwinner": ["1x2", "fulltimeresult"],
        "goalsovrunder": ["overunder", "totalgoals", "goalsoverunder"],
        "goalsoverunder": ["overunder", "totalgoals"],
        "bothteamstoscore": ["btts"],
    }
    for k, vs in syn.items():
        if (k in a and any(v in b for v in vs)) or (k in b and any(v in a for v in vs)):
            return True
    return False


def _selection_matches(sel_name: str, selection: str) -> bool:
    a, b = _norm(sel_name), _norm(selection)
    return bool(a) and bool(b) and (a == b or a in b or b in a)


# ════════════════════════════════════════════════════
# 8) HISTORICAL CALIBRATION (a partir de opps settled)
# ════════════════════════════════════════════════════
def build_calibration_index(settled_opps: list[dict]) -> dict:
    """A partir de oportunidades já liquidadas (GREEN/RED/VOID),
    cria índices de hit rate por:
      • mercado
      • liga
      • faixa de implied probability (10 buckets)
      • faixa de edge
      • combinação mercado × tier
    Retornado como dict consultável."""
    by_market = defaultdict(lambda: [0, 0])
    by_league = defaultdict(lambda: [0, 0])
    by_tier = defaultdict(lambda: [0, 0])
    by_imp_bucket = defaultdict(lambda: [0, 0])
    by_edge_bucket = defaultdict(lambda: [0, 0])
    by_market_tier = defaultdict(lambda: [0, 0])
    reliability = defaultdict(lambda: {"n": 0, "wins": 0, "stake": 0, "ret": 0})

    for o in settled_opps:
        status = o.get("result_status", "PENDENTE")
        if status not in ("GREEN", "RED"):
            continue
        win = 1 if status == "GREEN" else 0
        market = o.get("market", "?")
        league = o.get("league_name", "?")
        tier = o.get("league_tier") or "C"
        imp = _frac(o.get("implied_prob", 0))
        edge = _frac(o.get("edge", 0))
        odd = o.get("market_odd", 1.0) or 1.0

        by_market[market][0] += win
        by_market[market][1] += 1
        by_league[league][0] += win
        by_league[league][1] += 1
        by_tier[tier][0] += win
        by_tier[tier][1] += 1
        by_market_tier[f"{market}|{tier}"][0] += win
        by_market_tier[f"{market}|{tier}"][1] += 1

        ib = min(9, max(0, int(imp * 10)))
        eb_key = ("neg" if edge <= 0 else
                  "0-3" if edge <= 0.03 else
                  "3-6" if edge <= 0.06 else
                  "6-10" if edge <= 0.10 else
                  "10-15" if edge <= 0.15 else
                  "15+")
        by_imp_bucket[ib][0] += win
        by_imp_bucket[ib][1] += 1
        by_edge_bucket[eb_key][0] += win
        by_edge_bucket[eb_key][1] += 1

        rel = reliability[ib]
        rel["n"] += 1
        rel["wins"] += win
        rel["stake"] += 1
        rel["ret"] += odd if win else 0

    def _rate(d):
        return {k: {"hit_rate": round(v[0] / v[1], 4), "n": v[1]}
                for k, v in d.items() if v[1] >= 3}

    return {
        "by_market": _rate(by_market),
        "by_league": _rate(by_league),
        "by_tier": _rate(by_tier),
        "by_implied_bucket": _rate(by_imp_bucket),
        "by_edge_bucket": _rate(by_edge_bucket),
        "by_market_tier": _rate(by_market_tier),
        "reliability_diagram": [
            {
                "bucket": i,
                "midpoint": round(0.05 + i * 0.10, 2),
                "n": reliability[i]["n"],
                "hit_rate": round(reliability[i]["wins"] / reliability[i]["n"], 4) if reliability[i]["n"] else None,
                "roi": round((reliability[i]["ret"] - reliability[i]["stake"]) / reliability[i]["stake"], 4) if reliability[i]["stake"] else None,
            }
            for i in range(10)
        ],
        "total_settled": sum(v[1] for v in by_market.values()),
    }


def lookup_history_hit_rate(calib: dict, market: str, tier: str,
                            implied_prob: float) -> float | None:
    """Lookup com fallback: market×tier → market → implied_bucket → tier."""
    if not calib:
        return None
    key = f"{market}|{tier}"
    mt = calib.get("by_market_tier", {}).get(key)
    if mt and mt["n"] >= 8:
        return mt["hit_rate"]
    m = calib.get("by_market", {}).get(market)
    if m and m["n"] >= 10:
        return m["hit_rate"]
    ib = min(9, max(0, int(_frac(implied_prob) * 10)))
    bk = calib.get("by_implied_bucket", {}).get(ib)
    if bk and bk["n"] >= 15:
        return bk["hit_rate"]
    return None


# ════════════════════════════════════════════════════
# 9) ASIAN HANDICAP via Skellam aproximada
# ════════════════════════════════════════════════════
def _skellam_pmf(k: int, mu1: float, mu2: float, max_iter: int = 30) -> float:
    """Probabilidade de diferença = k para X1~Poi(mu1), X2~Poi(mu2)."""
    if mu1 <= 0 or mu2 <= 0:
        return 0
    # P(X-Y=k) = exp(-mu1-mu2) * (mu1/mu2)^(k/2) * I_|k|(2*sqrt(mu1*mu2))
    z = 2 * math.sqrt(mu1 * mu2)
    s = 0
    abs_k = abs(k)
    for n in range(max_iter):
        try:
            term = (z / 2) ** (2 * n + abs_k) / (math.factorial(n) * math.factorial(n + abs_k))
        except (OverflowError, ValueError):
            break
        s += term
    bessel = s
    return math.exp(-(mu1 + mu2)) * (mu1 / mu2) ** (k / 2) * bessel


def asian_handicap_probs(home_xg: float, away_xg: float, line: float) -> dict:
    """Calcula probabilidades de cobertura para AH dado xG."""
    if home_xg <= 0 or away_xg <= 0:
        return {"home_cover": 0, "away_cover": 0, "push": 0}
    diffs = list(range(-8, 9))
    pmf = {d: _skellam_pmf(d, home_xg, away_xg) for d in diffs}
    s = sum(pmf.values())
    if s > 0:
        pmf = {d: v / s for d, v in pmf.items()}
    home_cover, away_cover, push = 0, 0, 0
    for d, p in pmf.items():
        margin = d + line
        if abs(margin) < 0.001:
            push += p
        elif margin > 0:
            home_cover += p
        else:
            away_cover += p
    return {
        "home_cover": round(home_cover, 4),
        "away_cover": round(away_cover, 4),
        "push": round(push, 4),
    }


# ════════════════════════════════════════════════════
# 10) ENRIQUECIMENTO PRINCIPAL
# ════════════════════════════════════════════════════
def enrich_opportunity(
    opp: dict,
    match: dict | None,
    calib: dict | None = None,
    risk_profile: str = "balanced",
) -> dict:
    """Aplica TODAS as métricas v2 a uma oportunidade da v1.
    Retorna nova dict com chaves originais + chaves v2_*."""
    enriched = dict(opp)
    market = opp.get("market", "")
    selection = opp.get("selection", "")
    market_odd = opp.get("market_odd") or 0
    model_p = _frac(opp.get("model_prob", 0))
    edge_pct = opp.get("edge", 0)
    league_tier = opp.get("league_tier") or "C"

    # ─── Bookmaker depth & efficiency ───
    book_odds = []
    if match:
        book_odds = extract_book_odds_for_selection(
            match.get("all_markets") or {}, market, selection
        )
    if market_odd and market_odd not in book_odds:
        book_odds.append(market_odd)
    mes = market_efficiency(book_odds)

    # ─── Sharp Disagreement ───
    sd = sharp_disagreement(model_p, mes["median"] or market_odd)

    # ─── Histórico calibração ───
    history_hit = lookup_history_hit_rate(
        calib or {}, market, league_tier, opp.get("implied_prob", 0)
    )

    # ─── CCI 2.0 ───
    cci_data = composite_confidence_v2(
        edge=edge_pct,
        odd=market_odd,
        model_prob=model_p,
        market_score=mes["score"],
        sharp_delta=sd["delta_pp"],
        data_quality=opp.get("data_quality", 0),
        league_tier=league_tier,
        history_hit_rate=history_hit,
        implied_prob=opp.get("implied_prob", 0),
    )

    # ─── EV ajustado ───
    eva = ev_adjusted(
        edge_pct=edge_pct,
        cci=cci_data["cci"],
        quality=opp.get("data_quality", 0),
        sharp_delta=sd["delta_pp"],
        n_books=mes["n_books"],
    )

    # ─── Risk-Adjusted ───
    ras = risk_adjusted_score(eva, model_p, market_odd)

    # ─── Kelly Smart ───
    profile_caps = {"conservative": 0.020, "balanced": 0.040, "aggressive": 0.075}
    profile_fracs = {"conservative": 0.10, "balanced": 0.20, "aggressive": 0.35}
    cap = profile_caps.get(risk_profile, 0.040)
    frac = profile_fracs.get(risk_profile, 0.20)
    kelly = kelly_smart(
        p=model_p, odd=market_odd, fraction=frac, cap=cap,
        quality=opp.get("data_quality", 0.7),
        cci=cci_data["cci"],
    )

    # ─── Opus Score ───
    opus = opus_score(
        cci=cci_data["cci"], eva=eva, ras=ras,
        market_score=mes["score"], league_tier=league_tier,
    )

    # ─── Tags v2 ───
    tags = []
    if cci_data["label"] == "ELITE":
        tags.append("ELITE")
    if eva >= 6:
        tags.append("HIGH_EV")
    if mes["score"] >= 75 and edge_pct >= 4:
        tags.append("SHARP_OVERLAY")
    if mes["n_books"] >= 5:
        tags.append("DEEP_MARKET")
    if sd["delta_pp"] >= 8:
        tags.append("CONTRARIAN")
    if opp.get("bet365_available"):
        tags.append("BET365")
    if league_tier in ("S", "A"):
        tags.append("TOP_LEAGUE")
    if 1.55 <= market_odd <= 2.30:
        tags.append("GOLDEN_ODDS")
    if opp.get("data_quality", 0) >= 0.85:
        tags.append("PREMIUM_DATA")
    if history_hit is not None and opp.get("implied_prob", 0):
        if history_hit > _frac(opp.get("implied_prob", 0)) + 0.05:
            tags.append("HISTORY_BACKED")
    if (edge_pct >= 15 and market_odd >= 4.0) or opp.get("odds_suspect"):
        tags.append("RISKY")

    enriched["v2"] = {
        "opus_score": opus,
        "cci": cci_data,
        "eva": eva,
        "ras": ras,
        "kelly_smart": kelly,
        "market_efficiency": mes,
        "sharp_disagreement": sd,
        "history_hit_rate": history_hit,
        "tags": tags,
        "book_odds": [round(o, 3) for o in sorted(book_odds, reverse=True)],
        "n_books": mes["n_books"],
    }
    return enriched


# ════════════════════════════════════════════════════
# 11) AGREGAÇÕES PARA INSIGHTS
# ════════════════════════════════════════════════════
def aggregate_insights(opps_v2: list[dict]) -> dict:
    """Agregações para o Dashboard v2."""
    if not opps_v2:
        return {"total": 0, "by_label": {}, "by_market": [], "by_league": [],
                "by_tier": [], "top_tags": [], "opus_distribution": []}

    by_label = defaultdict(int)
    by_market = defaultdict(lambda: {"n": 0, "avg_eva": 0, "avg_opus": 0, "avg_edge": 0})
    by_league = defaultdict(lambda: {"n": 0, "avg_eva": 0, "avg_opus": 0, "country": ""})
    by_tier = defaultdict(lambda: {"n": 0, "avg_opus": 0, "avg_eva": 0})
    tag_counts = defaultdict(int)
    opus_buckets = [0] * 10  # 10-deciles 0-100

    for o in opps_v2:
        v2 = o.get("v2") or {}
        by_label[v2.get("cci", {}).get("label", "?")] += 1
        m = o.get("market", "?")
        by_market[m]["n"] += 1
        by_market[m]["avg_eva"] += v2.get("eva", 0)
        by_market[m]["avg_opus"] += v2.get("opus_score", 0)
        by_market[m]["avg_edge"] += o.get("edge", 0)
        lg = o.get("league_name", "?")
        by_league[lg]["n"] += 1
        by_league[lg]["avg_eva"] += v2.get("eva", 0)
        by_league[lg]["avg_opus"] += v2.get("opus_score", 0)
        by_league[lg]["country"] = o.get("league_country", "")
        t = o.get("league_tier", "C")
        by_tier[t]["n"] += 1
        by_tier[t]["avg_opus"] += v2.get("opus_score", 0)
        by_tier[t]["avg_eva"] += v2.get("eva", 0)
        for tg in v2.get("tags", []):
            tag_counts[tg] += 1
        opus = v2.get("opus_score", 0)
        opus_buckets[min(9, int(opus / 10))] += 1

    def _finalize(d, n_keys=("avg_eva", "avg_opus", "avg_edge")):
        out = []
        for k, v in d.items():
            if v["n"] == 0:
                continue
            for nk in n_keys:
                if nk in v:
                    v[nk] = round(v[nk] / v["n"], 2)
            out.append({"key": k, **v})
        return sorted(out, key=lambda x: x.get("avg_opus", 0), reverse=True)

    return {
        "total": len(opps_v2),
        "by_label": dict(by_label),
        "by_market": _finalize(by_market)[:25],
        "by_league": _finalize(by_league)[:30],
        "by_tier": _finalize(by_tier, n_keys=("avg_eva", "avg_opus")),
        "top_tags": sorted(
            [{"tag": k, "count": v} for k, v in tag_counts.items()],
            key=lambda x: x["count"], reverse=True,
        )[:20],
        "opus_distribution": opus_buckets,
    }


# ════════════════════════════════════════════════════
# 12) BANKROLL SIMULATION (Monte Carlo)
# ════════════════════════════════════════════════════
def simulate_bankroll(opps_v2: list[dict], starting_bank: float = 1000,
                      n_simulations: int = 500, max_picks: int = 100,
                      seed: int = 42) -> dict:
    """Monte Carlo: aplica os picks v2 em ordem (ranqueados por Opus Score),
    cada um com stake = kelly_adjusted * bank_atual.
    Retorna distribuição final de banca, drawdown médio e Sharpe."""
    import random
    rng = random.Random(seed)

    picks = sorted(
        [o for o in opps_v2 if o.get("v2") and o["v2"].get("opus_score", 0) >= 50],
        key=lambda o: o["v2"]["opus_score"],
        reverse=True,
    )[:max_picks]

    if not picks:
        return {"n_picks": 0, "starting_bank": starting_bank}

    finals, max_dd_list, paths_avg = [], [], [0] * (len(picks) + 1)
    for sim in range(n_simulations):
        bank = starting_bank
        peak = bank
        max_dd = 0
        path = [bank]
        for pk in picks:
            v2 = pk["v2"]
            stake_pct = v2["kelly_smart"]["adjusted"]
            odd = pk.get("market_odd", 1.0)
            p = _frac(pk.get("model_prob", 0))
            stake = bank * stake_pct
            if rng.random() < p:
                bank += stake * (odd - 1)
            else:
                bank -= stake
            peak = max(peak, bank)
            dd = (peak - bank) / peak if peak else 0
            max_dd = max(max_dd, dd)
            path.append(bank)
        finals.append(bank)
        max_dd_list.append(max_dd)
        for i, v in enumerate(path):
            paths_avg[i] += v

    paths_avg = [v / n_simulations for v in paths_avg]
    finals.sort()
    p10 = finals[int(0.10 * n_simulations)]
    p50 = finals[int(0.50 * n_simulations)]
    p90 = finals[int(0.90 * n_simulations)]
    mean_final = statistics.mean(finals)
    std_final = statistics.pstdev(finals) or 1
    sharpe = (mean_final - starting_bank) / std_final
    avg_dd = statistics.mean(max_dd_list)

    return {
        "n_picks": len(picks),
        "n_simulations": n_simulations,
        "starting_bank": starting_bank,
        "mean_final": round(mean_final, 2),
        "median_final": round(p50, 2),
        "p10_final": round(p10, 2),
        "p90_final": round(p90, 2),
        "best_case": round(max(finals), 2),
        "worst_case": round(min(finals), 2),
        "expected_roi_pct": round((mean_final - starting_bank) / starting_bank * 100, 2),
        "sharpe": round(sharpe, 3),
        "avg_max_drawdown_pct": round(avg_dd * 100, 2),
        "expected_path": [round(v, 2) for v in paths_avg],
    }


# ════════════════════════════════════════════════════
# 13) PIPELINE PRINCIPAL
# ════════════════════════════════════════════════════
def run_v2_enrichment(
    opportunities: list[dict],
    matches: list[dict],
    settled_history: list[dict] | None = None,
    risk_profile: str = "balanced",
) -> dict:
    """Pipeline completo da v2.
    Args:
      opportunities: lista de opps serializadas pela v1
      matches: lista de matches serializados pela v1 (para extrair all_markets)
      settled_history: opps históricas com result_status para calibração
      risk_profile: 'conservative' | 'balanced' | 'aggressive'
    Returns:
      {
        "opportunities": [opps enriquecidas],
        "insights": {agregações},
        "calibration": {índice histórico},
        "bankroll_sim": {simulação},
        "meta": {...}
      }
    """
    matches_by_id = {str(m.get("match_id")): m for m in matches}

    calib = build_calibration_index(settled_history or []) if settled_history else {}

    enriched = []
    for o in opportunities:
        m = matches_by_id.get(str(o.get("match_id")))
        try:
            enriched.append(enrich_opportunity(o, m, calib, risk_profile))
        except Exception as e:
            o2 = dict(o)
            o2["v2_error"] = str(e)
            enriched.append(o2)

    enriched.sort(key=lambda x: (x.get("v2") or {}).get("opus_score", 0), reverse=True)

    insights = aggregate_insights(enriched)
    bankroll = simulate_bankroll(enriched)

    return {
        "opportunities": enriched,
        "insights": insights,
        "calibration": calib,
        "bankroll_sim": bankroll,
        "meta": {
            "risk_profile": risk_profile,
            "n_opportunities": len(enriched),
            "n_matches": len(matches),
            "n_settled_history": len(settled_history or []),
            "engine_version": "v2.0.0",
        },
    }
