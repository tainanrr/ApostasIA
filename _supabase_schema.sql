
-- ═══════════════════════════════════════════════════════
-- ApostasIA — Schema do Banco de Dados
-- Execute este script no SQL Editor do Supabase
-- ═══════════════════════════════════════════════════════

-- 1. EXECUÇÕES DO PIPELINE
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    executed_at TIMESTAMPTZ DEFAULT NOW(),
    analysis_dates TEXT[] NOT NULL,
    total_matches INTEGER NOT NULL,
    total_leagues INTEGER NOT NULL,
    total_opportunities INTEGER NOT NULL,
    high_conf INTEGER DEFAULT 0,
    med_conf INTEGER DEFAULT 0,
    low_conf INTEGER DEFAULT 0,
    avg_edge NUMERIC(6,2),
    max_edge NUMERIC(6,2),
    run_time_seconds NUMERIC(8,2),
    api_calls_used INTEGER DEFAULT 0,
    mode TEXT DEFAULT 'API Real'
);

-- 2. OPORTUNIDADES IDENTIFICADAS
CREATE TABLE IF NOT EXISTS opportunities (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    run_id UUID REFERENCES pipeline_runs(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    match_id INTEGER NOT NULL,
    league_name TEXT,
    league_country TEXT,
    match_date DATE NOT NULL,
    match_time TEXT,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    venue TEXT,
    market TEXT NOT NULL,
    selection TEXT NOT NULL,
    bookmaker TEXT,
    market_odd NUMERIC(6,2) NOT NULL,
    fair_odd NUMERIC(6,2),
    model_prob NUMERIC(6,4),
    implied_prob NUMERIC(6,4),
    edge NUMERIC(6,4),
    edge_pct TEXT,
    kelly_fraction NUMERIC(6,4),
    kelly_bet_pct TEXT,
    confidence TEXT,
    reasoning TEXT,
    home_xg NUMERIC(4,2),
    away_xg NUMERIC(4,2),
    weather_note TEXT,
    fatigue_note TEXT,
    urgency_home NUMERIC(3,1),
    urgency_away NUMERIC(3,1),
    result_status TEXT DEFAULT 'PENDENTE',
    result_score TEXT,
    result_updated_at TIMESTAMPTZ,
    bet_placed BOOLEAN DEFAULT FALSE,
    bet_amount NUMERIC(10,2),
    bet_return NUMERIC(10,2),
    bet_profit NUMERIC(10,2),
    bet_notes TEXT
);

-- 3. PARTIDAS ANALISADAS
CREATE TABLE IF NOT EXISTS matches (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    run_id UUID REFERENCES pipeline_runs(id) ON DELETE CASCADE,
    match_id INTEGER NOT NULL,
    league_name TEXT,
    league_country TEXT,
    match_date DATE,
    match_time TEXT,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    home_xg NUMERIC(4,2),
    away_xg NUMERIC(4,2),
    prob_home NUMERIC(5,1),
    prob_draw NUMERIC(5,1),
    prob_away NUMERIC(5,1),
    prob_over25 NUMERIC(5,1),
    prob_btts NUMERIC(5,1),
    corners_expected NUMERIC(4,1),
    cards_expected NUMERIC(4,1),
    odds_home NUMERIC(6,2),
    odds_draw NUMERIC(6,2),
    odds_away NUMERIC(6,2),
    bookmaker TEXT,
    weather_temp NUMERIC(4,1),
    weather_wind NUMERIC(4,1),
    weather_desc TEXT,
    venue TEXT,
    referee TEXT,
    result_score TEXT,
    result_updated_at TIMESTAMPTZ
);

-- 4. ÍNDICES
CREATE INDEX IF NOT EXISTS idx_opportunities_match_date ON opportunities(match_date);
CREATE INDEX IF NOT EXISTS idx_opportunities_confidence ON opportunities(confidence);
CREATE INDEX IF NOT EXISTS idx_opportunities_result ON opportunities(result_status);
CREATE INDEX IF NOT EXISTS idx_opportunities_run ON opportunities(run_id);
CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(match_date);
CREATE INDEX IF NOT EXISTS idx_matches_run ON matches(run_id);

-- 5. VIEW: Performance semanal
CREATE OR REPLACE VIEW v_performance_summary AS
SELECT 
    DATE_TRUNC('week', match_date) AS semana,
    COUNT(*) AS total_oportunidades,
    COUNT(*) FILTER (WHERE result_status = 'GREEN') AS greens,
    COUNT(*) FILTER (WHERE result_status = 'RED') AS reds,
    ROUND(
        COUNT(*) FILTER (WHERE result_status = 'GREEN')::NUMERIC / 
        NULLIF(COUNT(*) FILTER (WHERE result_status IN ('GREEN','RED')), 0) * 100
    , 1) AS hit_rate_pct,
    COUNT(*) FILTER (WHERE bet_placed = TRUE) AS apostas_feitas,
    COALESCE(SUM(bet_profit) FILTER (WHERE bet_placed = TRUE), 0) AS lucro_total,
    ROUND(AVG(edge * 100) FILTER (WHERE result_status = 'GREEN'), 1) AS edge_medio_greens,
    ROUND(AVG(edge * 100) FILTER (WHERE result_status = 'RED'), 1) AS edge_medio_reds
FROM opportunities
WHERE result_status != 'PENDENTE'
GROUP BY DATE_TRUNC('week', match_date)
ORDER BY semana DESC;

-- 6. VIEW: Acurácia por confiança
CREATE OR REPLACE VIEW v_accuracy_by_confidence AS
SELECT 
    confidence,
    COUNT(*) AS total,
    COUNT(*) FILTER (WHERE result_status = 'GREEN') AS greens,
    COUNT(*) FILTER (WHERE result_status = 'RED') AS reds,
    ROUND(
        COUNT(*) FILTER (WHERE result_status = 'GREEN')::NUMERIC / 
        NULLIF(COUNT(*) FILTER (WHERE result_status IN ('GREEN','RED')), 0) * 100
    , 1) AS hit_rate_pct
FROM opportunities
WHERE result_status IN ('GREEN', 'RED')
GROUP BY confidence
ORDER BY confidence;

-- 7. VIEW: Performance por mercado
CREATE OR REPLACE VIEW v_performance_by_market AS
SELECT 
    market,
    COUNT(*) AS total,
    COUNT(*) FILTER (WHERE result_status = 'GREEN') AS greens,
    COUNT(*) FILTER (WHERE result_status = 'RED') AS reds,
    COUNT(*) FILTER (WHERE result_status = 'VOID') AS voids,
    COUNT(*) FILTER (WHERE result_status IN ('GREEN','RED')) AS decididos,
    ROUND(
        COUNT(*) FILTER (WHERE result_status = 'GREEN')::NUMERIC / 
        NULLIF(COUNT(*) FILTER (WHERE result_status IN ('GREEN','RED')), 0) * 100
    , 1) AS hit_rate_pct,
    ROUND(AVG(market_odd) FILTER (WHERE result_status IN ('GREEN','RED')), 2) AS odd_media,
    ROUND(AVG(edge * 100) FILTER (WHERE result_status IN ('GREEN','RED')), 2) AS edge_medio,
    ROUND(
        COALESCE(SUM(CASE WHEN result_status='GREEN' THEN market_odd - 1 ELSE -1 END) 
            FILTER (WHERE result_status IN ('GREEN','RED')), 0)
    , 2) AS lucro_unidades,
    ROUND(
        COALESCE(SUM(CASE WHEN result_status='GREEN' THEN market_odd - 1 ELSE -1 END) 
            FILTER (WHERE result_status IN ('GREEN','RED')), 0) /
        NULLIF(COUNT(*) FILTER (WHERE result_status IN ('GREEN','RED')), 0) * 100
    , 1) AS roi_pct
FROM opportunities
WHERE result_status IN ('GREEN', 'RED', 'VOID')
GROUP BY market
ORDER BY lucro_unidades DESC;

-- 8. VIEW: Performance por faixa de odds
CREATE OR REPLACE VIEW v_performance_by_odds_range AS
SELECT 
    CASE 
        WHEN market_odd < 1.30 THEN '1.00-1.30'
        WHEN market_odd < 1.50 THEN '1.30-1.50'
        WHEN market_odd < 1.80 THEN '1.50-1.80'
        WHEN market_odd < 2.00 THEN '1.80-2.00'
        WHEN market_odd < 2.50 THEN '2.00-2.50'
        WHEN market_odd < 3.00 THEN '2.50-3.00'
        WHEN market_odd < 5.00 THEN '3.00-5.00'
        ELSE '5.00+'
    END AS faixa_odds,
    COUNT(*) AS total,
    COUNT(*) FILTER (WHERE result_status = 'GREEN') AS greens,
    COUNT(*) FILTER (WHERE result_status = 'RED') AS reds,
    ROUND(
        COUNT(*) FILTER (WHERE result_status = 'GREEN')::NUMERIC / 
        NULLIF(COUNT(*) FILTER (WHERE result_status IN ('GREEN','RED')), 0) * 100
    , 1) AS hit_rate_pct,
    ROUND(
        COALESCE(SUM(CASE WHEN result_status='GREEN' THEN market_odd - 1 ELSE -1 END) 
            FILTER (WHERE result_status IN ('GREEN','RED')), 0)
    , 2) AS lucro_unidades,
    ROUND(
        COALESCE(SUM(CASE WHEN result_status='GREEN' THEN market_odd - 1 ELSE -1 END) 
            FILTER (WHERE result_status IN ('GREEN','RED')), 0) /
        NULLIF(COUNT(*) FILTER (WHERE result_status IN ('GREEN','RED')), 0) * 100
    , 1) AS roi_pct
FROM opportunities
WHERE result_status IN ('GREEN', 'RED')
GROUP BY faixa_odds
ORDER BY faixa_odds;

-- 9. VIEW: Performance por faixa de edge
CREATE OR REPLACE VIEW v_performance_by_edge_range AS
SELECT 
    CASE 
        WHEN edge * 100 < 3  THEN '0-3%'
        WHEN edge * 100 < 5  THEN '3-5%'
        WHEN edge * 100 < 10 THEN '5-10%'
        WHEN edge * 100 < 20 THEN '10-20%'
        WHEN edge * 100 < 50 THEN '20-50%'
        ELSE '50%+'
    END AS faixa_edge,
    COUNT(*) AS total,
    COUNT(*) FILTER (WHERE result_status = 'GREEN') AS greens,
    COUNT(*) FILTER (WHERE result_status = 'RED') AS reds,
    ROUND(
        COUNT(*) FILTER (WHERE result_status = 'GREEN')::NUMERIC / 
        NULLIF(COUNT(*) FILTER (WHERE result_status IN ('GREEN','RED')), 0) * 100
    , 1) AS hit_rate_pct,
    ROUND(
        COALESCE(SUM(CASE WHEN result_status='GREEN' THEN market_odd - 1 ELSE -1 END) 
            FILTER (WHERE result_status IN ('GREEN','RED')), 0)
    , 2) AS lucro_unidades,
    ROUND(
        COALESCE(SUM(CASE WHEN result_status='GREEN' THEN market_odd - 1 ELSE -1 END) 
            FILTER (WHERE result_status IN ('GREEN','RED')), 0) /
        NULLIF(COUNT(*) FILTER (WHERE result_status IN ('GREEN','RED')), 0) * 100
    , 1) AS roi_pct
FROM opportunities
WHERE result_status IN ('GREEN', 'RED')
GROUP BY faixa_edge
ORDER BY faixa_edge;

-- 10. VIEW: Performance por liga
CREATE OR REPLACE VIEW v_performance_by_league AS
SELECT 
    league_name,
    league_country,
    COUNT(*) AS total,
    COUNT(*) FILTER (WHERE result_status = 'GREEN') AS greens,
    COUNT(*) FILTER (WHERE result_status = 'RED') AS reds,
    ROUND(
        COUNT(*) FILTER (WHERE result_status = 'GREEN')::NUMERIC / 
        NULLIF(COUNT(*) FILTER (WHERE result_status IN ('GREEN','RED')), 0) * 100
    , 1) AS hit_rate_pct,
    ROUND(AVG(market_odd) FILTER (WHERE result_status IN ('GREEN','RED')), 2) AS odd_media,
    ROUND(
        COALESCE(SUM(CASE WHEN result_status='GREEN' THEN market_odd - 1 ELSE -1 END) 
            FILTER (WHERE result_status IN ('GREEN','RED')), 0)
    , 2) AS lucro_unidades,
    ROUND(
        COALESCE(SUM(CASE WHEN result_status='GREEN' THEN market_odd - 1 ELSE -1 END) 
            FILTER (WHERE result_status IN ('GREEN','RED')), 0) /
        NULLIF(COUNT(*) FILTER (WHERE result_status IN ('GREEN','RED')), 0) * 100
    , 1) AS roi_pct
FROM opportunities
WHERE result_status IN ('GREEN', 'RED')
GROUP BY league_name, league_country
ORDER BY lucro_unidades DESC;

-- 11. VIEW: Performance por país
CREATE OR REPLACE VIEW v_performance_by_country AS
SELECT 
    league_country,
    COUNT(*) AS total,
    COUNT(*) FILTER (WHERE result_status = 'GREEN') AS greens,
    COUNT(*) FILTER (WHERE result_status = 'RED') AS reds,
    ROUND(
        COUNT(*) FILTER (WHERE result_status = 'GREEN')::NUMERIC / 
        NULLIF(COUNT(*) FILTER (WHERE result_status IN ('GREEN','RED')), 0) * 100
    , 1) AS hit_rate_pct,
    ROUND(
        COALESCE(SUM(CASE WHEN result_status='GREEN' THEN market_odd - 1 ELSE -1 END) 
            FILTER (WHERE result_status IN ('GREEN','RED')), 0)
    , 2) AS lucro_unidades,
    ROUND(
        COALESCE(SUM(CASE WHEN result_status='GREEN' THEN market_odd - 1 ELSE -1 END) 
            FILTER (WHERE result_status IN ('GREEN','RED')), 0) /
        NULLIF(COUNT(*) FILTER (WHERE result_status IN ('GREEN','RED')), 0) * 100
    , 1) AS roi_pct
FROM opportunities
WHERE result_status IN ('GREEN', 'RED')
GROUP BY league_country
ORDER BY lucro_unidades DESC;

-- 12. VIEW: Performance diária
CREATE OR REPLACE VIEW v_performance_daily AS
SELECT 
    match_date,
    COUNT(*) AS total,
    COUNT(*) FILTER (WHERE result_status = 'GREEN') AS greens,
    COUNT(*) FILTER (WHERE result_status = 'RED') AS reds,
    ROUND(
        COUNT(*) FILTER (WHERE result_status = 'GREEN')::NUMERIC / 
        NULLIF(COUNT(*) FILTER (WHERE result_status IN ('GREEN','RED')), 0) * 100
    , 1) AS hit_rate_pct,
    ROUND(
        COALESCE(SUM(CASE WHEN result_status='GREEN' THEN market_odd - 1 ELSE -1 END) 
            FILTER (WHERE result_status IN ('GREEN','RED')), 0)
    , 2) AS lucro_unidades,
    ROUND(
        SUM(CASE WHEN result_status='GREEN' THEN market_odd - 1 ELSE -1 END) 
            FILTER (WHERE result_status IN ('GREEN','RED'))
    , 2) AS lucro_acumulado
FROM opportunities
WHERE result_status IN ('GREEN', 'RED')
GROUP BY match_date
ORDER BY match_date DESC;

-- 13. VIEW: Performance por confiança + mercado (cruzamento)
CREATE OR REPLACE VIEW v_performance_confidence_market AS
SELECT 
    confidence,
    market,
    COUNT(*) AS total,
    COUNT(*) FILTER (WHERE result_status = 'GREEN') AS greens,
    COUNT(*) FILTER (WHERE result_status = 'RED') AS reds,
    ROUND(
        COUNT(*) FILTER (WHERE result_status = 'GREEN')::NUMERIC / 
        NULLIF(COUNT(*) FILTER (WHERE result_status IN ('GREEN','RED')), 0) * 100
    , 1) AS hit_rate_pct,
    ROUND(
        COALESCE(SUM(CASE WHEN result_status='GREEN' THEN market_odd - 1 ELSE -1 END) 
            FILTER (WHERE result_status IN ('GREEN','RED')), 0)
    , 2) AS lucro_unidades
FROM opportunities
WHERE result_status IN ('GREEN', 'RED')
GROUP BY confidence, market
HAVING COUNT(*) >= 3
ORDER BY confidence, lucro_unidades DESC;

-- 14. ÍNDICES ADICIONAIS para performance
CREATE INDEX IF NOT EXISTS idx_opportunities_market ON opportunities(market);
CREATE INDEX IF NOT EXISTS idx_opportunities_league ON opportunities(league_name);
CREATE INDEX IF NOT EXISTS idx_opportunities_country ON opportunities(league_country);
CREATE INDEX IF NOT EXISTS idx_opportunities_edge ON opportunities(edge);
CREATE INDEX IF NOT EXISTS idx_opportunities_market_odd ON opportunities(market_odd);
CREATE INDEX IF NOT EXISTS idx_matches_match_id ON matches(match_id);
CREATE INDEX IF NOT EXISTS idx_opportunities_match_id ON opportunities(match_id);

-- 15. RLS (Row Level Security)
ALTER TABLE pipeline_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE opportunities ENABLE ROW LEVEL SECURITY;
ALTER TABLE matches ENABLE ROW LEVEL SECURITY;

-- Policies para service_role
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE policyname = 'service_full_pipeline_runs') THEN
        CREATE POLICY service_full_pipeline_runs ON pipeline_runs FOR ALL USING (true);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE policyname = 'service_full_opportunities') THEN
        CREATE POLICY service_full_opportunities ON opportunities FOR ALL USING (true);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE policyname = 'service_full_matches') THEN
        CREATE POLICY service_full_matches ON matches FOR ALL USING (true);
    END IF;
END $$;
