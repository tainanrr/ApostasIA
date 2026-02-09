
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

-- 7. RLS (Row Level Security)
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
