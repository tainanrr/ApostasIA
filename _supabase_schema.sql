
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
    confidence_score NUMERIC(5,1) DEFAULT 0,
    analysis_type TEXT DEFAULT 'PRE_JOGO',
    result_status TEXT DEFAULT 'PENDENTE',
    result_score TEXT,
    result_ht_score TEXT,
    result_corners TEXT,
    result_cards TEXT,
    result_shots TEXT,
    result_detail JSONB,
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

-- 4. MIGRAÇÃO: Adicionar colunas de detalhe de resultado em opportunities
ALTER TABLE opportunities ADD COLUMN IF NOT EXISTS result_ht_score TEXT;
ALTER TABLE opportunities ADD COLUMN IF NOT EXISTS result_corners TEXT;
ALTER TABLE opportunities ADD COLUMN IF NOT EXISTS result_cards TEXT;
ALTER TABLE opportunities ADD COLUMN IF NOT EXISTS result_shots TEXT;
ALTER TABLE opportunities ADD COLUMN IF NOT EXISTS result_detail JSONB;

-- 5. MIGRAÇÃO: Adicionar colunas detalhadas na tabela matches (IDs, stats, etc.)
-- ═══ IDs e identificação ═══
ALTER TABLE matches ADD COLUMN IF NOT EXISTS league_id INTEGER DEFAULT 0;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS home_team_id INTEGER DEFAULT 0;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS away_team_id INTEGER DEFAULT 0;

-- ═══ Fadiga, urgência, lesões ═══
ALTER TABLE matches ADD COLUMN IF NOT EXISTS home_fatigue NUMERIC(4,2) DEFAULT 0;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS away_fatigue NUMERIC(4,2) DEFAULT 0;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS urgency_home NUMERIC(4,2) DEFAULT 0.5;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS urgency_away NUMERIC(4,2) DEFAULT 0.5;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS injuries_home JSONB;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS injuries_away JSONB;

-- ═══ Árbitro detalhado ═══
ALTER TABLE matches ADD COLUMN IF NOT EXISTS referee_cards_avg NUMERIC(4,1) DEFAULT 0;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS referee_fouls_avg NUMERIC(4,1) DEFAULT 0;

-- ═══ Forma dos times ═══
ALTER TABLE matches ADD COLUMN IF NOT EXISTS home_form JSONB;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS away_form JSONB;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS home_form_points NUMERIC(4,2) DEFAULT 0;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS away_form_points NUMERIC(4,2) DEFAULT 0;

-- ═══ Força ataque/defesa ═══
ALTER TABLE matches ADD COLUMN IF NOT EXISTS home_attack NUMERIC(5,2) DEFAULT 0;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS home_defense NUMERIC(5,2) DEFAULT 0;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS away_attack NUMERIC(5,2) DEFAULT 0;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS away_defense NUMERIC(5,2) DEFAULT 0;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS model_alpha_h NUMERIC(6,4) DEFAULT 0;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS model_beta_h NUMERIC(6,4) DEFAULT 0;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS model_alpha_a NUMERIC(6,4) DEFAULT 0;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS model_beta_a NUMERIC(6,4) DEFAULT 0;

-- ═══ Médias de gols ═══
ALTER TABLE matches ADD COLUMN IF NOT EXISTS home_goals_scored_avg NUMERIC(4,2) DEFAULT 0;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS home_goals_conceded_avg NUMERIC(4,2) DEFAULT 0;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS away_goals_scored_avg NUMERIC(4,2) DEFAULT 0;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS away_goals_conceded_avg NUMERIC(4,2) DEFAULT 0;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS league_avg_goals NUMERIC(4,2) DEFAULT 2.7;

-- ═══ Classificação ═══
ALTER TABLE matches ADD COLUMN IF NOT EXISTS home_league_pos INTEGER DEFAULT 0;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS away_league_pos INTEGER DEFAULT 0;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS home_league_pts INTEGER DEFAULT 0;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS away_league_pts INTEGER DEFAULT 0;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS home_games_played INTEGER DEFAULT 0;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS away_games_played INTEGER DEFAULT 0;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS home_games_remaining INTEGER DEFAULT 0;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS away_games_remaining INTEGER DEFAULT 0;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS home_points_to_title INTEGER DEFAULT 99;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS away_points_to_title INTEGER DEFAULT 99;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS home_points_to_relegation INTEGER DEFAULT 99;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS away_points_to_relegation INTEGER DEFAULT 99;

-- ═══ Estatísticas de finalizações ═══
ALTER TABLE matches ADD COLUMN IF NOT EXISTS home_shots_total_avg NUMERIC(4,1) DEFAULT 0;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS away_shots_total_avg NUMERIC(4,1) DEFAULT 0;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS home_shots_on_target_avg NUMERIC(4,1) DEFAULT 0;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS away_shots_on_target_avg NUMERIC(4,1) DEFAULT 0;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS home_shots_blocked_avg NUMERIC(4,1) DEFAULT 0;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS away_shots_blocked_avg NUMERIC(4,1) DEFAULT 0;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS model_home_shots NUMERIC(4,1) DEFAULT 0;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS model_away_shots NUMERIC(4,1) DEFAULT 0;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS model_total_shots NUMERIC(4,1) DEFAULT 0;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS model_home_sot NUMERIC(4,1) DEFAULT 0;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS model_away_sot NUMERIC(4,1) DEFAULT 0;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS model_total_sot NUMERIC(4,1) DEFAULT 0;

-- ═══ Escanteios, cartões, faltas, posse ═══
ALTER TABLE matches ADD COLUMN IF NOT EXISTS home_corners_avg NUMERIC(4,1) DEFAULT 0;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS away_corners_avg NUMERIC(4,1) DEFAULT 0;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS home_cards_avg NUMERIC(4,1) DEFAULT 0;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS away_cards_avg NUMERIC(4,1) DEFAULT 0;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS home_fouls_avg NUMERIC(4,1) DEFAULT 0;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS away_fouls_avg NUMERIC(4,1) DEFAULT 0;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS home_possession NUMERIC(4,1) DEFAULT 0;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS away_possession NUMERIC(4,1) DEFAULT 0;

-- ═══ Odds extras ═══
ALTER TABLE matches ADD COLUMN IF NOT EXISTS odds_over25 NUMERIC(6,2) DEFAULT 0;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS odds_under25 NUMERIC(6,2) DEFAULT 0;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS odds_btts_yes NUMERIC(6,2) DEFAULT 0;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS odds_btts_no NUMERIC(6,2) DEFAULT 0;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS odds_corners_over NUMERIC(6,2) DEFAULT 0;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS odds_corners_under NUMERIC(6,2) DEFAULT 0;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS odds_cards_over NUMERIC(6,2) DEFAULT 0;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS odds_cards_under NUMERIC(6,2) DEFAULT 0;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS odds_ah_line NUMERIC(6,2) DEFAULT 0;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS odds_ah_home NUMERIC(6,2) DEFAULT 0;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS odds_ah_away NUMERIC(6,2) DEFAULT 0;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS odds_1x NUMERIC(6,2) DEFAULT 0;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS odds_x2 NUMERIC(6,2) DEFAULT 0;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS weather_rain NUMERIC(4,1) DEFAULT 0;

-- ═══ Qualidade dos dados ═══
ALTER TABLE matches ADD COLUMN IF NOT EXISTS data_quality NUMERIC(4,2) DEFAULT 0;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS has_real_odds BOOLEAN DEFAULT FALSE;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS has_real_standings BOOLEAN DEFAULT FALSE;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS has_real_weather BOOLEAN DEFAULT FALSE;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS home_has_real_data BOOLEAN DEFAULT FALSE;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS away_has_real_data BOOLEAN DEFAULT FALSE;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS odds_home_away_suspect BOOLEAN DEFAULT FALSE;

-- ═══ JSONB (todos os mercados e probabilidades do modelo) ═══
ALTER TABLE matches ADD COLUMN IF NOT EXISTS all_markets JSONB;
ALTER TABLE matches ADD COLUMN IF NOT EXISTS model_probs JSONB;

-- Índice para buscar por team_id (histórico)
CREATE INDEX IF NOT EXISTS idx_matches_home_team_id ON matches(home_team_id);
CREATE INDEX IF NOT EXISTS idx_matches_away_team_id ON matches(away_team_id);

-- 5. ÍNDICES
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

-- ═══════════════════════════════════════════════════════
-- MIGRAÇÃO: Novas colunas para confiança recalibrada e tipo de análise
-- Execute este bloco se as colunas ainda não existem
-- ═══════════════════════════════════════════════════════
ALTER TABLE opportunities ADD COLUMN IF NOT EXISTS confidence_score NUMERIC(5,1) DEFAULT 0;
ALTER TABLE opportunities ADD COLUMN IF NOT EXISTS analysis_type TEXT DEFAULT 'PRE_JOGO';

-- Índice para filtrar por tipo de análise (pré-jogo vs retroativa)
CREATE INDEX IF NOT EXISTS idx_opportunities_analysis_type ON opportunities(analysis_type);

-- Índice composto para dashboard por tipo de análise
CREATE INDEX IF NOT EXISTS idx_opportunities_analysis_result ON opportunities(analysis_type, result_status);

-- ═══════════════════════════════════════════════════════
-- MIGRAÇÃO: Corrigir analysis_type para oportunidades existentes
-- Compara horário do jogo com horário de criação do registro (pipeline run)
-- Se o jogo já tinha começado quando a análise foi criada → RETROATIVA
-- ═══════════════════════════════════════════════════════
UPDATE opportunities o
SET analysis_type = 'RETROATIVA'
WHERE o.analysis_type = 'PRE_JOGO'
  AND EXISTS (
    SELECT 1 FROM pipeline_runs pr
    WHERE pr.id = o.run_id
      AND (o.match_date::timestamp + COALESCE(o.match_time, '00:00')::time)
          < (pr.executed_at AT TIME ZONE 'America/Sao_Paulo') - interval '30 minutes'
  );

-- ═══════════════════════════════════════════════════════
-- MIGRAÇÃO: Recalcular confidence_score para oportunidades existentes (score=0)
-- Usa os mesmos 7 fatores do sistema Python, com dados disponíveis no banco
-- Score final = soma ponderada (0-100), mesmo algoritmo de calculate_confidence_score
-- ═══════════════════════════════════════════════════════
UPDATE opportunities
SET confidence_score = GREATEST(0, LEAST(100,
    -- FATOR 1: Edge (máx 25 pts) — edges moderados são mais confiáveis
    CASE
        WHEN COALESCE(edge, 0) * 100 <= 0 THEN 0
        WHEN edge * 100 <= 5  THEN 10
        WHEN edge * 100 <= 10 THEN 25   -- sweet spot
        WHEN edge * 100 <= 15 THEN 22
        WHEN edge * 100 <= 25 THEN 15
        WHEN edge * 100 <= 40 THEN 8
        ELSE 3                            -- edge extremo = provável erro
    END
    +
    -- FATOR 2: Odds (máx 20 pts) — odds baixas = mais previsíveis
    CASE
        WHEN COALESCE(market_odd, 2.0) <= 1.3 THEN 18
        WHEN market_odd <= 1.6 THEN 20   -- sweet spot
        WHEN market_odd <= 2.0 THEN 17
        WHEN market_odd <= 2.5 THEN 14
        WHEN market_odd <= 3.5 THEN 10
        WHEN market_odd <= 5.0 THEN 6
        ELSE 3
    END
    +
    -- FATOR 3: Probabilidade do modelo (máx 20 pts)
    CASE
        WHEN COALESCE(model_prob, 0) * 100 >= 75 THEN 20
        WHEN model_prob * 100 >= 60 THEN 17
        WHEN model_prob * 100 >= 50 THEN 14
        WHEN model_prob * 100 >= 40 THEN 10
        WHEN model_prob * 100 >= 30 THEN 6
        ELSE 3
    END
    +
    -- FATOR 4: Qualidade dos dados (estimativa: 7 pts para dados com odds reais)
    7
    +
    -- FATOR 5: Condições contextuais (estimativa: 8 pts padrão)
    8
    +
    -- FATOR 7: Concordância modelo-mercado (máx 10 pts)
    CASE
        WHEN COALESCE(market_odd, 0) > 0 AND COALESCE(model_prob, 0) > 0 THEN
            CASE
                WHEN (model_prob / (1.0 / market_odd)) BETWEEN 1.03 AND 1.25 THEN 10
                WHEN (model_prob / (1.0 / market_odd)) BETWEEN 1.01 AND 1.50 THEN 6
                WHEN (model_prob / (1.0 / market_odd)) > 1.50 THEN 2
                ELSE 0
            END
        ELSE 5
    END
))
WHERE confidence_score = 0 OR confidence_score IS NULL;

-- ═══════════════════════════════════════════════════════
-- CORREÇÃO: Jogos do dia 09/02/2026 foram analisados pré-jogo
-- (a análise original foi feita antes dos jogos começarem)
-- ═══════════════════════════════════════════════════════
UPDATE opportunities
SET analysis_type = 'PRE_JOGO'
WHERE match_date = '2026-02-09'
  AND analysis_type = 'RETROATIVA';
