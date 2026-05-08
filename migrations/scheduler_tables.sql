-- ============================================================================
-- Migração: Tabelas do Scheduler
-- Descrição: Cria as tabelas scheduler_config e execution_history no Supabase
-- para configuração de execuções agendadas e histórico de execuções.
-- ============================================================================

-- ----------------------------------------------------------------------------
-- Tabela: scheduler_config
-- Armazena a configuração de agendamento para execuções automatizadas
-- (pipeline de dados e processamento de resultados).
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS scheduler_config (
    id TEXT PRIMARY KEY,           -- Identificador: 'pipeline' ou 'results'
    enabled BOOLEAN DEFAULT false, -- Se o agendamento está ativo
    hours TEXT[] DEFAULT '{}',     -- Horários de execução, ex: {"07:00","12:00"}
    days_range INTEGER DEFAULT 2,  -- Intervalo de dias (1-5 dias)
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Insere configurações padrão para pipeline e results
INSERT INTO scheduler_config (id, enabled, hours, days_range) VALUES
('pipeline', false, '{"07:00","10:00","12:00","14:00","18:00","20:00"}', 2),
('results', false, '{"09:00","12:00","16:00","19:00","22:00"}', 2)
ON CONFLICT (id) DO NOTHING;

-- ----------------------------------------------------------------------------
-- Tabela: execution_history
-- Registra o log de execução tanto do pipeline quanto dos resultados.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS execution_history (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    type TEXT NOT NULL,            -- Tipo: 'pipeline' ou 'results'
    trigger TEXT DEFAULT 'manual', -- Origem: 'manual', 'scheduled', 'github_actions'
    status TEXT DEFAULT 'running', -- Estado: 'running', 'success', 'error'
    started_at TIMESTAMPTZ DEFAULT now(),
    finished_at TIMESTAMPTZ,
    details JSONB DEFAULT '{}',
    error_message TEXT
);

-- Índices para consultas frequentes por tipo e data
CREATE INDEX IF NOT EXISTS idx_execution_history_type ON execution_history(type);
CREATE INDEX IF NOT EXISTS idx_execution_history_started ON execution_history(started_at DESC);

-- ============================================================================
-- Row Level Security (RLS)
-- Habilita RLS e políticas para acesso via service_role.
-- ============================================================================

-- Habilita RLS na tabela scheduler_config
ALTER TABLE scheduler_config ENABLE ROW LEVEL SECURITY;

-- Política: service_role tem acesso total à scheduler_config
DROP POLICY IF EXISTS "service_role_all_scheduler_config" ON scheduler_config;
CREATE POLICY "service_role_all_scheduler_config"
    ON scheduler_config
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);

-- Habilita RLS na tabela execution_history
ALTER TABLE execution_history ENABLE ROW LEVEL SECURITY;

-- Política: service_role tem acesso total à execution_history
DROP POLICY IF EXISTS "service_role_all_execution_history" ON execution_history;
CREATE POLICY "service_role_all_execution_history"
    ON execution_history
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);
