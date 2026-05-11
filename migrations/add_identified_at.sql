-- ============================================================================
-- Migração: Adicionar campo identified_at às oportunidades
-- Descrição: Registra a data/hora exata em que cada oportunidade foi
--            identificada pelo sistema. Também garante que apenas apostas
--            realmente pré-game (antes do início do jogo) sejam marcadas
--            como PRE_JOGO.
--
-- Executar no Supabase SQL Editor para aplicar em produção.
-- ============================================================================

-- 1. Adicionar coluna identified_at (se não existir)
ALTER TABLE opportunities
ADD COLUMN IF NOT EXISTS identified_at TIMESTAMPTZ;

-- 2. Preencher identified_at para oportunidades existentes usando created_at
UPDATE opportunities
SET identified_at = created_at
WHERE identified_at IS NULL AND created_at IS NOT NULL;

-- 3. Índice para buscas por identified_at
CREATE INDEX IF NOT EXISTS idx_opportunities_identified_at
    ON opportunities (identified_at);

-- 4. Corrigir oportunidades marcadas como PRE_JOGO que na verdade
--    foram criadas DEPOIS do horário do jogo (bug corrigido no código)
UPDATE opportunities
SET analysis_type = 'RETROATIVA'
WHERE result_status = 'PENDENTE'
  AND analysis_type = 'PRE_JOGO'
  AND match_date IS NOT NULL
  AND match_time IS NOT NULL
  AND created_at IS NOT NULL
  AND (match_date || ' ' || match_time)::timestamp < (created_at - interval '30 minutes');
