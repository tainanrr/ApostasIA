-- ============================================================================
-- Migração: Atualizar horários do Scheduler
-- Descrição: Atualiza os horários de execução automática no Supabase.
--
-- Pipeline (busca por oportunidades): 07h, 10h, 12h, 14h, 18h, 20h
-- Results  (busca por resultados):    09h, 12h, 16h, 19h, 22h
--
-- Executar no Supabase SQL Editor para aplicar em produção.
-- ============================================================================

UPDATE scheduler_config
SET hours = '{"07:00","10:00","12:00","14:00","18:00","20:00"}',
    updated_at = now()
WHERE id = 'pipeline';

UPDATE scheduler_config
SET hours = '{"09:00","12:00","16:00","19:00","22:00"}',
    updated_at = now()
WHERE id = 'results';
