# Pipeline Caminhos SP

## Visão geral
Este pipeline gera saídas padronizadas (CSV e GeoJSON) para uso em Looker, PowerBI e QGIS, criando dimensões independentes para linhas, estações, terminais e distritos OD. Ele também gera a dimensão de linhas de ônibus com área operacional e lote, separadas por status quando aplicável.

## Dependências
- Python 3.9+
- `pandas`
- `openpyxl`

## Input obrigatório
O pipeline espera o arquivo compactado com as planilhas classificadas por dimensão em:
`data_raw/planilhas_CLASSIFICADAS_v5_4_rev3.zip`.

Este ZIP inclui os arquivos:
- `01_Linhas_Plano_Corpus_Ducen_v5_4_CLASSIFICADO.xlsx`
- `02_Estacoes_Plano_Corpus_Ducen_v5_4_CLASSIFICADO.xlsx`
- `03_Coordenadas_Plano_Corpus_Ducen_v5_4_CLASSIFICADO.xlsx`
- `OD2023_Distritos_Infra_Transporte.xlsx`
- `Relacionamentos_Linhas_Demanda_AreaOperacional_CORES_1a9_TP_TS.xlsx`

### Planilhas legadas (opcional)
Para continuar usando as planilhas antigas (`*_ATUALIZADO_GERAL.xlsx`), execute o
pipeline com a flag `--no-use-classified-zip`.

## Como executar (Windows PowerShell)
```powershell
# Na raiz do repositório
./scripts/run_pipeline.ps1
```

## Estrutura de saída
```
/data_out
  /lines
    linhas_existente.csv
    linhas_em_construcao.csv
    linhas_em_projeto.csv
    linhas_proposto.csv
  /stations
    estacoes_existente.csv
    estacoes_em_construcao.csv
    estacoes_em_projeto.csv
    estacoes_proposto.csv
  /terminals
    terminais_existente.csv
    terminais_em_construcao.csv
    terminais_em_projeto.csv
    terminais_proposto.csv
  /line_points
    linha_pontos_existente.csv
    linha_pontos_em_construcao.csv
    linha_pontos_em_projeto.csv
    linha_pontos_proposto.csv
  /bus
    linhas_onibus.csv
  /districts
    distritos_od.csv
    distritos_od.geojson
```

## Regras atendidas
- **Compatibilidade BI/GIS**: os CSVs usam nomes de colunas em snake_case e tipos amigáveis para Looker/PowerBI, além de GeoJSON para QGIS.
- **Separação por status**: todas as dimensões de linhas, estações, terminais e pontos de linha são exportadas por status (`existente`, `em_construcao`, `em_projeto`, `proposto`).
- **Operador/Área/Lote**:
  - Para metrô/trem, o campo `operador` vem de `subsistema`.
  - Para ônibus, `linhas_onibus.csv` inclui `area_operacional_num_1a9`, `lote_mode` e `consorcio_operador`.
- **Localização**: latitude, longitude, cidade e distrito (quando geometria disponível) são incluídos.
- **Distritos OD**: `distritos_od.csv` contém dados do OD2023 e `distritos_od.geojson` contém a geometria dos distritos.
- **Ônibus em São Paulo**: o pipeline restringe as saídas de ônibus à cidade de São Paulo.

## Dicionário de dados (principais campos)
### Linhas
- `linha_id`: identificador padronizado da linha.
- `linha`: nome original da linha.
- `sistema` / `subsistema`: modo e operador.
- `status`: `existente`, `em_construcao`, `em_projeto` ou `proposto`.
- `qtde_vias`, `qtde_estacoes`, `comprimento_km`.

### Estações
- `estacao_id`: identificador padronizado.
- `nome_da_estacao`.
- `latitude`, `longitude`.
- `distrito`: distrito OD associado.

### Terminais
- `terminal_id`: id do terminal (derivado do GTFS).
- `stop_name`, `latitude`, `longitude`, `distrito`.

### Linhas de ônibus
- `route_id`, `route_short_name`, `route_long_name`.
- `area_operacional_num_1a9`, `lote_mode`, `consorcio_operador`.

### Distritos OD
- `numerodistrito`, `nomedistrito`, métricas OD.
- `geometria_wkt` (para uso em BI).

## Observações
- Caso algum status não seja identificado na origem, o pipeline gera `desconhecido`.
- A atribuição de distrito usa o KMZ de Geosampa. Se o arquivo não estiver presente, o campo `distrito` permanece vazio.
