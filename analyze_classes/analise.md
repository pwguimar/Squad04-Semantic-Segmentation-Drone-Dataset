# Análise da Distribuição de Classes Binárias em Dataset de Segmentação

Este documento detalha o processo e os resultados da análise da distribuição de pixels para classes binárias em um conjunto de dados de segmentação semântica de drones. O objetivo é quantificar a proporção de cada classe (pixels) no dataset para identificar possíveis desequilíbrios.

## 1. Contexto do Problema

Em tarefas de segmentação semântica, um modelo é treinado para classificar cada pixel de uma imagem em uma categoria específica. A "distribuição de classes" refere-se à proporção de pixels pertencentes a cada categoria. Um **desequilíbrio de classes** ocorre quando uma classe é significativamente mais (ou menos) frequente que outra, o que pode impactar negativamente o desempenho do modelo treinado.

O dataset original possui dois tipos de segmentação:
* **Segmentação Binária:** Focada na distinção de duas classes principais (e.g., alvo vs. não-alvo).
* **Segmentação Multiclasse (5 classes):** Distingue cinco macro-grupos de objetos.

O grupo de trabalho optou por focar primeiramente na **segmentação binária** devido à complexidade das demais atividades e como uma etapa fundamental para o entendimento e validação da pipeline de análise de dados.

As definições de classe e seus mapeamentos são fornecidas através de mapeamentos diretos de cores RGB, identificados após análise das máscaras.

## 2. Documentação do Código - Análise de Distribuição de Classes

O código para esta análise está localizado em `analyze_classes/binary_classes.py` para a segmentação binária e em `analyze_classes/multiple_classes.py` para a segmentação multiclasse de 5 classes.

A seguir, a documentação das funções utilizadas na análise da distribuição de classes:

### 2.1. Funções
#### `_validate_masks_directory(directory_path)`

* **Propósito:** Verifica a existência e o conteúdo de um diretório de máscaras.
* **Lógica:**
    * Confirma se o `directory_path` fornecido é um diretório válido.
    * Lista todos os arquivos dentro do diretório que possuem extensões de imagem (`.png` e `.jpg`).
    * Verifica se há imagens encontradas.
* **Entrada:**
    * `directory_path` (string): O caminho completo para a pasta que contém as imagens de máscara.
* **Saída:**
    * `mask_files_list` (lista de strings): Uma lista dos nomes dos arquivos de imagem encontrados.
    * `total_found_masks` (inteiro): O número total de arquivos de imagem encontrados.
    * Retorna listas vazias e 0 se o diretório não for encontrado ou estiver vazio.

#### `_count_pixels_in_masks(masks_directory, mask_files_list, rgb_to_class_id_map, class_id_to_name_map, color_tolerance=1)`

* **Propósito:** Realiza a contagem central dos pixels, mapeando as cores RGB das máscaras para os IDs de classe definidos.
* **Lógica:**
    * Itera sobre cada arquivo de imagem de máscara.
    * **Carregamento:** Carrega a imagem da máscara como RGB (`cv2.imread(..., cv2.IMREAD_COLOR)` e `cv2.cvtColor(..., cv2.COLOR_BGR2RGB)`).
    * **Identificação de Cores Únicas (Depuração):** Para a primeira imagem processada, a função extrai e imprime todas as tuplas RGB únicas presentes. Esta etapa foi crucial para identificar as cores exatas nas máscaras e corrigir o mapeamento.
    * **Remapeamento de Pixels:** Para cada pixel, compara sua cor RGB com as cores alvo definidas em `rgb_to_class_id_map`. Utiliza uma `color_tolerance` (padrão de 1) para permitir pequenas variações nas cores dos pixels, mapeando-os para o `class_id` correspondente. Pixels que não correspondem a nenhuma cor mapeada são marcados com 255 (não mapeado).
    * **Contagem:** Acumula a contagem total de pixels para cada `class_id` remapeado.
    * **Depuração Pós-Remapeamento:** Para a primeira imagem, mostra os IDs de pixel únicos após o remapeamento e alerta sobre pixels não mapeados.
* **Entradas:**
    * `masks_directory` (string): Caminho para o diretório das máscaras.
    * `mask_files_list` (lista de strings): Nomes dos arquivos de máscara a serem processados.
    * `rgb_to_class_id_map` (dicionário): Mapeamento de tuplas RGB para IDs de classe (ex: `(R, G, B) -> ID`).
    * `class_id_to_name_map` (dicionário): Mapeamento de IDs de classe para seus nomes.
    * `color_tolerance` (inteiro): A diferença máxima permitida para cada canal RGB na comparação de cores.
* **Saída:**
    * `class_pixel_counts_dict` (dicionário): Contagem total de pixels para cada classe.

#### `_calculate_and_print_distribution(pixel_counts, min_threshold_percentage_val, title_prefix="")`

* **Propósito:** Calcula as porcentagens de cada classe e imprime a distribuição em formato tabular.
* **Lógica:**
    * Soma todos os pixels contados para obter o total.
    * Calcula a porcentagem de pixels de cada classe em relação ao total.
    * Imprime uma tabela ordenada, mostrando a contagem de pixels e a porcentagem para cada classe.
    * Identifica e lista classes com porcentagens abaixo de um `min_threshold_percentage_val`, indicando classes minoritárias.
* **Entradas:**
    * `pixel_counts` (dicionário): Contagem de pixels por classe.
    * `min_threshold_percentage_val` (float): Limiar percentual para identificar classes minoritárias.
    * `title_prefix` (string): Prefixo para os títulos das seções de impressão (ex: "em Pixels Binários").
* **Saída:**
    * `sorted_info` (lista de tuplas): Informações das classes ordenadas por contagem de pixels.
    * `percentages_dict` (dicionário): Porcentagens de cada classe.

#### `_plot_distribution(sorted_class_data, percentages_data, plot_title="Distribuição de Pixels por Classe")`

* **Propósito:** Gera um gráfico de barras visualizando a distribuição de classes.
* **Lógica:**
    * Cria um gráfico de barras usando `matplotlib`, onde o eixo X representa os nomes das classes e o eixo Y representa a porcentagem de pixels.
    * Ajusta rótulos, títulos e rotação para melhor legibilidade.
    * Exibe o gráfico.
* **Entradas:**
    * `sorted_class_data` (lista de tuplas): Dados das classes ordenados.
    * `percentages_data` (dicionário): Porcentagens de cada classe.
    * `plot_title` (string): Título para o gráfico.

## 3. Saída da Execução e Análise dos Resultados

Abaixo, a saída completa do console ao executar o código para a análise da distribuição de classes binárias:
```
--- Verificando o Diretório ---
SUCESSO: Encontrados 400 imagens.

Iniciando a contagem de pixels nas máscaras...
Todos os valores RGB únicos nesta máscara:
  (np.uint8(0), np.uint8(153), np.uint8(153))
  (np.uint8(204), np.uint8(153), np.uint8(255))
--------------------------------------------------
Valores de pixel únicos APÓS remapeamento (máx 20): [0 1]
--------------------------------------------------
Processados 50/400 arquivos...
Processados 100/400 arquivos...
Processados 150/400 arquivos...
Processados 200/400 arquivos...
Processados 250/400 arquivos...
Processados 300/400 arquivos...
Processados 350/400 arquivos...
Processados 400/400 arquivos...

--- Processamento Concluído ---

--- Distribuição de Classes (em Pixels Binários) ---
Classe: landing-zones   | Pixels: 213715769       | Porcentagem: 75.62%
Classe: obstacles       | Pixels: 68908231        | Porcentagem: 24.38%

--- Possíveis Classes Minoritárias (em Pixels Binários - abaixo de 2.0%) ---
```

### 3.1. Análise dos Resultados Obtidos

1.  **Diagnóstico do Formato das Máscaras:**
    * A etapa de depuração `"TODOS os valores RGB únicos nesta máscara:"` foi crucial, revelando as duas cores exatas presentes: `(0, 153, 153)` e `(204, 153, 255)`. Esta informação permitiu corrigir o mapeamento de cores `rgb_to_binary_id`.

2.  **Distribuição de Classes Binárias:**
    * **`landing-zones`**: 75.62% dos pixels do dataset.
    * **`obstacles`**: 24.38% dos pixels do dataset.

3.  **Desequilíbrio de Classes:**
    * O dataset apresenta um **desequilíbrio significativo** entre as classes binárias. A classe `landing-zones` (75.62%) é a classe majoritária, enquanto a classe `obstacles` (24.38%) é a classe minoritária. A proporção é de aproximadamente 3:1 (três vezes mais pixels de `landing-zones` do que de `obstacles`).
    * Embora a classe `obstacles` esteja acima do limiar de 2.0% para ser considerada "minoritária crítica" neste caso, um desequilíbrio de 3:1 ainda pode introduzir viés no modelo se não for tratado.

## 4. Conclusão

Esta análise de distribuição de classes é uma etapa vital no pré-processamento de dados para projetos de segmentação. O processo revelou não apenas o desequilíbrio inerente ao dataset binário, mas também a importância de um mapeamento preciso de cores RGB para IDs de classe, destacando um aspecto crucial da preparação de dados para este tipo de dataset. As informações obtidas direcionarão as próximas fases de modelagem para garantir que o modelo de segmentação binária seja eficaz em ambos os tipos de classes.