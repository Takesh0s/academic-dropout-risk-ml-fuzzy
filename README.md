# academic-dropout-risk-ml-fuzzy

Sistema inteligente para **estimação de risco de evasão acadêmica** combinando Machine Learning e Lógica Fuzzy em pipeline integrado.

> **Grupo 2** — Universidade Católica de Brasília  
> Disciplina: Inteligência Artificial · Prof. William Malvezzi  
> Data de entrega: 25/06/2026

---

## Sobre o Projeto

O **academic-dropout-risk-ml-fuzzy** é um sistema de apoio pedagógico que classifica estudantes em três níveis de risco de evasão e converte essa classificação em um **score interpretável (0–100)** para uso por coordenadores e professores.

A solução combina dois paradigmas complementares da Inteligência Artificial, conforme proposto no enunciado (Abordagem B — Integração):

- **Machine Learning** — Árvore de Decisão + Naive Bayes Gaussiano para classificação supervisionada
- **Lógica Fuzzy** — Sistema de Inferência Mamdani com 8 regras SE…ENTÃO para saída linguística interpretável

### Por que combinar ML e Fuzzy?

| Abordagem | Saída | Problema |
|---|---|---|
| ML puro | "78% de probabilidade de evasão" | Opaco para o coordenador pedagógico |
| Fuzzy puro | Score baseado em regras manuais | Sem aprendizado de dados históricos |
| **ML + Fuzzy (nossa abordagem)** | **"CRÍTICO — 88/100"** | Baseado em dados + linguagem acionável |

### Pipeline — Abordagem B (Integração)

| Etapa | Entrada | Saída |
|---|---|---|
| **1. ML** | 6 atributos do estudante | `P(evasão = alto)` — probabilidade calibrada |
| **2. Fuzzy** | `P(evasão)` + `frequencia` + `acessos_ava` | Score 0–100 + rótulo linguístico |

---

## Resultados

### Machine Learning

| Modelo | Acurácia | F1 macro | CV 5-fold |
|---|---|---|---|
| Árvore de Decisão | 84,1% | 0,844 | 0,798 ± 0,039 |
| **Naive Bayes Gaussiano** | **92,1%** | **0,920** | **0,929 ± 0,018** |

O Naive Bayes Gaussiano foi selecionado como modelo principal por dois motivos: (1) desempenho superior em todas as métricas e (2) o método `predict_proba()` retorna probabilidades calibradas que servem como ponte natural para o sistema Fuzzy.

### Integração ML + Fuzzy

| Métrica | Resultado |
|---|---|
| Concordância ML ↔ Fuzzy | **86,5%** |
| Score médio — alto risco | 84,4 / 100 |
| Score médio — baixo risco | 17,3 / 100 |
| Separação entre extremos | **67 pontos** |

A separação de 67 pontos entre os extremos valida que o sistema Fuzzy discrimina corretamente os perfis opostos. Os ~13,5% de discordância não são necessariamente erros — são casos em que o Fuzzy incorpora informação comportamental atual (frequência e acessos ao AVA) que o ML histórico não captou.

---

## Entregáveis (AVA — 25/06/2026)

| Arquivo | Status | Descrição |
|---|---|---|
| `risco_evasao.ipynb` | ✅ | Notebook principal — código completo e comentado linha a linha |
| `base_evasao_academica.csv` | ✅ gerado ao executar | Base de dados sintética (420 registros, 6 features, 3 classes) |
| `resultados_integracao.csv` | ✅ gerado ao executar | Saída completa do pipeline ML + Fuzzy |
| `README.md` | ✅ | Este arquivo |
| `Relatorio_Final_Fuzzy.pdf` | ✅ | Relatório de 10 a 20 páginas em formato PDF |
| `Fuzzy.pptx` | ✅ | Apresentação de slides em formato PPTX |
| [Vídeo de defesa (YouTube)](https://youtu.be/QwDAqz0FWGs) | ✅ | Até 15 minutos — https://youtu.be/QwDAqz0FWGs |

---

## Arquitetura

```
┌─────────────────────────────────────────────────────────────────┐
│                     PIPELINE INTEGRADO                          │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  BASE DE DADOS — 420 registros · 6 features · 3 classes │    │
│  └──────────────────────────┬──────────────────────────────┘    │
│                             │                                   │
│                             ↓  Pré-processamento                │
│             StandardScaler + divisão 70/30 estratificada        │
│                             │                                   │
│               ┌─────────────┴──────────────┐                    │
│               ↓                            ↓                    │
│   ┌───────────────────┐      ┌───────────────────────┐          │
│   │ Árvore de Decisão │      │ Naive Bayes Gaussiano │          │
│   │ acc=84,1%         │      │ acc=92,1%  ← melhor   │          │
│   └───────────────────┘      └──────────┬────────────┘          │
│                                         │                       │
│                              P(evasão = alto) [0–1]             │
│                                         │                       │
│              ┌──────────────────────────┘                       │
│              │  +  frequencia  +  acessos_ava                   │
│              ↓                                                  │
│   ┌──────────────────────────────────────────┐                  │
│   │       SISTEMA FUZZY MAMDANI              │                  │
│   │  3 entradas · 8 regras · 4 termos saída  │                  │
│   │  defuzzificação por centroide            │                  │
│   └──────────────────┬───────────────────────┘                  │
│                      │                                          │
│              Score 0–100 + rótulo linguístico                   │
│            (baixo · médio · alto · crítico)                     │
└─────────────────────────────────────────────────────────────────┘
```

### Estrutura de pastas

```
academic-dropout-risk-ml-fuzzy/
├── risco_evasao.ipynb              Notebook principal (Google Colab)
├── base_evasao_academica.csv       Base de dados sintética — gerada ao executar
├── resultados_integracao.csv       Saída do pipeline ML + Fuzzy — gerada ao executar
├── README.md                       Este arquivo
└── .gitignore
```

---

## Como Executar

### Google Colab (recomendado)

```
1. Acesse colab.research.google.com
2. Arquivo → Fazer upload de notebook → selecione risco_evasao.ipynb
3. Ctrl+F9 — executar tudo
```

A **célula 1** instala o `scikit-fuzzy` automaticamente via `!pip install scikit-fuzzy --quiet`. Nenhuma instalação manual é necessária.

Os arquivos `base_evasao_academica.csv` e `resultados_integracao.csv` são gerados durante a execução e ficam disponíveis no painel de arquivos do Colab (ícone de pasta à esquerda).

### Execução Local

```bash
pip install scikit-fuzzy scikit-learn pandas numpy matplotlib seaborn
jupyter notebook risco_evasao.ipynb
```

> **Importante:** executar sempre do início ao fim (`Ctrl+F9` ou `Kernel → Restart & Run All`).  
> `random_state=42` está fixado em todo o código — os resultados são idênticos em qualquer ambiente.

---

## Descrição das Células do Notebook

| Célula | Descrição |
|---|---|
| 1 | Instalação do scikit-fuzzy e imports; registro de versões das bibliotecas |
| 2 | Geração da base sintética (420 registros, distribuições Normal e Poisson) |
| 3 | Análise exploratória: histogramas, boxplots por classe, heatmap de correlação |
| 4 | Pré-processamento: encoding ordinal, StandardScaler, split 70/30 estratificado |
| 5 | Função `avaliar_modelo()` — métricas, matriz de confusão, validação cruzada |
| 6 | Árvore de Decisão: treinamento, visualização da árvore, importância de features |
| 7 | Naive Bayes Gaussiano: treinamento e avaliação |
| 8 | Comparação dos dois modelos de ML |
| 9 | Definição das funções de pertinência fuzzy (trapmf + trimf) |
| 10 | Visualização das funções de pertinência das 3 entradas e 1 saída |
| 11 | Definição das 8 regras fuzzy SE…ENTÃO |
| 12 | Função `executar_inferencia()` — Mamdani completo (fuzzificação → inferência → defuzzificação) |
| 13 | Pipeline integrado ML + Fuzzy sobre o conjunto de teste; geração do CSV de resultados |
| 14 | Resultados finais, discussão crítica e análise de concordância ML ↔ Fuzzy |

---

## Sistema Fuzzy — Variáveis e Regras

### Variáveis de entrada

| Variável | Universo | Termos linguísticos | Função de pertinência |
|---|---|---|---|
| `prob_evasao_ml` | 0 – 1 | baixa · média · alta | Trapezoidal (bordas) + Triangular (centro) |
| `frequencia` | 0 – 100% | baixa · média · alta | Trapezoidal (bordas) + Triangular (centro) |
| `engajamento_ava` | 0 – 30 acessos/sem | baixo · médio · alto | Trapezoidal (bordas) + Triangular (centro) |

### Variável de saída

| Variável | Universo | Termos linguísticos | Função de pertinência |
|---|---|---|---|
| `risco_final` | 0 – 100 | baixo · médio · alto · crítico | Trapezoidal (bordas) + Triangular (centro) |

### Processo de Inferência Mamdani

```
1. Fuzzificação   → valor numérico → grau μ(x) via fuzz.interp_membership
2. Regras         → E = mínimo dos graus dos antecedentes
3. Implicação     → corte da função de pertinência do consequente no nível de ativação
4. Agregação      → máximo das contribuições para o mesmo consequente (fuzz.fuzzy_or)
5. Defuzzificação → centroide da área agregada (fuzz.defuzz, método 'centroid')
```

### Regras SE…ENTÃO

| Regra | Condição SE | Consequente ENTÃO | Justificativa |
|---|---|---|---|
| R1 | `prob_alta` E `freq_baixa` | CRÍTICO | Probabilidade alta + baixa presença → situação de emergência |
| R2 | `prob_alta` E `eng_baixo` | CRÍTICO | Probabilidade alta + sem acesso ao AVA → aluno desapareceu |
| R3 | `prob_alta` E `freq_media` | ALTO | Probabilidade alta com frequência mediana |
| R4 | `prob_media` E `freq_baixa` | ALTO | Risco moderado agravado pela ausência física |
| R5 | `prob_media` E `eng_medio` | MÉDIO | Engajamento regular atenua o risco |
| R6 | `prob_media` E `freq_alta` E `eng_medio` | MÉDIO | Frequência alta + engajamento médio → situação controlável |
| R7 | `prob_baixa` E `freq_alta` | BAIXO | ML confiante + alta presença → situação segura |
| R8 | `prob_baixa` E `eng_alto` | BAIXO | ML confiante + aluno muito ativo no AVA |

---

## Machine Learning — Modelos e Parâmetros

### Base de Dados Sintética

Dados reais de estudantes são protegidos pela LGPD, portanto a base foi gerada sinteticamente com parâmetros estatisticamente coerentes com a literatura (INEP, 2022):

| Feature | Perfil baixo risco | Perfil médio risco | Perfil alto risco | Distribuição |
|---|---|---|---|---|
| `frequencia` | μ = 88% | μ = 70% | μ = 48% | Normal (σ = 12) |
| `nota_media` | μ = 7,8 | μ = 5,8 | μ = 3,8 | Normal (σ = 1,5) |
| `atividades_entregues` | μ = 87% | μ = 65% | μ = 40% | Normal (σ = 15) |
| `acessos_ava` | μ = 14/sem | μ = 8/sem | μ = 4/sem | Poisson |
| `reprovas_anteriores` | 70% com zero | distribuição moderada | 40% com 2+ | Categórica |
| `situacao_financeira` | maioria estável | misto | maioria instável | Categórica |

420 registros · 140 por classe · balanceamento deliberado para evitar viés do modelo.

### Árvore de Decisão

```python
DecisionTreeClassifier(
    max_depth        = 6,          # controla overfitting
    min_samples_leaf = 5,          # decisões com poucos dados são instáveis
    criterion        = 'gini',     # mais eficiente computacionalmente que entropia
    class_weight     = 'balanced', # compensa desbalanceamento residual
    random_state     = 42
)
```

### Naive Bayes Gaussiano

Aplica o Teorema de Bayes com suposição de independência condicional entre features:

```
P(classe | features) ∝ P(features | classe) × P(classe)
```

```python
GaussianNB(var_smoothing=1e-9)  # evita P=0 por variância nula
```

O método `predict_proba()` retorna probabilidades calibradas por classe — a coluna `P(alto)` é a entrada principal do sistema Fuzzy (variável `prob_evasao_ml`).

### Métricas utilizadas

| Métrica | O que mede | Por que utilizamos |
|---|---|---|
| Acurácia | Proporção de acertos totais | Baseline de comparação |
| Precisão macro | Dos preditos como X, quantos são X | Penaliza falsos positivos |
| Recall macro | Dos que são X, quantos o modelo identificou | Penaliza falsos negativos |
| F1-score macro | Média harmônica precisão/recall | Mais equilibrada para multiclasse balanceada |
| CV 5-fold estratificado | Estima generalização; detecta overfitting | Valida que os resultados não são do split específico |

---

## Referências

RUSSELL, Stuart; NORVIG, Peter. *Inteligência Artificial*. 3. ed. Rio de Janeiro: Elsevier, 2013.

ZADEH, Lotfi A. Fuzzy Sets. *Information and Control*, v. 8, n. 3, p. 338–353, 1965.

SCIKIT-LEARN. *User Guide*. Disponível em: https://scikit-learn.org/stable/user_guide.html. Acesso em: jun. 2026.

SCIKIT-FUZZY. *Documentation*. Disponível em: https://pythonhosted.org/scikit-fuzzy/. Acesso em: jun. 2026.

MALVEZZI, William. *Slides da disciplina de Inteligência Artificial: Lógica Fuzzy e Machine Learning*. Material interno, UCB, 2026.

---

## Equipe — Grupo 2

| Nome |
|---|
| João Pedro Nunes Neto |
| Leonardo dos Santos Silva |
| Lucas Gabriel Pereira Guerra |
| Luis Felipe Nunes da Fonseca Figueiredo |
| Luiz Phillipe de Souza Santos |