# 🎓 academic-dropout-risk-ml-fuzzy

Sistema inteligente para **estimação de risco de evasão acadêmica** combinando Machine Learning e Lógica Fuzzy em pipeline integrado.

> Desenvolvido pelo **Grupo 2** — Universidade Católica de Brasília  
> Disciplina: Inteligência Artificial · Prof. William Malvezzi

---

## 📌 Sobre o Projeto

O **academic-dropout-risk-ml-fuzzy** é um sistema de apoio pedagógico que classifica estudantes em três níveis de risco de evasão e converte essa classificação em um **score interpretável (0–100)** para coordenadores e professores agirem.

A abordagem combina dois paradigmas complementares da IA:

- 🤖 **Machine Learning** — Árvore de Decisão + Naive Bayes Gaussiano para classificação supervisionada
- 🔶 **Lógica Fuzzy** — Sistema de Inferência Mamdani com 8 regras SE…ENTÃO para saída linguística interpretável

### Pipeline — Abordagem B (Integração)

| Etapa | Entrada | Saída |
|---|---|---|
| **1. ML** | 6 atributos do estudante | `P(evasão = alto)` — probabilidade calibrada |
| **2. Fuzzy** | `P(evasão)` + `frequencia` + `acessos_ava` | Score 0–100 + rótulo linguístico |

### Resultados

| Modelo | Acurácia | F1 macro | CV 5-fold |
|---|---|---|---|
| Árvore de Decisão | 84.1% | 0.844 | 0.798 ± 0.039 |
| **Naive Bayes Gaussiano** | **92.1%** | **0.920** | **0.929 ± 0.018** |

| Integração ML + Fuzzy | |
|---|---|
| Concordância ML ↔ Fuzzy | **86.5%** |
| Score médio — alto risco | 84.4 / 100 |
| Score médio — baixo risco | 17.3 / 100 |
| Separação entre extremos | 67 pontos |

---

## 📋 Entregáveis (AVA — 25/06/2025)

| Arquivo | Status | Descrição |
|---|---|---|
| `risco_evasao.ipynb` | ✅ | Notebook principal — código completo e comentado |
| `base_evasao_academica.csv` | ✅ gerado ao executar | Base de dados sintética (420 registros) |
| `resultados_integracao.csv` | ✅ gerado ao executar | Saída completa do pipeline ML + Fuzzy |
| `README.md` | ✅ | Este arquivo |
| Relatório final (PDF) | ⬜ | 10 a 20 páginas — elaborar com base no notebook |
| Apresentação (PPTX ou PDF) | ⬜ | 10 a 15 minutos — usar os gráficos gerados |

---

## 🏗️ Arquitetura

```
┌─────────────────────────────────────────────────────────────────┐
│                     PIPELINE INTEGRADO                          │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  BASE DE DADOS — 420 registros · 6 features · 3 classes │   │
│  └──────────────────────────┬──────────────────────────────┘   │
│                             │                                   │
│                             ↓  Pré-processamento                │
│             StandardScaler + divisão 70/30 estratificada        │
│                             │                                   │
│               ┌─────────────┴──────────────┐                   │
│               ↓                            ↓                   │
│   ┌───────────────────┐      ┌───────────────────────┐         │
│   │  Árvore de Decisão │      │  Naive Bayes Gaussiano │         │
│   │  acc=84.1%         │      │  acc=92.1%  ← melhor  │         │
│   └───────────────────┘      └──────────┬────────────┘         │
│                                         │                       │
│                              P(evasão = alto) [0–1]             │
│                                         │                       │
│              ┌──────────────────────────┘                       │
│              │  +  frequencia  +  acessos_ava                   │
│              ↓                                                   │
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

## 🔶 Sistema Fuzzy — Variáveis e Regras

### Variáveis de entrada

| Variável | Universo | Termos linguísticos | Função |
|---|---|---|---|
| `prob_evasao_ml` | 0 – 1 | baixa · média · alta | Trapezoidal + Triangular |
| `frequencia` | 0 – 100% | baixa · média · alta | Trapezoidal + Triangular |
| `engajamento_ava` | 0 – 30 acessos/sem | baixo · médio · alto | Trapezoidal + Triangular |

### Variável de saída

| Variável | Universo | Termos linguísticos | Função |
|---|---|---|---|
| `risco_final` | 0 – 100 | baixo · médio · alto · crítico | Trapezoidal + Triangular |

### Regras SE…ENTÃO

| Regra | Condição SE | Consequente ENTÃO | Justificativa |
|---|---|---|---|
| R1 | `prob_alta` E `freq_baixa` | CRÍTICO | Prob. alta + baixa presença → emergência |
| R2 | `prob_alta` E `eng_baixo` | CRÍTICO | Prob. alta + sem AVA → aluno sumiu |
| R3 | `prob_alta` E `freq_media` | ALTO | Prob. alta com frequência mediana |
| R4 | `prob_media` E `freq_baixa` | ALTO | Risco moderado agravado pela ausência |
| R5 | `prob_media` E `eng_medio` | MÉDIO | Engajamento regular atenua o risco |
| R6 | `prob_media` E `freq_alta` E `eng_medio` | MÉDIO | Frequência alta + eng. médio → controlável |
| R7 | `prob_baixa` E `freq_alta` | BAIXO | ML confiante + alta presença → seguro |
| R8 | `prob_baixa` E `eng_alto` | BAIXO | ML confiante + muito ativo no AVA |

---

## 🤖 Machine Learning — Modelos

### Árvore de Decisão

```
max_depth        = 6          # controla overfitting
min_samples_leaf = 5          # decisões com poucos dados são instáveis
criterion        = 'gini'     # mais eficiente que entropia
class_weight     = 'balanced' # compensa desbalanceamento residual
```

### Naive Bayes Gaussiano

Aplica o Teorema de Bayes com suposição de independência condicional entre features:

```
P(classe | features) ∝ P(features | classe) × P(classe)
var_smoothing = 1e-9  # evita P=0 por variância nula
```

### Métricas utilizadas

| Métrica | O que mede |
|---|---|
| Acurácia | Proporção de acertos — pode enganar em dados desbalanceados |
| Precisão macro | Dos preditos como X, quantos são X — penaliza falsos positivos |
| Recall macro | Dos que são X, quantos o modelo achou — penaliza falsos negativos |
| F1-score macro | Média harmônica precisão/recall — mais equilibrada para multiclasse |
| CV 5-fold estratificado | Estima generalização e detecta overfitting |

---

## 🚀 Como executar

### Google Colab (recomendado)

```
1. Acesse colab.research.google.com
2. Arquivo → Fazer upload de notebook → selecione risco_evasao.ipynb
3. Ctrl+F9 — executar tudo
```

A primeira célula instala o `scikit-fuzzy` automaticamente.  
Os CSVs são gerados durante a execução e ficam no painel de arquivos do Colab.

### Local

```bash
pip install scikit-fuzzy scikit-learn pandas numpy matplotlib seaborn
jupyter notebook risco_evasao.ipynb
```

> **Ordem de execução:** sempre do início ao fim (`Ctrl+F9` ou `Kernel → Restart & Run All`).  
> `random_state=42` em todo o código — resultados reproduzíveis em qualquer ambiente.

---

## 📁 Documentação

| Documento | Status | Descrição |
|---|---|---|
| `risco_evasao.ipynb` | ✅ | Código completo com comentários e discussão crítica |
| Relatório final (PDF) | ⬜ | 10–20 páginas conforme estrutura do enunciado |
| Apresentação (PPTX) | ⬜ | 10–15 minutos · todos os integrantes participam |

---

## 👥 Equipe

| Nome |
|---|
| João Pedro Nunes Neto |
| Leonardo dos Santos Silva |
| Lucas Gabriel Pereira Guerra |
| Luis Felipe Nunes da Fonseca Figueiredo |
| Luiz Phillipe de Souza Santos |