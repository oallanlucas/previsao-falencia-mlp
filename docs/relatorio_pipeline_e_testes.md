# Relatório técnico: pipeline e testes estatísticos

Este relatório explica, passo a passo, como os dados viram previsões e como o melhor modelo é escolhido e comparado.
Os números finais estão nos notebooks (seções indicadas em cada passo) e nos arquivos de `resultados/`.

**Sumário**

1. [Dados e formulação](#1-dados-e-formulação)
2. [Divisão dos dados e validação cruzada](#2-divisão-dos-dados-e-validação-cruzada)
3. [Pré-processamento](#3-pré-processamento)
4. [Métricas](#4-métricas)
5. [Regressão logística (notebook 03)](#5-regressão-logística-notebook-03)
6. [MLP (notebook 04)](#6-mlp-notebook-04)
7. [Seleção robusta](#7-seleção-robusta)
8. [Os testes estatísticos, um a um](#8-os-testes-estatísticos-um-a-um)
9. [Comparação final no teste (notebook 05)](#9-comparação-final-no-teste-notebook-05)
10. [Alerta por horizonte e viés de tempo (notebook 06)](#10-alerta-por-horizonte-e-viés-de-tempo-notebook-06)
11. [Limitações](#11-limitações)
12. [Referências](#12-referências)

---

## 1. Dados e formulação

**Base.** *Polish Companies Bankruptcy* (UCI 365): 5 arquivos ARFF (`1year` a `5year`), 64 indicadores financeiros
(razões contábeis X1 a X64) e a variável `class` (1 = faliu). Depois de remover as linhas duplicadas dentro de cada
arquivo, ficam **43.004 balanços** e **2.083 falências** (4,8%).

**O que cada arquivo significa.** O arquivo `kyear` traz balanços tirados **6 − k anos antes** do desfecho. Para uma
empresa que faliu no ano *T*: `5year` = balanço de *T* − 1; `4year` = *T* − 2; …; `1year` = *T* − 5. As empresas que
não faliram aparecem em todos os arquivos, com balanços de 2007 a 2013. A proporção de falidas sobe de 3,9% (`1year`)
para 7,0% (`5year`).

**Alvo multiclasse.**

| classe *y* | significado | origem |
|---|---|---|
| 0 | não faliu no período observado | `class = 0` em qualquer arquivo |
| *h* = 1, …, 5 | faliu *h* anos depois do balanço | `class = 1` no arquivo `(6 − h)year` |

Os modelos estimam $P(y = h \mid x)$ para as 6 classes. Três perguntas saem do mesmo modelo:

- **ano exato**: $P(y = h)$ — avaliado como "classe *h* contra o resto";
- **faliu (em qualquer um dos 5 anos)**: $1 - P(y = 0)$;
- **alerta em até *h* anos** (a tarefa do enunciado): $P(y \le h, y \ge 1) = \sum_{j=1}^{h} P(y = j)$.

**Vazamento evitado.** O arquivo de origem não entra como variável: para as falidas ele determina a classe.

---

## 2. Divisão dos dados e validação cruzada

1. **Treino 80% / teste 20%**, estratificado pela classe de 6 níveis (`train_test_split`, semente 42). O teste
   fica guardado e é usado **uma única vez**, no notebook 05.
2. **Validação cruzada (CV) estratificada de 5 folds** dentro do treino. Em cada rodada, 4 folds treinam e o fold
   restante, nunca visto naquele ajuste, é previsto. Juntando os 5 folds, cada empresa do treino recebe uma
   **previsão fora da amostra** (OOF, *out-of-fold*). Todas as escolhas usam só essas previsões.
3. **CV repetida 3 × 5** (seleção robusta e testes): a mesma CV em 3 divisões diferentes do treino (`random_state`
   42, 43 e 44), totalizando 15 folds. A divisão 0 é a mesma usada em todas as outras escolhas.
4. **Validação interna da MLP**: dentro de cada ajuste da rede, 15% do treino (estratificado) escolhem a melhor
   época e fazem a parada antecipada. O fold da CV nunca participa dessa escolha.

---

## 3. Pré-processamento

Todos os passos ficam num `Pipeline` do scikit-learn (ou imbalanced-learn), ajustado **só nos dados de treino** de
cada rodada e depois aplicado aos dados de avaliação. Ordem:

| Passo | Opções testadas | Por quê |
|---|---|---|
| 1. Escala | padronização; robusta (mediana/IQR); winsorização 1–99% + padronização; **quantis → normal** | as razões têm caudas extremas (curtose ~10⁴) |
| 2. Faltantes | mediana do treino | X37 (44% faltante) foi descartado; nenhuma coluna nova é criada |
| 3. Redução (obrigatória) | **Pearson (\|r\| > 0,8) + VIF iterativo (< 5)**; PCA com 95% da variância | colinearidade forte entre razões |
| 4. Desbalanceamento | nenhum; pesos de classe; oversampling; SMOTE | 95% das linhas são da classe 0 |

- **Quantis → normal**: cada indicador é trocado por $\Phi^{-1}(\hat F(x))$, onde $\hat F$ é a distribuição
  acumulada empírica do treino (1.000 quantis). Valores de ±10⁶ viram no máximo ±5,2 e a ordem é preservada.
- **Filtro de correlação**: as variáveis são ordenadas pela força univariada $|AUC - 0{,}5|$; percorrendo essa
  ordem, uma variável sai se tiver $|r| > 0{,}8$ com alguma já mantida. Depois, o **VIF** (diagonal da inversa da
  matriz de correlação) remove, uma a uma, a variável de maior VIF até todas ficarem abaixo de 5.
- **Correção de *priors***: pesos de classe, oversampling e SMOTE treinam como se as 6 classes fossem igualmente
  frequentes. As probabilidades voltam para a frequência real do treino (Saerens et al., 2002):
  $p_k \propto \hat p_k \cdot \pi_k / (1/6)$.

---

## 4. Métricas

Para cada ano *h* (classe *h* contra o resto) e para "faliu":

- **PR-AUC** (*average precision*): área sob a curva precisão × recall. Com classes raras, é mais informativa que
  a ROC-AUC, e a linha de base de um modelo aleatório é a **prevalência** (0,6% a 1,2% por ano; 4,8% para "faliu").
- **ROC-AUC**: probabilidade de uma falida aleatória ter escore maior que uma não falida aleatória (base 0,5).
- **Recall a 5% de falsos positivos**: fração das falidas detectadas quando o limiar marca 5% das não falidas.

**Critério de escolha: PR-AUC média dos anos 1 a 5.** Também: log loss e Brier multiclasse, Brier e ECE por
horizonte (calibração).

---

## 5. Regressão logística (notebook 03)

1. **Fatorial completo de pré-processamento**: 4 escalas × 2 reduções × 4 desbalanceamentos = 32 combinações,
   regressão multinomial com L2 e $C = 1$, avaliadas na CV (divisão 0).
2. **Busca de hiperparâmetros** com o melhor pré-processamento: 15 sorteios de $C \sim$ log-uniforme$(10^{-4}, 10^2)$
   e penalidade L1 ou L2.
3. **Seleção robusta** (seção 7): os 3 melhores candidatos distintos dos passos 1 e 2 vão para a CV repetida 3 × 5
   e para os testes pareados; o vencedor é reajustado em todo o treino.

---

## 6. MLP (notebook 04)

**Arquitetura**: camadas densas em funil (cada camada com metade dos neurônios da anterior) → *batch normalization*
→ ReLU → *dropout*; saída *softmax* com 6 classes; perda de entropia cruzada; AdamW; recorte do gradiente em 5.

**Etapas:**

1. **Pré-processamento, um fator por vez** (escala → redução → desbalanceamento), com a MLP base 256-128-64.
2. **Busca aleatória de hiperparâmetros** (15 configurações): número de camadas (1–4), largura (64–512), *dropout*
   (0,05–0,5), taxa de aprendizado (3·10⁻⁴–3·10⁻³), *weight decay* (10⁻⁶–10⁻³), lote (128–512).
3. **Treino longo**: a partir da configuração da busca, 14 experimentos com 100 a 5.000 épocas, agendas da taxa
   (platô, constante, **cosseno**) e regularização mais forte (*dropout* 0,5; *weight decay* 10⁻³ ou 10⁻²; rede com o
   dobro de neurônios). Curvas de aprendizado registram, época a época, a perda de treino e a PR-AUC na validação
   interna e no fold da CV.
4. **Seleção robusta** (seção 7) com **média de 5 sementes** e CV repetida 3 × 5.
5. **Modelo final**: 5 redes (sementes 0 a 4) treinadas em todo o treino; a previsão é a média das probabilidades.

**Média de sementes.** Cada rede depende da semente (inicialização, ordem dos lotes, validação interna). A média das
probabilidades de várias redes reduz essa variância — é um *ensemble* simples, sem mudar a arquitetura.

---

## 7. Seleção robusta

**Problema.** Na CV de 5 folds com 1 semente, a PR-AUC média de uma mesma configuração varia cerca de 0,005 só por
trocar a semente, e as melhores configurações costumam diferir menos do que isso. Escolher pelo maior valor nessa
situação é, em boa parte, escolher pelo acaso (e o vencedor tende a parecer melhor do que é — o "otimismo da
seleção").

**Solução**, nos dois notebooks de modelagem:

1. Separar os **candidatos**: os 3 melhores na CV simples (na MLP, mais a configuração da busca como referência;
   ficam de fora os de custo acima de 25 min por CV, isto é, os de 5.000 épocas, que a seção 4 já mostra não
   ganharem dos de 1.000).
2. Reavaliá-los com mais informação: **CV repetida 3 × 5** (15 folds) e, na MLP, **média de 5 sementes** em cada
   fold. O vencedor é o de **maior PR-AUC média nos 15 folds**.
3. Medir se a vantagem do vencedor é real, com dois testes pareados (seção 8): **t corrigido de Nadeau-Bengio**
   nas 15 diferenças por fold e ***bootstrap* pareado por empresa** nas previsões fora da amostra, com correção de
   **Holm** para as comparações com cada candidato.

**Leitura do resultado.** "Diferença significativa: sim" quer dizer que os dois testes rejeitam a igualdade a 5%
depois da correção. "Não" quer dizer que os dados não distinguem os dois candidatos: o vencedor continua sendo o de
maior média, mas a escolha entre eles é praticamente indiferente para o critério.

---

## 8. Os testes estatísticos, um a um

Todos os testes são **pareados**: os dois modelos são avaliados nas **mesmas empresas** (ou nos **mesmos folds**).
Como as duas previsões de uma mesma empresa são muito correlacionadas, a variância da *diferença* é bem menor do que
a soma das variâncias. Por isso um teste pareado detecta diferenças que dois intervalos de confiança separados não
detectam — é isso que os torna "mais potentes". O código está na seção "Funções do projeto" dos notebooks 03, 04
e 05.

### 8.1 Teste t corrigido de Nadeau e Bengio (CV repetida)

**Pergunta:** a diferença média entre dois modelos nos folds da CV é diferente de zero?

Sejam $d_1, \dots, d_J$ as diferenças da métrica (modelo A − modelo B) nos $J = k \cdot r$ folds (aqui $k = 5$,
$r = 3$, $J = 15$), com $\bar d$ e $s^2$ a média e a variância amostral. O teste t comum usaria
$\mathrm{Var}(\bar d) = s^2 / J$, mas os folds **compartilham dados de treino** (cada empresa está no treino de 4 dos
5 folds), então as diferenças são positivamente correlacionadas e esse teste rejeita demais. A correção de Nadeau e
Bengio (2003) usa

$$t = \frac{\bar d}{\sqrt{\left(\dfrac{1}{J} + \dfrac{n_\text{val}}{n_\text{treino}}\right) s^2}}, \qquad
\frac{n_\text{val}}{n_\text{treino}} = \frac{1}{k - 1} = \frac{1}{4},$$

comparado com uma *t* de Student com $J - 1 = 14$ graus de liberdade. O termo $1/4$ domina: com 15 folds, a variância
é inflada em cerca de 4,75 vezes em relação ao teste ingênuo. É um teste conservador, recomendado por Bouckaert e
Frank (2004) para comparar algoritmos com CV repetida.

### 8.2 *Bootstrap* pareado por empresa

**Pergunta:** qual é o intervalo de confiança da diferença na métrica, e ele contém o zero?

1. Sorteiam-se $B$ reamostras **com reposição** das empresas, **estratificadas pela classe** (cada classe mantém o
   seu número de empresas — essencial com classes de 0,6%, para não haver reamostras sem nenhum caso).
2. Em cada reamostra $b$, calcula-se a métrica **dos dois modelos nas mesmas empresas** e a diferença $d_b^*$.
3. IC de 95% = percentis 2,5% e 97,5% de $d^*$. p-valor bilateral = $2 \min\{\Pr(d^* \le 0), \Pr(d^* \ge 0)\}$.

Na seleção robusta, a métrica é a PR-AUC média calculada nas previsões fora da amostra de cada uma das 3 repetições
da CV e depois promediada (a mesma reamostra de empresas vale para as 3). No teste, $B = 2.000$.

### 8.3 Teste de permutação pareado

**Pergunta:** se os dois modelos fossem equivalentes, uma diferença tão grande quanto a observada seria comum?

Sob $H_0$, as previsões dos dois modelos para uma empresa são intercambiáveis. Em cada uma de 2.000 rodadas,
**trocam-se as previsões dos dois modelos em cada empresa com probabilidade 1/2** e recalcula-se a diferença. O
p-valor é

$$p = \frac{1 + \#\{|d_\text{perm}| \ge |d_\text{obs}|\}}{1 + 2000}.$$

Não supõe nenhuma distribuição; o menor p-valor possível é $1/2001 \approx 0{,}0005$.

### 8.4 Teste de DeLong (ROC-AUC)

**Pergunta:** as ROC-AUC de dois escores medidos nas mesmas empresas são iguais?

A ROC-AUC é uma estatística U de Mann-Whitney. DeLong et al. (1988) estimam a matriz de covariância das duas AUC a
partir das "componentes estruturais" de cada falida e de cada não falida; então

$$z = \frac{\widehat{AUC}_1 - \widehat{AUC}_2}{\sqrt{\mathrm{Var}_1 + \mathrm{Var}_2 - 2\,\mathrm{Cov}_{12}}}
\sim N(0, 1).$$

Usamos o algoritmo rápido de Sun e Xu (2014). Como é analítico, não depende de reamostragem.

### 8.5 Teste exato de McNemar (recall a 5% de falsos positivos)

**Pergunta:** no ponto de operação escolhido, um modelo detecta mais falidas que o outro?

Para cada modelo, o limiar é o que dá 5% de falsos positivos **na validação** (previsões fora da amostra) e é aplicado
ao teste. Entre as falidas do teste, só os **pares discordantes** informam: $b$ = detectadas só pela MLP,
$c$ = detectadas só pela logística. Sob $H_0$, $b \sim \mathrm{Binomial}(b + c, 1/2)$; o p-valor é o binomial
exato bilateral.

### 8.6 Correção de Holm para comparações múltiplas

Ao testar várias hipóteses (por exemplo, os 6 alvos: anos 1 a 5 e "faliu"), a chance de pelo menos um falso positivo
cresce. A correção de Holm-Bonferroni ordena os p-valores $p_{(1)} \le \dots \le p_{(m)}$ e usa

$$\tilde p_{(i)} = \max_{j \le i} \min\{1, (m - j + 1)\, p_{(j)}\},$$

controlando a probabilidade de qualquer falso positivo no conjunto em 5%, com mais poder que a correção de
Bonferroni simples. É aplicada dentro de cada família de testes.

### 8.7 Resumo: que teste responde a quê

| Onde | Pergunta | Teste |
|---|---|---|
| Seleção robusta (03 e 04) | o vencedor é melhor que o candidato *c*? | t corrigido (15 folds) + *bootstrap* pareado (OOF), Holm |
| Teste (05) | a MLP tem PR-AUC maior no ano *h* / em "faliu"? | *bootstrap* pareado + permutação pareada, Holm |
| Teste (05) | a MLP tem ROC-AUC maior? | DeLong, Holm |
| Teste (05) | a MLP detecta mais falidas a 5% de FP? | McNemar exato |
| CV repetida (05) | a MLP é melhor nos 15 folds? | t corrigido, Holm |

---

## 9. Comparação final no teste (notebook 05)

1. Tabela treino / validação / teste dos dois modelos finais (treino = dentro da amostra; validação = OOF da CV;
   teste = 20% guardados).
2. Curvas ROC e precisão × recall por ano e para "faliu".
3. Intervalos de *bootstrap* e os **testes da seção 8**.
4. Matrizes de confusão para "faliu" (limiar de 5% de FP escolhido na validação) e para o ano da falência.
5. Redução de dimensão: quantas variáveis o filtro de correlação + VIF mantém em cada fold.
6. **Calibração**: para falir em até 1, 3 e 5 anos, probabilidade média prevista × taxa observada por decil;
   Brier, Brier *skill score* $= 1 - \text{Brier}/\text{Brier}_\text{prevalência}$, ECE e log loss.
7. **Quando a MLP falha**: recall por ano verdadeiro, por número de faltantes, por valores extremos e por porte.
8. Custo computacional.

---

## 10. Alerta por horizonte e viés de tempo (notebook 06)

- **Alerta em até *h* anos**: PR-AUC, ROC-AUC e recall a 5% de FP de $\sum_{j \le h} P(y = j)$, na validação e no
  teste, com IC de *bootstrap*.
- **Viés de tempo**: como o arquivo não entra no modelo, verificamos se o escore acompanha a taxa de falência de cada
  arquivo e se a detecção muda com o ano verdadeiro da falência.

---

## 11. Limitações

- **Censura**: as empresas ativas só "não faliram até o fim do período"; o multiclasse as trata como "não falha em
  nenhum horizonte".
- **Sem identificador de empresa**: a mesma empresa pode estar em vários arquivos e, portanto, no treino e no teste.
- **Períodos diferentes**: falidas e ativas vêm de janelas de tempo diferentes (2000–2012 e 2007–2013).
- **Faltantes informativos** não entram no modelo, por decisão do projeto (nenhuma variável nova).
- A CV repetida e as sementes reduzem, mas não eliminam, o otimismo da seleção; o teste, usado uma única vez, é a
  estimativa final.

---

## 12. Referências

- Bouckaert, R. R.; Frank, E. (2004). Evaluating the replicability of significance tests for comparing learning
  algorithms. *PAKDD*.
- DeLong, E. R.; DeLong, D. M.; Clarke-Pearson, D. L. (1988). Comparing the areas under two or more correlated
  receiver operating characteristic curves. *Biometrics*, 44(3).
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2).
- McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions. *Psychometrika*, 12(2).
- Nadeau, C.; Bengio, Y. (2003). Inference for the generalization error. *Machine Learning*, 52(3).
- Saerens, M.; Latinne, P.; Decaestecker, C. (2002). Adjusting the outputs of a classifier to new a priori
  probabilities. *Neural Computation*, 14(1).
- Sun, X.; Xu, W. (2014). Fast implementation of DeLong's algorithm for comparing the areas under correlated receiver
  operating characteristic curves. *IEEE Signal Processing Letters*, 21(11).
- Tomczak, S. (2016). Polish Companies Bankruptcy [Dataset]. UCI Machine Learning Repository.
- Zięba, M.; Tomczak, S. K.; Tomczak, J. M. (2016). Ensemble boosted trees with synthetic features generation in
  application to bankruptcy prediction. *Expert Systems with Applications*, 58.
