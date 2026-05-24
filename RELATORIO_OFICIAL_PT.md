# Relatorio Tecnico - Projeto 2 (Aprendizagem Profunda)

## 1. Enquadramento
Este trabalho foi desenvolvido no ambito da UC de Aprendizagem Profunda, com foco na classificacao automatica de imagens fluoroscopicas de CPRE. O problema consiste em atribuir cada imagem a uma de quatro classes clinicas: Biliary_Leaks, Lithiasis, Normal e Stricture.

O criterio principal de avaliacao e o F1-score macro no conjunto de teste, com baseline de referencia igual a 0.738.

## 2. Equipa
- Grupo: G4
- Elementos:
  - pg60004 - Filipa Oliveira da Silva
  - pg60099 - Francisco Ricardo Teixeira Silva
  - pg58750 - Lara Cristiana Sousa Antunes
  - e12961 - Marco Scianna

## 3. Estado atual do trabalho
Ao longo da iteracao do projeto foi feito um ciclo completo de limpeza, correcao e experimentacao no notebook principal de treino:

- limpeza de celulas redundantes e alinhamento do notebook com a estrutura da entrega;
- correcao do carregamento de checkpoints no modulo `checkpoint_utils.py`;
- substituicao de modelos sem pesos pre-treinados disponiveis por variantes suportadas pelo `timm`;
- validacao do fluxo de treino e avaliacao em GPU;
- experimentacao com diferentes estrategias de balanceamento e fine-tuning;
- registo dos melhores checkpoints e das metricas obtidas em validacao.

O notebook de referencia de treino foi progressivamente simplificado para manter apenas o fluxo util: importacao, EDA, dataset, modelo, treino, avaliacao e Grad-CAM.

## 4. Dados e preparacao
O trabalho utiliza o dataset MIQR-CC, organizado em treino, validacao e teste. Em todas as etapas foi mantida a separacao entre os tres subconjuntos para evitar fuga de informacao.

O pre-processamento final adotado foi orientado ao dominio medico:

- melhoramento de contraste local com CLAHE;
- redimensionamento uniforme das imagens para 224x224;
- normalizacao com os parametros classicos de ImageNet para reutilizacao de pesos pre-treinados;
- augmentacao moderada apenas no treino, limitada a flip horizontal;
- sem oversampling no dataloader final, recorrendo apenas a class weights na loss.

Esta escolha resultou de varios testes: o uso simultaneo de oversampling e class weights mostrou-se instavel e favoreceu overfitting precoce.

## 5. Metodologia
### 5.1 Modelo e transfer learning
Foram testados varios backbones pre-treinados via `timm`. As variantes verificadas no ambiente foram:

- `tf_efficientnetv2_s`
- `tf_efficientnetv2_m`
- `convnext_tiny`
- `convnext_small`

O modelo usado nos ultimos testes foi `tf_efficientnetv2_s`, com pesos pre-treinados, porque ofereceu melhor equilibrio entre estabilidade de treino e capacidade de generalizacao no dataset.

### 5.2 Estrategia de treino
A configuracao final de treino foi ajustada iterativamente para reduzir overfitting:

- resolucao: 224x224;
- batch size: 16;
- otimizador: AdamW;
- scheduler: CosineAnnealingLR;
- loss: CrossEntropyLoss com class weights;
- regularizacao no backbone com `drop_rate=0.2` e `drop_path_rate=0.2`;
- warmup com treino apenas da cabeca durante 12 epocas;
- unfreeze posterior do backbone com learning rate muito mais baixo;
- early stopping com patience = 10.

O treino passou a usar grupos de parametros distintos para cabeca e backbone, permitindo reduzir o LR do backbone apos o warmup sem alterar a logica global do notebook.

### 5.3 Ferramentas de interpretabilidade
Foi adicionada uma secao de Grad-CAM para inspecionar as regioes da imagem usadas pelo modelo nas predicoes finais. Isto permitiu produzir mapas de calor por classe e melhorar a leitura clinica dos resultados.

## 6. Resultados obtidos ate agora
Os melhores resultados observados durante a fase de experimentacao ainda nao superaram a baseline de 0.738 no conjunto de teste.

### 6.1 Melhor fase observada no treino
Nos ultimos testes com `tf_efficientnetv2_s`, a validacao atingiu o melhor valor em torno de:

- melhor F1 macro em validacao: 0.5660

Este valor correspondeu ao melhor checkpoint observado durante o treino, mas ainda ficou abaixo do objetivo da baseline.

### 6.2 Comportamento do treino
Os logs mostraram um padrao consistente:

- subida rapida do desempenho no treino;
- melhoria inicial na validacao durante o warmup;
- degradacao ou estagnacao apos o backbone ser parcialmente desbloqueado;
- overfitting marcado quando a capacidade de treino era demasiado agressiva.

Isto levou a varias iteracoes de correcao:

- remocao de oversampling;
- uso apenas de class weights na loss;
- reducao de augmentation;
- passagem para um backbone mais adequado;
- congelamento inicial do backbone;
- reducao do learning rate apos o unfreeze.

### 6.3 Estado final atual
O ultimo estado estabilizado do notebook inclui:

- `tf_efficientnetv2_s` como backbone;
- train loader sem sampler;
- class weights na loss;
- warmup com backbone congelado;
- fine-tuning posterior com LR reduzido.

O melhor checkpoint identificado ate ao momento continua abaixo da baseline, pelo que o projeto permanece em fase de refinamento experimental.

## 7. Artefactos produzidos
Foram gerados e/ou mantidos os seguintes artefactos de suporte:

- curvas de treino com loss e F1;
- matriz de confusao no conjunto de teste;
- relatorio de classificacao por classe;
- estimativa de AUC-ROC macro;
- mapas Grad-CAM por classe;
- checkpoints `.pth` do melhor modelo observado em validacao.

## 8. Discussao
As principais conclusoes tecnicas extraidas ate ao momento sao:

- o dataset e suficientemente desbalanceado para justificar class weights;
- oversampling adicional nao trouxe ganho consistente e agravou a instabilidade;
- a regularizacao ajudou a estabilizar o treino, mas ainda nao foi suficiente para ultrapassar a baseline;
- o comportamento do modelo sugere que a selecao do backbone e a forma de unfreeze sao os fatores mais sensiveis;
- o melhor checkpoint atual ainda necessita de mais iteracao para atingir a meta de desempenho.

## 9. Conclusao
O objetivo do projeto continua a ser obter desempenho competitivo na classificacao multi-classe de imagens de CPRE e ultrapassar a baseline proposta.

Conclusao final do estado atual:

- Baseline superada? nao
- Melhor F1 macro observado em validacao: 0.5660
- Ganho face a baseline de teste: ainda nao obtido

Apesar de o resultado atual nao superar a baseline, o trabalho ja consolidou uma pipeline completa e reproduzivel, com treino em GPU, checkpointing, avaliacao detalhada e interpretabilidade.

## 10. Reprodutibilidade
Ambiente:

- Python: 3.12.2
- Biblioteca de treino: PyTorch + torchvision
- Modelos: `timm`
- Metricas: scikit-learn

Execucao minima:

1. Instalar dependencias: `pip install -r requirements.txt`
2. Garantir a estrutura do dataset em `../dataset`
3. Executar `MIQR_CC_solution.ipynb` ou `mine.ipynb` por ordem
4. Aguardar a escrita do melhor checkpoint em `../models/`

## 11. Entregaveis
- Notebook de referencia e experimentacao
- Modelo treinado (`.pth`)
- Figuras de avaliacao
- Grad-CAM
- Relatorio final em PDF
