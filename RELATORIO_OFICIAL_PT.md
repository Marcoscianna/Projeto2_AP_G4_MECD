# Relatorio Tecnico - Projeto 2 (Aprendizagem Profunda)

## 1. Enquadramento
Este trabalho foi desenvolvido no ambito da UC de Aprendizagem Profunda, com foco na classificacao automatica de imagens fluoroscopicas de CPRE. O problema consiste em atribuir cada imagem a uma de quatro classes clinicas: Biliary_Leaks, Lithiasis, Normal e Stricture.

O criterio principal de avaliacao e o F1-score macro no conjunto de teste, sendo a baseline de referencia 0.738.

## 2. Equipa
- Grupo: G4
- Elementos:
  - pg60004 - Filipa Oliveira da Silva
  - pg60099 - Francisco Ricardo Teixeira Silva
  - pg58750 - Lara Cristiana Sousa Antunes
  - e12961 - Marco Scianna

## 3. Dados e preparacao
O trabalho utiliza o dataset MIQR-CC, organizado em treino, validacao e teste. Em todas as etapas foi mantida a separacao entre os tres subconjuntos para evitar fuga de informacao.

No pre-processamento, foi adotada uma pipeline orientada ao dominio medico:
- melhoramento de contraste local com CLAHE;
- redimensionamento uniforme das imagens;
- normalizacao para entrada em redes pre-treinadas;
- augmentacao apenas no treino (flip horizontal, rotacao e color jitter).

Este conjunto de transformacoes foi escolhido para aumentar a robustez do modelo a variacoes de contraste e ruido comuns em fluoroscopia.

## 4. Metodologia
Foi utilizada uma arquitetura EfficientNetV2-S via timm, com transfer learning. A opcao por este modelo deve-se ao bom equilibrio entre capacidade representacional e custo computacional.

Configuracao principal de treino (a confirmar na versao final executada):
- resolucao: 224x224;
- batch size: 16;
- otimizador: AdamW;
- scheduler: CosineAnnealingLR;
- loss: CrossEntropy com class weights;
- early stopping com patience = 10.

O uso de pesos por classe foi importante para lidar com o desbalanceamento observado no dataset.

## 5. Resultados
### 5.1 Resultado principal
- F1-macro (teste): [preencher]
- Baseline: 0.738
- Diferenca face a baseline: [preencher]

### 5.2 Metricas complementares
- Accuracy (teste): [preencher]
- AUC-ROC macro: [preencher]
- Desempenho por classe (classification report): [resumir]

### 5.3 Evidencia visual
Foram gerados os seguintes artefactos de suporte:
- curvas de treino (loss e F1);
- matriz de confusao no conjunto de teste;
- mapas Grad-CAM para interpretabilidade.

## 6. Discussao
Pontos a destacar na versao final:
- classes com melhor e pior desempenho;
- tipos de confusao mais frequentes;
- impacto esperado de CLAHE e class weights;
- limitacoes da abordagem atual e melhorias futuras.

## 7. Conclusao
O objetivo do projeto e obter desempenho competitivo na classificacao multi-classe de imagens de CPRE e, idealmente, superar a baseline proposta.

Conclusao final apos execucao completa:
- Baseline superada? [sim/nao]
- Valor final de F1-macro: [preencher]
- Ganho face a baseline: [preencher]

## 8. Reprodutibilidade
Ambiente:
- Python: [preencher]
- Dependencias: `requirements.txt`

Execucao minima:
1. Instalar dependencias: `pip install -r requirements.txt`
2. Definir dataset, quando necessario: `export MIQR_DATASET_DIR="/caminho/para/dataset"`
3. Executar `MIQR_CC_solution.ipynb` por ordem.

## 9. Entregaveis
- Notebook final: `MIQR_CC_solution.ipynb`
- Modelo treinado (`.pth`)
- Figuras de avaliacao (curvas, matriz de confusao, Grad-CAM)
- Relatorio final em PDF
