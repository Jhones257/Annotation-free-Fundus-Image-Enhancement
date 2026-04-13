# Kaggle Tutorial: Fine-Tuning da GFE-Net com Dataset Itapecuru

Este tutorial explica como realizar o **fine-tuning** do modelo GFE-Net usando as imagens de retinografia do dataset Itapecuru (câmera Phelcom Eyer), partindo dos pesos originais do autor, rodando no Kaggle com GPU gratuita.

---

## Visão Geral

O GFE-Net é treinado com **pares supervisionados**: imagem degradada + imagem limpa concatenadas lado a lado. Como o dataset Itapecuru possui apenas imagens individuais com rótulos de qualidade (não pares naturais), usamos **degradação sintética** para criar os pares:

1. Imagens `quality=0` (boas) servem como **referência limpa** (ground truth)
2. Degradações sintéticas (blur, spots, iluminação, catarata) são aplicadas para gerar a versão **degradada**
3. Cada par `[degradada | limpa]` é concatenado horizontalmente (1536×768)

O fine-tuning carrega os pesos pré-treinados do autor e continua o treinamento com os dados do domínio Phelcom Eyer, adaptando o modelo às características específicas dessas imagens.

---

## Passo a Passo Completo

### Passo 1: Preparar os Dados de Treinamento (Localmente)

Execute o script de preparação que lê o CSV, filtra as imagens por qualidade/tipo, e gera os pares sintéticos:

```powershell
cd Annotation-free-Fundus-Image-Enhancement

python scripts/prepare_gfenet_training.py `
    --csv itapecuru_all_labeled_v3_com_tipo.csv `
    --images_dir ORIGINAL_CLEAN_768x768 `
    --output_dir datasets/gfenet_finetune `
    --num_degradations 8 `
    --image_size 768
```

**O que isso faz:**
- Filtra imagens `quality=0` + `tipo=color` → **364 imagens limpas**
- Gera **8 variantes degradadas** por imagem → **~2.912 pares de treino**
- Copia imagens `quality=2` + `tipo=color` → **~1.327 imagens para validação**
- Resultado: pasta `datasets/gfenet_finetune/` com a estrutura:

```
datasets/gfenet_finetune/
├── source/         ← pares [degradada|limpa] para treino (~2.912 imagens)
├── source_mask/    ← máscaras binárias dos pares (~2.912 PNGs)
├── target/         ← imagens rejeitadas para validação (~1.327 imagens)
└── target_mask/    ← máscaras das imagens de validação (~1.327 PNGs)
```

**Tipos de degradação aplicados:**
| Código | Tipo | Descrição |
|--------|------|-----------|
| `1000` | Blur | Desfoque gaussiano |
| `0100` | Spots | Artefatos pontuais (sujeira na lente) |
| `0010` | Iluminação | Brilho/contraste irregular, halo, sombras |
| `0110` | Spots + Iluminação | Combinação |
| `1010` | Blur + Iluminação | Combinação |
| `1100` | Blur + Spots | Combinação |
| `1110` | Blur + Spots + Iluminação | Todas as degradações |
| `0001` | Catarata | Simulação de nebulosidade por catarata |

**Parâmetros opcionais:**
- `--quality_train 0 1`: usar também imagens `quality=1` como referência (mais dados, menor qualidade de referência)
- `--num_degradations 4`: menos variantes por imagem (mais rápido, menos dados)
- `--tipo color eye`: incluir também imagens do tipo `eye`
- `--overwrite`: regenerar dados existentes

> **Tempo estimado**: ~15-40 minutos dependendo do hardware.

---

### Passo 2: Verificar os Dados Gerados

Antes de empacotar, valide que os dados foram gerados corretamente:

```powershell
# Verificar contagem de arquivos
(Get-ChildItem datasets/gfenet_finetune/source -File).Count
(Get-ChildItem datasets/gfenet_finetune/source_mask -File).Count
(Get-ChildItem datasets/gfenet_finetune/target -File).Count
(Get-ChildItem datasets/gfenet_finetune/target_mask -File).Count
```

Verifique visualmente que as imagens em `source/` têm largura 1536px (degradada à esquerda, limpa à direita).

---

### Passo 3: Empacotar para o Kaggle

> **Dica**: para referência futura, lembre-se do passo 1 e 2 são feitos localmente. Do passo 3 em diante é no Kaggle.

```powershell
# Limpar arquivos desnecessários
Remove-Item -Recurse -Force results, __pycache__ -ErrorAction SilentlyContinue
Get-ChildItem -Recurse -Filter "__pycache__" -Directory | Remove-Item -Recurse -Force

# Verificar que os pesos originais estão presentes
Test-Path checkpoints/gfenet/latest_net_G.pth  # Deve retornar True

# Compactar (subir um nível antes)
cd ..
Compress-Archive -Path .\Annotation-free-Fundus-Image-Enhancement -DestinationPath .\AFE_finetune.zip -Force
```

**Tamanho esperado**: ~3-8 GB dependendo da quantidade de pares gerados.

Se o ZIP ficar muito grande (>20GB, limite do Kaggle), considere:
- Reduzir `--num_degradations` para 4
- Usar apenas `--quality_target` vazio (sem imagens de validação no ZIP)
- Comprimir imagens com menor qualidade JPEG

---

### Passo 4: Upload para o Kaggle

1. Acesse [kaggle.com/datasets](https://www.kaggle.com/datasets) → **New Dataset**
2. Faça upload do `AFE_finetune.zip`
3. Nome sugerido: `AFE GFENet Finetune - Itapecuru`
4. Publique (pode ser privado)
5. Anote o slug: `seuusuario/afe-gfenet-finetune`

---

### Passo 5: Criar e Configurar o Notebook Kaggle

1. Acesse [kaggle.com/code](https://www.kaggle.com/code) → **New Notebook**
2. Configure:
   - **Accelerator**: GPU T4 ×2 ou GPU P100
   - **Internet**: ON (para pip install, se necessário)
3. Em **Add Data**, adicione o dataset do passo 4
4. Importe o notebook `notebooks/kaggle_gfenet_finetune.ipynb` fornecido no repositório, **OU** copie as células manualmente

---

### Passo 6: Executar o Fine-Tuning no Kaggle

No notebook, ajuste `KAGGLE_DATASET_SLUG` e execute as células em sequência:

1. **Setup**: Descompacta, instala deps
2. **Verificação**: Confirma estrutura de dados e pesos
3. **Cópia de pesos**: Copia `latest_net_G.pth` para `checkpoints/gfenet_finetune/`
4. **Treinamento**: Executa `train.py` com fine-tuning

**Comando de treinamento usado internamente:**
```bash
python train.py \
    --dataroot datasets/gfenet_finetune \
    --name gfenet_finetune \
    --model gfenet \
    --dataset_mode cataract_with_mask \
    --direction AtoB \
    --norm instance \
    --input_nc 6 \
    --load_size 768 --crop_size 768 \
    --batch_size 4 \
    --gpu_ids 0 \
    --lr 0.0005 \
    --lr_policy linear \
    --n_epochs 30 \
    --n_epochs_decay 10 \
    --save_epoch_freq 10 \
    --continue_train \
    --epoch latest \
    --display_id -1 \
    --no_flip
```

**Justificativa dos hiperparâmetros para fine-tuning:**
| Parâmetro | Original | Fine-Tuning | Motivo |
|-----------|----------|-------------|--------|
| `lr` | 0.002 | **0.0005** | 4× menor para não destruir features pré-aprendidas |
| `n_epochs` | 150 | **30** | Menos epochs necessários (partindo de pesos bons) |
| `n_epochs_decay` | 50 | **10** | Decay mais curto |
| `batch_size` | 8 | **4** | Imagens 768×768 consomem mais VRAM |
| `name` | gfenet | **gfenet_finetune** | Não sobrescrever pesos originais |
| `continue_train` | - | **sim** | Carrega pesos pré-treinados |
| `no_flip` | não | **sim** | Reduz augmentação para estabilidade no fine-tuning |

---

### Passo 7: Avaliar e Exportar Resultados

Após o treinamento, o notebook executa:

1. **Inferência com modelo fine-tunado** nas imagens `quality=2`
2. **Inferência com modelo original** para comparação
3. **Comparação visual** lado a lado (input × original × fine-tunado)
4. **Exportação** dos pesos `.pth` e resultados como ZIP

Baixe o `outputs.zip` pela aba **Output** do notebook no Kaggle.

---

## Troubleshooting

| Problema | Solução |
|----------|---------|
| **OOM (Out of Memory)** | Reduza `--batch_size` para 2 ou 1 |
| **Loss não converge** | Reduza o LR (ex: 0.0002) ou aumente epochs |
| **Loss explode** | LR muito alto; tente 0.0001 |
| **Pesos não encontrados** | Verifique que `checkpoints/gfenet_finetune/latest_net_G.pth` existe |
| **Dataset vazio** | Verifique que `datasets/gfenet_finetune/source/` tem arquivos |
| **ZIP muito grande** | Reduza `--num_degradations` ou exclua `target/` do ZIP |
| **Tempo de GPU esgotado** (12h) | Reduza epochs ou use `--save_epoch_freq 5` para checkpoints mais frequentes |
| **`ModuleNotFoundError`** | Certifique-se de rodar `pip install -r requirements.txt` |

---

## Distribuição dos Dados do Itapecuru

| quality × tipo | color | eye | gray | Total |
|----------------|-------|-----|------|-------|
| **0 (boa)** | 364 | 254 | 188 | 806 |
| **1 (usável)** | 961 | 307 | 642 | 1.910 |
| **2 (rejeitada)** | 1.327 | 249 | 4.490 | 6.066 |
| **Total** | 2.652 | 810 | 5.320 | 8.782 |

O fine-tuning padrão usa as **364 imagens** quality=0/color como referência limpa, gerando **~2.912 pares** de treino com 8 degradações cada.
