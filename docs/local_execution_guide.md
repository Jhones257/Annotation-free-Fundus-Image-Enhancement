# Guia Local Completo: Annotation-free Fundus Image Enhancement

Este documento descreve como executar o repositório localmente, explicando todas as arquiteturas disponíveis (ArcNet, SCR-Net, GFENet e utilitários auxiliares), seus casos de uso e os passos para treinamento e inferência.

---

## 1. Arquiteturas e finalidades

| Modelo | Foco | Quando usar |
| --- | --- | --- |
| **ArcNet** (`models/arcnet_model.py`) | Restauração de imagens cataratosas via *unsupervised domain adaptation* (UDA). | Quando você possui imagens cataratosas (domínio fonte) e imagens limpas não pareadas (domínio alvo) e precisa reduzir o efeito de baixa transparência. Necessita treinamento específico no seu conjunto (os pesos oficiais não generalizam bem para novos domínios). |
| **SCR-Net** (`models/scrnet_model.py`) | *Structure-consistent restoration*: preserva a morfologia do fundo de olho enquanto remove degradações. | Adequado para cenários com máscaras binárias do disco óptico e necessidade de preservar estruturas anatômicas. Treine quando possuir pares fonte-alvo ou dados com máscaras confiáveis. |
| **GFENet** (`models/gfenet_model.py`) | *Generic Fundus Enhancement* com auto-supervisão em frequência; aceita máscaras de catarata e gera duas saídas (imagem restaurada e componente de alta frequência). | Ideal para inferência em imagens degradadas sem ground truth, desde que você forneça máscaras alinhadas. Possui pesos oficiais (`checkpoints/gfenet/latest_net_G.pth`) para replicar o artigo. |
| **Pix2Pix / CycleGAN / Base** | Mantidos do repositório original do PyTorch CycleGAN + Pix2Pix, usados como utilitários internos para ArcNet/SCRNet/GFENet. | Use apenas se desejar treinar modelos genéricos adicionais. |

Arquivos de referência:
- `models/base_model.py`: define o ciclo de treinamento/validação comum.
- `models/networks.py`: implementa geradores/discriminadores (UNet, ResNet, GFENet backbone etc.).
- `data/*_dataset.py`: loaders customizados (ex.: `cataract_with_mask_dataset` para GFENet/SCR-Net).

---

## 2. Preparar o ambiente local

1. **Requisitos**
   - Windows 10/11 ou Linux
   - Python ≥ 3.8
   - GPU NVIDIA + CUDA/cuDNN (recomendado)

2. **Criar ambiente (exemplo com conda)**
   ```bash
   conda create -n fundus python=3.9 -y
   conda activate fundus
   pip install -r requirements.txt
   ```

   > O README também oferece comandos equivalentes com `conda install`. Caso esteja sem GPU, remova dependências de CUDA dos pacotes PyTorch e instale uma versão `cpuonly`.

3. **Dependências adicionais**
   - `visdom` e `dominate` para visualização (`pip install visdom dominate`).
   - `gdown` (opcional) para baixar pesos do Google Drive (`pip install gdown`).

---

## 3. Preparar datasets

### 3.1 Estrutura geral
Para ArcNet/SCR-Net/GFENet o repositório usa subpastas específicas. Exemplo GFENet (`cataract_with_mask_dataset`):
```
<dataroot>
  source/
  source_mask/
  target/<imagens RGB>
  target_mask/<máscaras PNG>
```
Para inferência apenas `target/` e `target_mask/` são usados, mas as pastas `source*` devem existir.

### 3.2 Geração automática de máscaras
Use o script criado neste repositório:
```bash
python scripts/prepare_gfenet_inference.py \
    --input_dir ORIGINAL_CLEAN_768x768 \
    --output_dir datasets/my_gfenet_eval
```
- Copia as imagens mantendo a hierarquia.
- Cria máscaras binárias com `util/get_mask.py`.
- Garante a existência das pastas `source/` e `source_mask/` vazias.

### 3.3 Simulações de baixa qualidade
Para reproduzir os experimentos originais:
- `data/get_low_quality/run_pre_process.py`: normaliza e recorta.
- `data/get_low_quality/main_degradation.py`: aplica degradações sintéticas.
- `util/get_mask.py`: gera máscara para imagens de catarata ou alvos.

---

## 4. Baixar pesos oficiais

| Modelo | Caminho esperado |
| --- | --- |
| ArcNet | `checkpoints/arcnet/latest_net_G.pth` |
| SCR-Net | `checkpoints/scrnet/latest_net_G.pth` |
| GFENet | `checkpoints/gfenet/latest_net_G.pth` |

Baixe via links do README (`Google Drive` ou `Baidu`). Exemplo com `gdown`:
```bash
mkdir -p checkpoints/gfenet
cd checkpoints/gfenet
gdown https://drive.google.com/uc?id=1cerN6u0aRKr1aiNl31pt7hdALSFJjt7i -O latest_net_G.pth
```

---

## 5. Executar cada arquitetura localmente

### 5.1 ArcNet
- **Objetivo:** adaptar imagens cataratosas ao domínio limpo usando dados não pareados.
- **Treinamento:**
  ```bash
  python train.py --dataroot ./images/cataract_dataset \
      --name arcnet --model arcnet --netG unet_256 \
      --input_nc 6 --direction AtoB \
      --dataset_mode cataract_guide_padding --norm batch \
      --batch_size 8 --gpu_ids 0
  ```
- **Inferência:**
  ```bash
  python test.py --dataroot ./images/cataract_dataset \
      --name arcnet --model arcnet --netG unet_256 \
      --input_nc 6 --direction AtoB \
      --dataset_mode cataract_guide_padding --norm batch \
      --gpu_ids 0
  ```

### 5.2 SCR-Net
- **Objetivo:** restaurar mantendo consistência estrutural com apoio de máscaras.
- **Treinamento:**
  ```bash
  python train.py --dataroot ./images/cataract_dataset \
      --name scrnet --model scrnet --input_nc 3 \
      --direction AtoB --dataset_mode cataract_with_mask \
      --norm instance --batch_size 8 --gpu_ids 0 \
      --lr_policy linear --n_epochs 150 --n_epochs_decay 50
  ```
- **Inferência:**
  ```bash
  python test.py --dataroot ./images/cataract_dataset \
      --name scrnet --model scrnet --netG unet_combine_2layer \
      --direction AtoB --dataset_mode cataract_with_mask \
      --input_nc 3 --output_nc 3
  ```

### 5.3 GFENet
- **Objetivo:** pipeline genérico de realce (sem pares) usando componentes de alta frequência.
- **Treinamento (caso queira atualizar pesos):**
  ```bash
  python train.py --dataroot ./images/cataract_dataset \
      --name train_gfenet --model gfenet --direction AtoB \
      --dataset_mode cataract_with_mask --norm instance \
      --batch_size 8 --gpu_ids 0 --lr_policy linear \
      --n_epochs 150 --n_epochs_decay 50
  ```
- **Inferência com pesos oficiais:**
  ```bash
  python test.py \
      --dataroot datasets/my_gfenet_eval \
      --name gfenet --model gfenet \
      --dataset_mode cataract_with_mask \
      --load_size 768 --crop_size 768 \
      --results_dir ./results \
      --num_test 100000 \
      --preserve_subfolders --eval
  ```
  - `--preserve_subfolders`: utiliza a modificação realizada no projeto para manter a hierarquia original no diretório `results/`.
  - `--eval`: garante comportamento determinístico do BatchNorm.

### 5.4 Ciclo completo com Visdom (opcional)
- Inicie o servidor para acompanhar métricas:
  ```bash
  python -m visdom.server
  ```
- Acesse `http://localhost:8097` para visualizar perdas e imagens em tempo real.

---

## 6. Pós-processamento e resultados

- Os outputs ficam em `results/<name>/<phase>_<epoch>/images`. Cada entrada gera múltiplos arquivos (`*_real*`, `*_fake*`, `*_HFC`).
- Um `index.html` é criado automaticamente para inspeção visual rápida.
- Para converter as imagens processadas de volta aos nomes originais, use scripts auxiliares (por exemplo, mover `*_fake_TB.png` para uma nova pasta/arquivo zip) conforme sua necessidade.

---

## 7. Boas práticas

1. **Backups**: mantenha um diretório dedicado para checkpoints próprios (`checkpoints/<experimento>`). Evite sobrescrever os pesos originais.
2. **Logs**: confira `checkpoints/<name>/loss_log.txt` para depurar perdas durante o treino.
3. **Testes rápidos**: defina `--num_test 10` em `test.py` para validar configuração antes de processar todo o dataset.
4. **Gerenciamento de VRAM**: reduza `--load_size` e `--crop_size` ou use `--gpu_ids -1` (modo CPU) quando necessário, sabendo que a execução será mais lenta.
5. **Reprodutibilidade**: configure seeds (`torch.manual_seed`) se quiser resultados idênticos em múltiplas execuções.

---

## 8. Resumo

- **ArcNet**: domínio não pareado, re-treine para cada novo cenário.
- **SCR-Net**: preserva estrutura com máscaras; ideal quando o alvo precisa manter topologia.
- **GFENet**: melhor opção para inferência direta com pesos oficiais, contanto que você gere máscaras alinhadas.
- Scripts auxiliares (`prepare_gfenet_inference.py`, `util/get_mask.py`) simplificam a preparação de dados.

> **💡 Refência Completa de Comandos:** Acesse [docs/cli_reference.md](cli_reference.md) para ver **todas as opções** (flags) de linha de comando aceitas e conferir **comandos avançados** para continuar treinamentos (`--continue_train`), usar Weights & Biases (`--use_wandb`) ou mudar resoluções sem perder qualidade.

Com este guia detalhado você consegue montar o ambiente local, preparar datasets, escolher a arquitetura correta e executar os pipelines completos de treinamento e inferência para cada abordagem disponível no repositório. Bons experimentos!
