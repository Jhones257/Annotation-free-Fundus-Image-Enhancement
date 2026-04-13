# Guia de Referência de Linha de Comando (CLI)

Este documento contém a lista exaustiva de todas as opções de linha de comando aceitas pelos scripts de treinamento (`train.py`) e teste/inferência (`test.py`), além de exemplos avançados de uso.

## 1. Modelos Disponíveis (`--model`)
- **arcnet**: Modelo ArcNet (`arcnet_model.py`)
- **cycle_gan**: Modelo CycleGAN (`cycle_gan_model.py`)
- **gfenet**: Modelo GFENet (`gfenet_model.py`)
- **pix2pix**: Modelo Pix2Pix (`pix2pix_model.py`)
- **pixDA_sobel**: Modelo PixDA Sobel (`pixDA_sobel_model.py`)
- **scrnet**: Modelo SCR-Net (`scrnet_model.py`)
- **test**: Modelo padrão para teste (pass-through)

## 2. Datasets Disponíveis (`--dataset_mode`)
- **cataract**: Dataset básico de catarata
- **cataract_guide_padding**: Dataset com padding contendo o mapa de guia (padrão)
- **cataract_with_mask**: Dataset com máscara de segmentação
- **aligned**: Imagens pareadas (alinhadas)
- **image_folder**: Pasta genérica de imagens

## 3. Todas as Opções (Flags)

### Opções Básicas (Comuns para Treino e Teste)
| Argumento | Padrão | Descrição |
|-----------|---------|-------------|
| `--dataroot` | *requerido* | Caminho das imagens (ex: pastas trainA, trainB, etc.) |
| `--name` | `experiment_name` | Nome do experimento (salva checkpoints e resultados com este nome) |
| `--gpu_ids` | `0` | IDs das GPUs (ex: `0`, `0,1,2`. Use `-1` para CPU) |
| `--checkpoints_dir` | `./checkpoints` | Diretório onde os modelos são salvos |
| `--results_dir` | `./results/` | Diretório onde os resultados são salvos |
| `--phase` | `train` / `test` | Fase atual da execução |

### Parâmetros de Arquitetura (Modelo)
| Argumento | Padrão | Descrição |
|-----------|---------|-------------|
| `--model` | `arcnet` | Qual modelo utilizar |
| `--input_nc` | `6` | Número de canais de entrada (3=RGB, 1=Gray, 6=RGB+Guide) |
| `--output_nc` | `3` | Número de canais de saída |
| `--ngf` | `64` | Filtros do gerador na última camada convolucional |
| `--ndf` | `64` | Filtros do discriminador na primeira camada convolucional |
| `--netG` | `unet_256` | Arquitetura do gerador (`resnet_9blocks`, `unet_256`, `unet_128`, etc.) |
| `--netD` | `basic` | Arquitetura do discriminador (`basic`, `n_layers`, `pixel`) |
| `--norm` | `batch` | Tipo de normalização (`instance`, `batch`, `none`) |
| `--init_type` | `normal` | Inicialização de pesos (`normal`, `xavier`, `kaiming`, `orthogonal`) |
| `--no_dropout` | - | Desativa dropout no gerador se presente |

### Parâmetros de Dataset
| Argumento | Padrão | Descrição |
|-----------|---------|-------------|
| `--dataset_mode` | `cataract_guide_padding`| Como carregar o dataset |
| `--direction` | `AtoB` | Direção do mapeamento (AtoB ou BtoA) |
| `--batch_size` | `1` | Tamanho do lote (batch) |
| `--load_size` | `306` | Tamanho para redimensionar imagens de entrada |
| `--crop_size` | `256` | Tamanho real das imagens cortadas (crop) passadas para a rede |
| `--preprocess` | `resize_and_crop` | Tipo de pré-processamento |
| `--no_flip` | - | Desativa flip horizontal de data augmentation |
| `--serial_batches` | - | Remove aleatoriedade de lotes (não embaralha) |
| `--num_threads` | `4` | Threads para carregar dados (0 para principal) |
| `--preserve_subfolders` | - | Mantém estrutura de pastas original em `--results_dir` |

### Parâmetros de Treinamento
| Argumento | Padrão | Descrição |
|-----------|---------|-------------|
| `--n_epochs` | `100` | Número de épocas com learning rate inicial (constante) |
| `--n_epochs_decay` | `100` | Número de épocas de decaimento linear da learning rate até 0 |
| `--lr` | `0.0002` | Learning rate inicial para Adam |
| `--beta1` | `0.5` | Momentum do Adam |
| `--gan_mode` | `lsgan` | Função de perda do GAN (`vanilla`, `lsgan`, `wgangp`) |
| `--lr_policy` | `linear` | Política de learning rate (`linear`, `step`, `plateau`, `cosine`) |
| `--save_latest_freq`| `5000` | A cada N iterações, grava o modelo mais recente |
| `--save_epoch_freq` | `30` | A cada N épocas, salva explicitamente o modelo na nuvem |
| `--continue_train`| - | Retorna ao `--name` e continua da última época salva |

### Parâmetros de Inferência e Teste
| Argumento | Padrão | Descrição |
|-----------|---------|-------------|
| `--eval` | - | Ativa modo `eval` fixando running stats no BatchNorm / desativando dropout |
| `--num_test` | `126` | Quantidade máxima de imagens avaliadas num batch |
| `--epoch` | `latest` | Identificador de qual modelo processar (ex: `100`, `latest`) |
| `--target_gt_dir`| `target_gt` | Pasta com o ground-truth do alvo se precisar checar métricas |

### Weights & Biases e Visdom (Display)
| Argumento | Padrão | Descrição |
|-----------|---------|-------------|
| `--display_freq` | `400` | Frequência de exibição do progresso de treino |
| `--use_wandb` | - | Habilita tracking inteligente via Weights & Biases |
| `--wandb_project`| `fundus-enhancement` | Nome do projeto na dashboard W&B |
| `--wandb_mode` | `online` | Modo (`online`, `offline`, `disabled`) |
| `--display_id` | `1` | ID base para servidor Visdom (0 desativa Visdom) |
| `--display_server` | `http://localhost` | Endereço do painel do Visdom |
| `--display_port` | `8097` | Porta do portal do Visdom |


## 4. Comandos e Combinações Avançadas

### Treinamento Avançado

**Continuar de onde parou:**
Use a flag `--continue_train` certificando-se de fornecer rigorosamente o mesmo `--name`.
```bash
python train.py --dataroot ./datasets/fundus --name gfenet_experimento_01 --model gfenet \
    --continue_train --epoch_count 120
```

**Treinamento Rápido em GPU Específica ou Múltiplas:**
Delegue cálculos para uma GPU alternativa (ex: GPU 1) ou treine em várias (0, 1 e 2). Para CPU forçado use `-1`.
```bash
python train.py --dataroot ./datasets/fundus --name arcnet_fast --model arcnet --gpu_ids 0,1,2 --batch_size 16
```

**Monitoramento Profissional na nuvem via W&B:**
Ideal para gravar métricas temporais quando remoto. O Visdom funciona muito bem localmente.
```bash
python train.py --dataroot ./datasets/fundus --name train_gfenet --model gfenet \
    --use_wandb --wandb_project meu_tcc --wandb_run_name tentativa_final \
    --direction AtoB --dataset_mode cataract_with_mask
```

### Inferência e Controle Preciso

**Mudar a época exata da arquitetura de pesos (`--epoch`):**
Por vezes o `latest` sofre de overfit. Se o melhor check de loss ocorreu na época 50:
```bash
python test.py --dataroot datasets/my_gfenet_eval --name scrnet --model scrnet \
    --epoch 50 --num_test 5000 --eval
```

**Alterando Redimensionamento na Inferência sem Cortar o Fundus Completo:**
Evite recorte padrão de 256x256 durante o teste equalizando o scale com o crop no tamanho nativo.
```bash
python test.py --dataroot datasets/my_gfenet_eval --name gfenet --model gfenet \
    --load_size 768 --crop_size 768 --results_dir ./saidas_finais --preserve_subfolders
```
