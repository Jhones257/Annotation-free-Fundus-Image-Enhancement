# Kaggle Tutorial: Executando GFENet com Seu Código + Dataset Empacotados

Este tutorial parte do cenário em que você **já adaptou o código localmente** (ex.: `prepare_gfenet_inference.py`, flag `--preserve_subfolders`) e deseja levar **exatamente o mesmo diretório** — código + pesos + dataset preparado — para o Kaggle como um único Dataset. Assim você evita divergências entre versões do GitHub e o ambiente realmente utilizado nos experimentos.

---

## 1. Preparar o diretório local

1. No repositório clonado, gere o dataroot da GFENet com máscaras:
   ```
   python scripts/prepare_gfenet_inference.py --input_dir ORIGINAL_CLEAN_768x768 --output_dir datasets/my_gfenet_eval
   ```
2. Crie `checkpoints/gfenet/` (se ainda não existir) e coloque o `latest_net_G.pth` oficial dentro dela.
3. Teste um `python test.py ...` localmente para garantir que todas as dependências e caminhos funcionam.

> Resultado esperado: sua pasta `Annotation-free-Fundus-Image-Enhancement/` agora contém **todo o código personalizado** + `datasets/my_gfenet_eval` + `checkpoints/gfenet/latest_net_G.pth`.

---

## 2. Compactar tudo para enviar ao Kaggle

1. Opcional: remova pastas grandes desnecessárias (`results/`, `__pycache__`, etc.) para reduzir o tamanho.
2. No PowerShell, estando um nível acima da pasta do projeto:
   ```powershell
   Compress-Archive -Path .\Annotation-free-Fundus-Image-Enhancement -DestinationPath .\AFE_full_package.zip
   ```
   (Ajuste o nome conforme preferir.)

Esse arquivo `.zip` conterá o repositório completo, incluindo dataset preparado e pesos.

---

## 3. Criar um Dataset no Kaggle

1. Acesse [kaggle.com/datasets](https://www.kaggle.com/datasets) → **Create Dataset**.
2. Faça upload do `AFE_full_package.zip` criado acima.
3. Defina nome/descrição claros (ex.: "Annotation-free Fundus Enhancement - Full Package").
4. Publique e anote o slug resultante (ex.: `seuusuario/afe-full-package`).

Agora qualquer Notebook poderá montar exatamente a mesma árvore de arquivos que você tem localmente.

---

## 4. Notebook Kaggle: configuração base

1. Crie um Notebook com GPU (T4 ou V100). Ative **Internet** somente se precisar baixar algo extra.
2. Adicione o Dataset do passo anterior em **Add data → Your Datasets**.
3. No início do Notebook, descompacte e instale as dependências:

   ```
   !mkdir -p /kaggle/working/project
   !unzip -q /kaggle/input/afe-full-package/AFE_full_package.zip -d /kaggle/working/project
   %cd /kaggle/working/project/Annotation-free-Fundus-Image-Enhancement
   !pip install -r requirements.txt
   ```

4. (Opcional) Reexecute `scripts/prepare_gfenet_inference.py` caso tenha inserido novas imagens após gerar o ZIP.

> Todas as modificações feitas localmente (scripts, flags novas, etc.) já estarão presentes, pois o projeto foi inteiro enviado.

---

## 5. Rodar a GFENet dentro do Notebook

Certifique-se de que `datasets/my_gfenet_eval` e `checkpoints/gfenet/latest_net_G.pth` existem (devem ter vindo no ZIP). Então execute:

```
!python test.py \
    --model gfenet \
    --dataset_mode cataract_with_mask \
    --name gfenet \
    --dataroot datasets/my_gfenet_eval \
    --results_dir /kaggle/working/results \
    --load_size 768 --crop_size 768 \
    --num_test 100000 \
    --preserve_subfolders --eval
```

Principais flags:
- `--preserve_subfolders`: usa sua alteração para manter a hierarquia original das imagens.
- `--eval`: força BatchNorm/Dropout em modo inferência.
- `--num_test`: deixe grande para processar tudo.

O HTML com amostras ficará em `/kaggle/working/results/gfenet/test_latest/index.html`.

---

## 6. Exportar resultados mantendo a estrutura

As saídas ficam em `/kaggle/working/results/gfenet/test_latest/images/<mesmos subdirs>`. Se quiser fazer download:

```
!cd /kaggle/working && zip -r GFENet_outputs.zip results/gfenet/test_latest/images
```

O arquivo aparecerá na aba **Output** e poderá ser baixado.

---

## 7. Dicas e troubleshooting

| Situação | Como resolver |
| --- | --- |
| `ModuleNotFoundError: util` ao rodar scripts auxiliares | Já foi tratado em `prepare_gfenet_inference.py` (adiciona o root ao `sys.path`). Certifique-se de usar a versão atualizada incluída no ZIP. |
| `--preserve_subfolders` não reconhecido | O Notebook não está usando o pacote atualizado. Confirme se `test.py` e `options/base_options.py` do ZIP contém a alteração. |
| Falta de GPU/tempo | Reduza `--load_size` ou processe em lotes menores de imagens. |
| Peso não encontrado | Verifique se `checkpoints/gfenet/latest_net_G.pth` veio no ZIP ou foi adicionado manualmente ao Notebook. |

---

Seguindo esses passos você garante que o Kaggle use **exatamente o mesmo código e dados** que você preparou localmente, evitando discrepâncias entre repositórios e simplificando a reprodução dos resultados da GFENet. 
