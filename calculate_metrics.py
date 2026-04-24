import os
import glob
import cv2
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    pass

try:
    from skimage.metrics import peak_signal_noise_ratio as psnr_metric
    from skimage.metrics import structural_similarity as ssim_metric
except ImportError:
    print("Instale o scikit-image: pip install scikit-image")

try:
    import torch
    import pyiqa
except ImportError:
    print("Instale o pyiqa e torch para calcular BRISQUE e NIQE: pip install pyiqa torch torchvision")

def get_image_paths(original_dir, enhanced_dir):
    # Buscar arquivos de forma recursiva para incluir as subpastas
    enhanced_paths = glob.glob(os.path.join(enhanced_dir, '**', '*.*'), recursive=True)
    enhanced_paths = [p for p in enhanced_paths if os.path.isfile(p)]
    
    image_pairs = []
    
    for enh_path in enhanced_paths:
        filename = os.path.basename(enh_path)
        # Obter o caminho da subpasta relativo ao diretório base
        rel_path = os.path.relpath(enh_path, enhanced_dir)
        
        # O caminho original provável mantém a mesma estrutura de subpasta
        base_rel_path = rel_path.replace('_fake_B', '').replace('_filtered', '')
        orig_path = os.path.join(original_dir, base_rel_path)
        
        # Fallback: se não achar na mesma subpasta, busca pelo nome em todo o diretório original
        if not os.path.exists(orig_path):
            base_name = filename.replace('_fake_B', '').replace('_filtered', '')
            orig_search = glob.glob(os.path.join(original_dir, '**', base_name), recursive=True)
            orig_path = orig_search[0] if orig_search else None
        
        # Usa o rel_path (caminho com a subpasta) no csv para melhor identificação nos resultados
        image_pairs.append({'enhanced': enh_path, 'original': orig_path, 'filename': rel_path})
        
    return image_pairs

def calc_single_matrix_metrics(label):
    print(f"\nPor favor, insira os valores da matriz de confusão da MCFE-net ({label}).")
    try:
        boas_boas = int(input("Imagens Boas classificadas como Boas: "))
        usaveis_usaveis = int(input("Imagens Usáveis classificadas como Usáveis: "))
        rejeitadas_rejeitadas = int(input("Imagens Rejeitadas classificadas como Rejeitadas: "))
        
        boas_usaveis = int(input("Imagens Boas classificadas como Usáveis: "))
        boas_rejeitadas = int(input("Imagens Boas classificadas como Rejeitadas: "))
        usaveis_boas = int(input("Imagens Usáveis classificadas como Boas: "))
        usaveis_rejeitadas = int(input("Imagens Usáveis classificadas como Rejeitadas: "))
        rejeitadas_boas = int(input("Imagens Rejeitadas classificadas como Boas: "))
        rejeitadas_usaveis = int(input("Imagens Rejeitadas classificadas como Usáveis: "))
        
        total = boas_boas + usaveis_usaveis + rejeitadas_rejeitadas + boas_usaveis + boas_rejeitadas + usaveis_boas + usaveis_rejeitadas + rejeitadas_boas + rejeitadas_usaveis
        accuracy = (boas_boas + usaveis_usaveis + rejeitadas_rejeitadas) / total if total > 0 else 0
        
        tp1, fp1, fn1 = boas_boas, usaveis_boas + rejeitadas_boas, boas_usaveis + boas_rejeitadas
        p1 = tp1 / (tp1 + fp1) if (tp1 + fp1) > 0 else 0
        r1 = tp1 / (tp1 + fn1) if (tp1 + fn1) > 0 else 0
        f1_1 = 2 * p1 * r1 / (p1 + r1) if (p1 + r1) > 0 else 0
        
        tp2, fp2, fn2 = usaveis_usaveis, boas_usaveis + rejeitadas_usaveis, usaveis_boas + usaveis_rejeitadas
        p2 = tp2 / (tp2 + fp2) if (tp2 + fp2) > 0 else 0
        r2 = tp2 / (tp2 + fn2) if (tp2 + fn2) > 0 else 0
        f1_2 = 2 * p2 * r2 / (p2 + r2) if (p2 + r2) > 0 else 0
        
        tp3, fp3, fn3 = rejeitadas_rejeitadas, boas_rejeitadas + usaveis_rejeitadas, rejeitadas_boas + rejeitadas_usaveis
        p3 = tp3 / (tp3 + fp3) if (tp3 + fp3) > 0 else 0
        r3 = tp3 / (tp3 + fn3) if (tp3 + fn3) > 0 else 0
        f1_3 = 2 * p3 * r3 / (p3 + r3) if (p3 + r3) > 0 else 0
        
        macro_f1 = (f1_1 + f1_2 + f1_3) / 3
        
        matrix_array = np.array([
            [boas_boas, boas_usaveis, boas_rejeitadas],
            [usaveis_boas, usaveis_usaveis, usaveis_rejeitadas],
            [rejeitadas_boas, rejeitadas_usaveis, rejeitadas_rejeitadas]
        ])
        
        print(f"\nResultados calculados ({label}):")
        print(f"Acurácia (FIQA aprox.): {accuracy:.4f}")
        print(f"F1-Score (WFIQA aprox.): {macro_f1:.4f}")
        
        return {
            'Accuracy': accuracy, 
            'Total': total, 
            'F1_Score': macro_f1,
            'TP_Total': tp1 + tp2 + tp3,
            'FP_Total': fp1 + fp2 + fp3,
            'matrix': matrix_array
        }
    except ValueError:
        print("Valores inválidos inseridos.")
        return None

def calculate_fiqa_wfiqa_from_matrix():
    print("\n--- Módulo FIQA / WFIQA ---")
    print("Você escolheu calcular métricas baseadas na avaliação da MCFE-net.")
    
    res_before = calc_single_matrix_metrics("Antes do Melhoramento")
    res_after = calc_single_matrix_metrics("Depois do Melhoramento")
    
    if res_before and res_after:
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            classes = ['Boas', 'Usáveis', 'Rejeitadas']
            
            sns.heatmap(res_before['matrix'], annot=True, fmt='d', cmap='Oranges', xticklabels=classes, yticklabels=classes, ax=axes[0])
            axes[0].set_title('Antes do Melhoramento')
            axes[0].set_xlabel('Previsto')
            axes[0].set_ylabel('Real')
            
            sns.heatmap(res_after['matrix'], annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=axes[1])
            axes[1].set_title('Depois do Melhoramento')
            axes[1].set_xlabel('Previsto')
            axes[1].set_ylabel('Real')
            
            plt.tight_layout()
            plt.savefig('confusion_matrices_comparison.png')
            print("\n[Sucesso] Gráfico comparativo das matrizes de confusão salvo em 'confusion_matrices_comparison.png'")
        except ImportError:
            print("\n[Aviso] Bibliotecas matplotlib ou seaborn não instaladas. Não foi possível plotar as matrizes de confusão.")
            
        return {'before': res_before, 'after': res_after}
    
    return None

def main():
    print("=== Avaliação de Métricas de Qualidade de Imagem de Fundo de Olho ===")
    print("Selecione as métricas que deseja calcular separadas por vírgula (ex: 1,2,4):")
    print("1 - PSNR (Requer imagens originais)")
    print("2 - SSIM (Requer imagens originais)")
    print("3 - FIQA / WFIQA (Inserção manual de Matriz de Confusão MCFE-net)")
    print("4 - BRISQUE (No-Reference)")
    print("5 - NIQE (No-Reference)")
    print("6 - TODAS AS MÉTRICAS")
    
    choices = input("Sua escolha: ").split(',')
    choices = [c.strip() for c in choices]
    
    calc_psnr = '1' in choices or '6' in choices
    calc_ssim = '2' in choices or '6' in choices
    calc_fiqa = '3' in choices or '6' in choices
    calc_brisque = '4' in choices or '6' in choices
    calc_niqe = '5' in choices or '6' in choices

    original_dir = "RETINOGRAFO_EYER_PROCESSED"
    enhanced_dir = "results/gfenet_gray_1ch_200e/test_latest/images/filtered"
    
    if not os.path.exists(enhanced_dir):
        print(f"Erro: O diretório de imagens melhoradas não foi encontrado: {enhanced_dir}")
        return

    # Inicializa os modelos PyIQA se necessário
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    brisque_metric, niqe_metric = None, None
    if calc_brisque:
        print("Carregando modelo BRISQUE...")
        brisque_metric = pyiqa.create_metric('brisque', device=device)
    if calc_niqe:
        print("Carregando modelo NIQE...")
        niqe_metric = pyiqa.create_metric('niqe', device=device)

    # Processamento Manual de FIQA/WFIQA
    fiqa_results = None
    if calc_fiqa:
        fiqa_results = calculate_fiqa_wfiqa_from_matrix()

    print("\nMapeando imagens...")
    pairs = get_image_paths(original_dir, enhanced_dir)
    print(f"Encontradas {len(pairs)} imagens melhoradas para avaliar.")

    results = []
    
    for pair in tqdm(pairs, desc="Calculando métricas das imagens"):
        img_res = {'Filename': pair['filename']}
        
        # Carrega a imagem melhorada
        enh_img = cv2.imread(pair['enhanced'])
        if enh_img is None:
            continue
            
        # Métricas Full-Reference (PSNR, SSIM)
        if calc_psnr or calc_ssim:
            if pair['original'] and os.path.exists(pair['original']):
                orig_img = cv2.imread(pair['original'])
                
                # Garantir mesmo tamanho para comparação justa
                if orig_img.shape != enh_img.shape:
                    orig_img = cv2.resize(orig_img, (enh_img.shape[1], enh_img.shape[0]))
                
                if calc_psnr:
                    img_res['PSNR'] = psnr_metric(orig_img, enh_img)
                if calc_ssim:
                    # win_size deve ser ajustado, channel_axis informa que é imagem colorida
                    img_res['SSIM'] = ssim_metric(orig_img, enh_img, channel_axis=2, data_range=255)
            else:
                if calc_psnr: img_res['PSNR'] = None
                if calc_s_Before'] = fiqa_results['before']['Accuracy']
        df['WFIQA_F1_Before'] = fiqa_results['before']['F1_Score']
        df['FIQA_Accuracy_After'] = fiqa_results['after']['Accuracy']
        df['WFIQA_F1_After'] = fiqa_results['after']
        # Métricas No-Reference (BRISQUE, NIQE) com pyiqa
        if calc_brisque or calc_niqe:
            # PyIQA recebe o caminho do arquivo
            try:
                if calc_brisque:
                    score_b = brisque_metric(pair['enhanced']).item()
                    img_res['BRISQUE'] = score_b
                if calc_niqe:
                    score_n = niqe_metric(pair['enhanced']).item()
                    img_res['NIQE'] = score_n
            except Exception as e:
                if calc_brisque: img_res['BRISQUE'] = None
                if calc_niqe: img_res['NIQE'] = None

        results.append(img_res)

    # Criação do DataFrame e Salvamento
    df = pd.DataFrame(results)
    
    # Adicionar as métricas globais de FIQA se computadas
    if fiqa_results:
        df['FIQA_Accuracy'] = fiqa_results['Accuracy']
        df['WFIQA_F1'] = fiqa_results['F1_Score']

    csv_output = "metricas_qualidade_resultados.csv"
    df.to_csv(csv_output, index=False)
    print(f"\nResultados detalhados salvos em '{csv_output}'.")

    # Calcula estatísticas e gera arquivo TXT de análise
    txt_output = "analise_metricas_resumo.txt"
    with open(txt_output, 'w') as f:
        f.write("=== ANÁLISE RESUMIDA DAS MÉTRICAS DE QUALIDADE ===\n\n")
        f.write(f"Total de imagens avaliadas: {len(df)}\n\n")_Before', 'WFIQA_F1_Before', 'FIQA_Accuracy_After', 'WFIQA_F1_After
        
        metrics_cols = [col for col in df.columns if col not in ['Filename', 'FIQA_Accuracy', 'WFIQA_F1']]
        
        for col in metrics_cols:
            if df[col].notnull().sum() > 0:
                mean_val = df[col].mean()
                std_val = df[col].std()
                f.write(f"Métrica: {col}\n")
                f.write(f"  - Média: {mean_val:.4f}\n")
                f.write(f"  - Desvio Padrão: {std_val:.4f}\n\n")
                
                # Breve análise
                if col == 'PSNR':
                    f.write("  * Análise: Valores mais altos de PSNR indicam menor distorção e maior fidelidade à imagem original.\n\n")
                elif col == 'SSIM':
                    f.write("  * Análise: O SSIM varia de -1 a 1 (ou 0 a 1). Valores mais próximos a 1 indicam alta similaridade estrutural com a original.\n\n")
                elif col == 'BRISQUE':
                    f.write("  * Análise BRISQUE: É uma métrica sem referência. Valores MENORES indicam melhor qualidade perceptual e menos artefatos/ruído.\n\n")
                elif col == 'NIQE':
                    f.write("  * Análise NIQE: Também sem referência. Valores MENORES indicam uma imagem mais natural e com melhor qualidade.\n\n")

        if fiqa_results:
            f.write("Métricas FIQA / WFIQA (Baseadas na Rede MCFE-net)\n")
            f.write("  [Antes do Melhoramento]\n")
            f.write(f"    - Total de Acertos (TP Global): {fiqa_results['before']['TP_Total']}, Erros (FP/FN Globais): {fiqa_results['before']['FP_Total']}\n")
            f.write(f"    - FIQA (Acurácia Global): {fiqa_results['before']['Accuracy']:.4f}\n")
            f.write(f"    - WFIQA (Macro F1-Score): {fiqa_results['before']['F1_Score']:.4f}\n\n")
            f.write("  [Depois do Melhoramento]\n")
            f.write(f"    - Total de Acertos (TP Global): {fiqa_results['after']['TP_Total']}, Erros (FP/FN Globais): {fiqa_results['after']['FP_Total']}\n")
            f.write(f"    - FIQA (Acurácia Global): {fiqa_results['after']['Accuracy']:.4f}\n")
            f.write(f"    - WFIQA (Macro F1-Score): {fiqa_results['after']['F1_Score']:.4f}\n\n")
            
            diff_acc = fiqa_results['after']['Accuracy'] - fiqa_results['before']['Accuracy']
            diff_f1 = fiqa_results['after']['F1_Score'] - fiqa_results['before']['F1_Score']
            f.write(f"  * Ganho de Qualidade: Acurácia ({diff_acc:+.4f}) | F1-Score ({diff_f1:+.4f})\n")
            f.write("  * Análise: O aumento da acurácia e F1-Score do MCFE-net das imagens originais para as melhoradas indica o sucesso da restauração da imagem de fundo de olho para diagnósticos clínicos.\n\n")

    print(f"Resumo da análise salvo em '{txt_output}'.")

if __name__ == "__main__":
    main()
