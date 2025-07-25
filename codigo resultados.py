import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Caminho do banco de dados
DB_PATH ="database.db"
OUTPUT_DIR = "output2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def read_tables(db_path):
    conn = sqlite3.connect(db_path)
    df_res = pd.read_sql_query("SELECT * FROM pso_resultados", conn)
    df_exec = pd.read_sql_query("SELECT * FROM pso_execucao", conn)
    conn.close()
    return df_res, df_exec

def salvar_relatorio_txt(df_res, df_exec, output_path):
    cpu_medio = df_exec['uso_medio_cpu'].mean()
    memoria_media = df_exec['uso_max_memoria_mb'].mean()
    tempo_total_medio = df_exec['tempo_total_seg'].mean()
    fitness_medio = df_res['peso'].mean()
    melhor_fitness = df_res['peso'].max()
    media_geral = np.mean([cpu_medio, memoria_media, tempo_total_medio, fitness_medio, melhor_fitness])
    relatorio = (
        f"Uso médio de CPU: {cpu_medio:.2f}\n"
        f"Memória média (MB): {memoria_media:.2f}\n"
        f"Tempo total médio (segundos): {tempo_total_medio:.2f}\n"
        f"Fitness médio: {fitness_medio:.4f}\n"
        f"Melhor fitness: {melhor_fitness:.4f}\n"
        f"Média geral: {media_geral:.4f}\n"
    )
    with open(output_path, 'w') as f:
        f.write(relatorio)

def plot_entropy_by_generation(df_res, output_dir):
    pos_cols = ['pos_camada', 'pos_n1', 'pos_n2', 'pos_n3', 'pos_lr']
    entropy = df_res.groupby('num_iteracao')[pos_cols].var().sum(axis=1)
    plt.figure(figsize=(10,5))
    plt.plot(entropy, color='purple')
    plt.title('Evolução da Entropia por Geração')
    plt.xlabel('Geração')
    plt.ylabel('Entropia (bits)')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'entropia_por_geracao.png'))
    plt.close()

def plot_fitness_convergence_by_generation(df_res, output_dir):
    grouped = df_res.groupby('num_iteracao')
    best = grouped['peso'].max()
    mean = grouped['peso'].mean()
    worst = grouped['peso'].min()
    plt.figure(figsize=(10,5))
    plt.plot(best, label='Melhor Fitness', color='blue')
    plt.plot(mean, label='Fitness Médio', color='orange')
    plt.plot(worst, label='Pior Fitness', color='green')
    plt.title('Convergência do Fitness por Geração')
    plt.xlabel('Geração')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'fitness_por_geracao.png'))
    plt.close()

def plot_histograms_by_generation(df_res, output_dir):
    pos_cols = ['pos_camada', 'pos_n1', 'pos_n2', 'pos_n3', 'pos_lr']
    fig, axes = plt.subplots(1, 5, figsize=(20,4))
    for i, col in enumerate(pos_cols):
        axes[i].hist(df_res[col], bins=30, color='skyblue', edgecolor='black')
        mean_val = df_res[col].mean()
        axes[i].axvline(mean_val, color='red', linestyle='--', label=f'Média: {mean_val:.2f}')
        axes[i].set_title(f'Dimensão {i+1}')
        axes[i].set_xlabel('Valor')
        axes[i].set_ylabel('Frequência')
        axes[i].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'histogramas_geral.png'))
    plt.close()

def plot_pca_2d_by_generation(df_res, output_dir):
    pos_cols = ['pos_camada', 'pos_n1', 'pos_n2', 'pos_n3', 'pos_lr']
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(df_res[pos_cols])
    plt.figure(figsize=(8,6))
    plt.scatter(X_pca[:,0], X_pca[:,1], alpha=0.7)
    plt.title('PCA: Projeção 2D das 5 Dimensões (Todas as Gerações)')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% da variância)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% da variância)')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'pca_2d_geral.png'))
    plt.close()

def plot_pca_explained_variance(df_res, output_dir):
    pos_cols = ['pos_camada', 'pos_n1', 'pos_n2', 'pos_n3', 'pos_lr']
    pca = PCA(n_components=len(pos_cols))
    pca.fit(df_res[pos_cols])
    explained_variance = pca.explained_variance_ratio_
    plt.figure(figsize=(8,5))
    plt.bar(range(1, len(pos_cols)+1), explained_variance*100, color='teal')
    plt.xlabel('Componentes Principais')
    plt.ylabel('Variância Explicada (%)')
    plt.title('Variância Explicada por Componente Principal (PCA)')
    plt.xticks(range(1, len(pos_cols)+1))
    plt.grid(axis='y')
    plt.savefig(os.path.join(output_dir, 'pca_variancia_explicada.png'))
    plt.close()

# Execução principal
df_res, df_exec = read_tables(DB_PATH)
relatorio_path = os.path.join(OUTPUT_DIR, "resumo_estatisticas.txt")
salvar_relatorio_txt(df_res, df_exec, relatorio_path)

# Gráficos por geração
plot_entropy_by_generation(df_res, OUTPUT_DIR)
plot_fitness_convergence_by_generation(df_res, OUTPUT_DIR)
plot_histograms_by_generation(df_res, OUTPUT_DIR)
plot_pca_2d_by_generation(df_res, OUTPUT_DIR)
plot_pca_explained_variance(df_res, OUTPUT_DIR)
