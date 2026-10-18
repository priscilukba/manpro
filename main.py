import os
import re
import pickle
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["OMP_NUM_THREADS"] = "4" 

# =====================================================================
# 1. MEMBACA KAMUS RAKSASA DARI CSV
# =====================================================================
print("Memuat Kamus dari folder 'kamus'...")
try:
    kamus_method = pd.read_csv('kamus/kamus_method.csv')['method'].dropna().astype(str).tolist()
    kamus_problem = pd.read_csv('kamus/kamus_problem.csv')['problem'].dropna().astype(str).tolist()
    kamus_dataset = pd.read_csv('kamus/kamus_dataset.csv')['dataset'].dropna().astype(str).tolist()
except FileNotFoundError:
    print("ERROR: File kamus tidak ditemukan di folder 'kamus'!")
    exit()

# Ubah list menjadi format Regex yang aman (menggunakan re.escape)
regex_method = r"\b(" + "|".join(map(re.escape, kamus_method)) + r")\b"
regex_problem = r'([^.!?]*(' + "|".join(map(re.escape, kamus_problem)) + r')[^.!?]*[.!?])'
regex_dataset = r"\b(" + "|".join(map(re.escape, kamus_dataset)) + r")\b"

print(f"Total Metode: {len(kamus_method)}")
print(f"Total Kata Kunci Problem: {len(kamus_problem)}")
print(f"Total Sumber Dataset: {len(kamus_dataset)}")

# =====================================================================
# 2. FUNGSI EKSTRAKSI
# =====================================================================
def ekstrak_komponen_cepat(teks, max_char=7000):
    teks_pendek = str(teks)[:max_char]
    kalimat_list = [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", re.sub(r"\s+", " ", teks_pendek).strip()) if len(s.strip()) > 15]
    
    if not kalimat_list: return {"problem": "Tidak spesifik", "method": "Umum", "dataset": "Internal"}
    
    teks_gabung = " ".join(kalimat_list)
    
    # Ekstraksi menggunakan Regex dari CSV
    methods = list(set(re.findall(regex_method, teks_gabung, re.I)))
    datasets = list(set(re.findall(regex_dataset, teks_gabung, re.I)))
    
    # Problem: Cari kalimat pertama yang mengandung kata kunci masalah
    kalimat_prob = re.search(regex_problem, teks_gabung, re.I)
    prob_text = kalimat_prob.group(1).strip() if kalimat_prob else "Tidak spesifik"
    
    # Jika methods kosong, coba cari kata "metode" atau "algoritma"
    if not methods:
        if re.search(r'menggunakan\s+(metode|algoritma)\s+([A-Z][a-zA-Z0-9]+)', teks_gabung):
             methods = [re.search(r'menggunakan\s+(metode|algoritma)\s+([A-Z][a-zA-Z0-9]+)', teks_gabung).group(2)]
    
    return {
        "problem": prob_text,
        "method": ", ".join(methods) if methods else "Umum",
        "dataset": ", ".join(datasets) if datasets else "Data internal"
    }

# =====================================================================
# 3. CEK CACHE
# =====================================================================
CACHE_FILE = "cache_embeddings.pkl"

if os.path.exists(CACHE_FILE):
    print("\nMemuat data dan vektor dari cache...")
    with open(CACHE_FILE, "rb") as f:
        cache_data = pickle.load(f)
        df_all = cache_data["df_all"]
        embeddings_teknis_db = cache_data["emb_teknis"]
        embeddings_problem_db = cache_data["emb_problem"]
    print(f"Berhasil memuat {len(df_all)} data dari cache.")
else:
    print("\nCache belum ada. Melakukan ekstraksi dari awal...")
    
    print("Memuat Model SBERT...")
    sbert = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    
    try:
        df_sib = pd.read_excel('data/SIB.xlsx', sheet_name='Skripsi_Batch2')
        df_infor = pd.read_excel('data/INFOR.xlsx', sheet_name='Skripsi_Info')
        df_dsa = pd.read_excel('data/DSA.xlsx', sheet_name='Skripsi')
    except FileNotFoundError:
        print("ERROR: File Excel tidak ditemukan!")
        exit()

    df_sib['Jurusan'] = 'SIB'
    df_infor['Jurusan'] = 'INFOR'
    df_dsa['Jurusan'] = 'DSA'

    kolom_penting = ['Judul', 'Chapter1', 'Tahun', 'Jurusan']
    df_all = pd.concat([df_sib[kolom_penting], df_infor[kolom_penting], df_dsa[kolom_penting]], ignore_index=True)
    df_all = df_all.dropna(subset=['Judul', 'Chapter1'])

    print(f"Mengekstrak Problem, Method, Dataset dari {len(df_all)} skripsi...")
    results = [ekstrak_komponen_cepat(teks) for teks in tqdm(df_all['Chapter1'], desc="Ekstraksi")]
    
    df_all['Problem'] = [r['problem'] for r in results]
    df_all['Method'] = [r['method'] for r in results]
    df_all['Dataset'] = [r['dataset'] for r in results]

    df_all[['Judul', 'Jurusan', 'Tahun', 'Method', 'Dataset']].to_csv('hasil_labeling.csv', index=False)

    print("\nMembuat Vektor Pencarian...")
    df_all['Teks_Teknis'] = df_all['Method'] + " " + df_all['Dataset']
    embeddings_teknis_db = sbert.encode(df_all['Teks_Teknis'].tolist(), convert_to_tensor=True)
    
    df_all['Teks_Problem'] = df_all['Problem']
    embeddings_problem_db = sbert.encode(df_all['Teks_Problem'].tolist(), convert_to_tensor=True)

    print("Menyimpan cache...")
    with open(CACHE_FILE, "wb") as f:
        pickle.dump({
            "df_all": df_all,
            "emb_teknis": embeddings_teknis_db,
            "emb_problem": embeddings_problem_db
        }, f)
    print("Cache berhasil disimpan!")

# =====================================================================
# 4. SISTEM PENCARIAN INTERAKTIF
# =====================================================================
def analisis_ide_skripsi():
    # Load model jika hanya membaca dari cache
    global sbert
    if 'sbert' not in globals():
        print("\nMemuat Model SBERT untuk Pencarian...")
        sbert = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    print("\n" + "=" * 60)
    print("SISTEM VALIDASI & ANALISIS IDE SKRIPSI")
    print("Ketik 'exit' pada judul untuk keluar.")
    print("=" * 60)
    
    while True:
        judul_user = input("\nMasukkan Judul Skripsi Anda:\n>> ")
        if judul_user.lower() == 'exit':
            print("Terima kasih!")
            break
            
        deskripsi_user = input("\nMasukkan Latar Belakang / Deskripsi Singkat:\n>> ")
        
        print("\n[AI Sedang Menganalisis Ide Anda...]")
        ide_user = ekstrak_komponen_cepat(judul_user + ". " + deskripsi_user)
        
        print(f"\nEkstraksi Ide Anda:")
        print(f"- Problem : {ide_user['problem'][:100]}...")
        print(f"- Method  : {ide_user['method']}")
        print(f"- Dataset : {ide_user['dataset']}")
        
        teks_teknis_user = f"{ide_user['method']} {ide_user['dataset']}"
        emb_teknis_user = sbert.encode(teks_teknis_user, convert_to_tensor=True)
        emb_problem_user = sbert.encode(ide_user['problem'], convert_to_tensor=True)
        
        scores_teknis = util.cos_sim(emb_teknis_user, embeddings_teknis_db)[0]
        scores_problem = util.cos_sim(emb_problem_user, embeddings_problem_db)[0]
        
        final_scores = (scores_teknis * 0.7) + (scores_problem * 0.3)
        top_results = torch.topk(final_scores, k=5)
        
        print("\n" + "-" * 60)
        print("HASIL PENCARIAN DATABASE:")
        
        found_match = False
        for score, idx in zip(top_results[0], top_results[1]):
            skor = score.item()
            
            if skor > 0.40:
                found_match = True
                row = df_all.iloc[idx.item()]
                
                print(f"\n[Skor Kemiripan: {skor:.4f}]")
                print(f"Jurusan/Tahun  : {row['Jurusan']} ({row['Tahun']})")
                print(f"Judul Skripsi  : {row['Judul']}")
                print(f"Method-nya     : {row['Method']}")
                print(f"Dataset-nya    : {row['Dataset']}")
                
                # Pengecekan kesamaan metode dan dataset
                user_methods = [m.strip().lower() for m in ide_user['method'].split(',')]
                db_methods = [m.strip().lower() for m in str(row['Method']).split(',')]
                method_sama = any(um in db_methods for um in user_methods if um != "umum")
                
                user_datasets = [d.strip().lower() for d in ide_user['dataset'].split(',')]
                db_datasets = [d.strip().lower() for d in str(row['Dataset']).split(',')]
                dataset_sama = any(ud in db_datasets for ud in user_datasets if ud != "data internal")
                
                if method_sama and dataset_sama:
                    print(">> BAHAYA: Metode dan Dataset Anda SANGAT MIRIP dengan skripsi ini.")
                elif method_sama:
                    print(">> PERHATIAN: Metode Anda SAMA, tapi Dataset berbeda. (Bisa jadi Research Gap).")
                elif dataset_sama:
                    print(">> PERHATIAN: Dataset Anda SAMA, tapi Metode berbeda. (Bisa jadi Research Gap).")
                else:
                    print(">> AMAN: Topik mungkin mirip, tapi kombinasi Metode & Dataset Anda berbeda.")
        
        if not found_match:
            print("\nTidak ada skripsi di database yang memiliki skor kemiripan di atas 0.40.")
            print(">> STATUS: RESEARCH GAP DITEMUKAN! Ide Anda sangat orisinil.")
            
        print("-" * 60)

if __name__ == "__main__":
    analisis_ide_skripsi()