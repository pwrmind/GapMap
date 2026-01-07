import torch
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
from sklearn.cluster import HDBSCAN

class BlueOceanFinder:
    def __init__(self, model_name="mxbai-embed-large", device=None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # В 2026 году эндпоинт /api/embed является стандартом для векторов
        self.ollama_url = "http://localhost:11434/api/embed"
        print(f"[*] Система запущена на устройстве: {self.device}")

    def _get_embeddings_batch(self, texts, batch_size=32):
        """Получение эмбеддингов через Ollama с поддержкой нового API."""
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Векторизация"):
            batch = texts[i:i+batch_size]
            for text in batch:
                try:
                    res = requests.post(self.ollama_url, 
                                     json={"model": self.model_name, "input": text}, 
                                     timeout=15).json()
                    
                    # Проверка ключей ответа (зависит от версии Ollama)
                    if 'embeddings' in res:
                        # Новый формат возвращает список списков
                        all_embeddings.append(res['embeddings'][0])
                    elif 'embedding' in res:
                        all_embeddings.append(res['embedding'])
                    else:
                        raise ValueError(f"Некорректный ответ API: {res}")
                except Exception as e:
                    print(f"\n[!] Ошибка на тексте '{text[:30]}': {e}")
                    all_embeddings.append([0.0] * 1024) # Заполнитель
                    
        return torch.tensor(all_embeddings, dtype=torch.float32).to(self.device)

    def find_gaps(self, queries_df, catalog_df, batch_size=500):
        """Поиск семантических разрывов между спросом и предложением."""
        print("[*] Шаг 1: Векторизация данных...")
        q_vecs = self._get_embeddings_batch(queries_df['query'].tolist())
        p_vecs = self._get_embeddings_batch(catalog_df['product_name'].tolist())

        print("[*] Шаг 2: Расчет дистанций на GPU...")
        min_distances = []
        
        # Батчинг для защиты от переполнения VRAM (видеопамяти)
        for i in range(0, len(q_vecs), batch_size):
            q_batch = q_vecs[i : i+batch_size]
            # Вычисляем попарные евклидовы расстояния
            dists = torch.cdist(q_batch, p_vecs, p=2) 
            min_d, _ = torch.min(dists, dim=1)
            min_distances.append(min_d.cpu())
            torch.cuda.empty_cache()

        gap_scores = torch.cat(min_distances).numpy()
        queries_df['gap_score'] = gap_scores
        
        # Индекс ниши = (разрыв в смысле) * (логарифм частоты спроса)
        queries_df['niche_index'] = queries_df['gap_score'] * np.log1p(queries_df['volume'])
        
        return queries_df, q_vecs

    def cluster_niches(self, df, vecs, threshold_percentile=50):
        """Группировка запросов-пустот в товарные категории."""
        # Фильтруем запросы, которые выше порога "необычности"
        threshold = np.percentile(df['niche_index'], threshold_percentile)
        n_mask = df['niche_index'] >= threshold
        
        n_samples = n_mask.sum()
        if n_samples < 2:
            print(f"[*] Слишком мало данных для кластеризации (найдено: {n_samples})")
            return df[n_mask]

        niche_vecs = vecs[n_mask.values].cpu().numpy()
        
        # Настройка HDBSCAN для работы с малыми и шумными выборками
        clusterer = HDBSCAN(min_cluster_size=2, metric='euclidean', copy=True)
        clusters = clusterer.fit_predict(niche_vecs)
        
        # Создаем копию для вывода, чтобы не ломать основной DF
        result_df = df[n_mask].copy()
        result_df['cluster'] = clusters
        return result_df

# --- Основной блок исполнения ---
if __name__ == "__main__":
    # Спрос: что люди ищут, но чего может не быть
    data_queries = {
        'query': [
            "термос для еды с подогревом от usb", 
            "складная беговая дорожка под кровать",
            "коврик для мышки с подогревом", 
            "чехол на айфон синий с блестками", 
            "куртка мужская осенняя мембранная"
        ],
        'volume': [4500, 12000, 800, 15000, 60000]
    }
    
    # Предложение: что уже есть на полках (база конкурентов)
    data_catalog = {
        'product_name': [
            "Обычный термос 1л стальной", 
            "Беговая дорожка магнитная Pro", 
            "Чехол на iPhone 15 прозрачный",
            "Куртка зимняя на меху", 
            "Коврик для мыши текстильный"
        ]
    }

    queries_df = pd.DataFrame(data_queries)
    catalog_df = pd.DataFrame(data_catalog)

    # Инициализация и запуск
    finder = BlueOceanFinder()
    
    # 1. Считаем разрывы
    scored_df, all_q_vecs = finder.find_gaps(queries_df, catalog_df)
    
    # 2. Выделяем ниши (для теста 5 строк ставим низкий порог 40%)
    niches_df = finder.cluster_niches(scored_df, all_q_vecs, threshold_percentile=40)

    # 3. Красивый вывод
    print("\n" + "="*50)
    print("АНАЛИЗ 'ГОЛУБЫХ ОКЕАНОВ' (TOP NICHES 2026)")
    print("="*50)

    if not niches_df.empty:
        # Сначала выводим не сгруппированные запросы, если кластеры не собрались
        if (niches_df['cluster'] == -1).all():
            print("\nУникальные перспективные запросы:")
            print(niches_df.sort_values(by='niche_index', ascending=False)[['query', 'niche_index']])
        else:
            # Группировка по кластерам
            for c_id in sorted(niches_df['cluster'].unique()):
                cluster_data = niches_df[niches_df['cluster'] == c_id]
                label = f"Ниша (Кластер #{c_id})" if c_id != -1 else "Разрозненные возможности"
                print(f"\n{label}:")
                for _, row in cluster_data.iterrows():
                    print(f" - {row['query']} (Индекс: {row['niche_index']:.2f})")
    else:
        print("Ниш не обнаружено. Попробуйте снизить порог или расширить данные.")
