import asyncio
import torch
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import List
from pydantic import BaseModel, HttpUrl, ConfigDict
from pydantic_settings import BaseSettings
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm.asyncio import tqdm
import httpx
from sklearn.cluster import HDBSCAN

# --- 1. Конфигурация ---

class AppConfig(BaseSettings):
    model_config = ConfigDict(env_prefix="GAP_", protected_namespaces=())
    
    ollama_url: HttpUrl = "http://localhost:11434/api/embed"
    model_name: str = "mxbai-embed-large"
    batch_size: int = 500
    concurrent_requests: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class NicheResult(BaseModel):
    query: str
    gap_score: float
    volume: int
    niche_index: float
    cluster: int = -1

# --- 2. Провайдер эмбеддингов ---

class EmbeddingProvider(ABC):
    @abstractmethod
    async def get_embeddings(self, texts: List[str]) -> torch.Tensor:
        pass

class OllamaProvider(EmbeddingProvider):
    def __init__(self, config: AppConfig):
        self.config = config
        self.semaphore = asyncio.Semaphore(config.concurrent_requests)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _fetch_one(self, client: httpx.AsyncClient, text: str) -> List[float]:
        async with self.semaphore:
            payload = {"model": self.config.model_name, "input": text}
            resp = await client.post(str(self.config.ollama_url), json=payload, timeout=20.0)
            resp.raise_for_status()
            data = resp.json()
            # Берем первый элемент, если API вернуло список списков
            emb = data.get("embeddings") or data.get("embedding")
            return emb[0] if isinstance(emb[0], list) else emb

    async def get_embeddings(self, texts: List[str]) -> torch.Tensor:
        async with httpx.AsyncClient() as client:
            tasks = [self._fetch_one(client, t) for t in texts]
            results = await tqdm.gather(*tasks, desc="Асинхронная векторизация")
            # Явно приводим к 2D: [кол-во текстов, размер вектора]
            return torch.tensor(results, dtype=torch.float32).to(self.config.device).view(len(texts), -1)

# --- 3. Аналитический движок ---

class BlueOceanEngine:
    def __init__(self, config: AppConfig, provider: EmbeddingProvider):
        self.config = config
        self.provider = provider

    def calculate_cosine_gap(self, q_vecs: torch.Tensor, p_vecs: torch.Tensor) -> torch.Tensor:
        q_norm = torch.nn.functional.normalize(q_vecs, p=2, dim=1)
        p_norm = torch.nn.functional.normalize(p_vecs, p=2, dim=1)
        
        gaps = []
        for i in range(0, len(q_norm), self.config.batch_size):
            q_batch = q_norm[i : i + self.config.batch_size]
            # Теперь p_norm гарантированно 2D, t() сработает корректно
            sims = torch.mm(q_batch, p_norm.t())
            max_sim, _ = torch.max(sims, dim=1)
            gaps.append(1 - max_sim)
            
        return torch.cat(gaps)

    async def analyze(self, queries: pd.DataFrame, catalog: pd.DataFrame):
        q_vecs = await self.provider.get_embeddings(queries['query'].tolist())
        p_vecs = await self.provider.get_embeddings(catalog['product_name'].tolist())

        print(f"[*] Анализ {len(queries)} запросов против {len(catalog)} товаров...")
        gap_scores = self.calculate_cosine_gap(q_vecs, p_vecs).cpu().numpy()
        
        queries['gap_score'] = gap_scores
        queries['niche_index'] = queries['gap_score'] * np.log1p(queries['volume'])
        
        return self._cluster(queries, q_vecs)

    def _cluster(self, df: pd.DataFrame, vecs: torch.Tensor):
        # Используем порог 50-й перцентили для теста, если данных мало
        threshold = df['niche_index'].quantile(0.5) if len(df) < 10 else (df['niche_index'].mean() + df['niche_index'].std())
        mask = df['niche_index'] >= threshold
        
        if mask.sum() < 2:
            return df[mask]

        target_vecs = vecs[mask.values].cpu().numpy()
        clusterer = HDBSCAN(min_cluster_size=2, metric='euclidean')
        
        res_df = df[mask].copy()
        res_df['cluster'] = clusterer.fit_predict(target_vecs)
        return res_df

# --- 4. Запуск ---

async def main():
    config = AppConfig()
    provider = OllamaProvider(config)
    engine = BlueOceanEngine(config, provider)

    queries_data = pd.DataFrame({
        'query': ["термос с usb подогревом", "беговая дорожка складная", "синий чехол", "умный ошейник с ИИ"],
        'volume': [1500, 3200, 50000, 800]
    })
    catalog_data = pd.DataFrame({
        'product_name': ["Обычный термос", "Чехол синий силикон", "Беговая дорожка стационарная"]
    })

    results = await engine.analyze(queries_data, catalog_data)

    print("\n" + "="*60)
    print(f"{'КЛАСТЕР':<12} | {'ЗАПРОС':<30} | {'INDEX':<7} | {'GAP'}")
    print("-" * 60)
    
    if not results.empty:
        for _, row in results.sort_values('niche_index', ascending=False).iterrows():
            item = NicheResult(**row.to_dict())
            c_tag = f"#{item.cluster}" if item.cluster != -1 else "UNIQ"
            print(f"{c_tag:<12} | {item.query[:30]:<30} | {item.niche_index:>7.2f} | {item.gap_score:.2f}")

if __name__ == "__main__":
    asyncio.run(main())
