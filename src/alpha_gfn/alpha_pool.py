from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from alphagen.models.alpha_pool import AlphaPool, AlphaPoolBase
from alphagen.data.expression import Expression
from alphagen_qlib.stock_data import StockData


class AlphaPoolGFN(AlphaPool):
    def __init__(
        self,
        capacity: int,
        stock_data: StockData,
        target: Expression,
        ic_mut_threshold: float = 0.3,
        ssl_k: int = 3,
        ssl_tau: float = 0.1
    ):
        super().__init__(capacity, stock_data, target)
        self.ic_mut_threshold = ic_mut_threshold
        self.ssl_k = ssl_k  # K for k-nearest neighbors
        self.ssl_tau = ssl_tau  # Temperature parameter for similarity weight
        # Initialize embeddings storage with the same structure as other factor properties
        self.embeddings: List[Optional[Tensor]] = [None for _ in range(capacity + 1)]

    def try_new_expr(self, expr: Expression, embedding: Optional[Tensor] = None) -> Tuple[float, float]:
        value = self._normalize_by_day(expr.evaluate(self.data))
        ic_ret, ic_mut = self._calc_ics(value, ic_mut_threshold=0.99)
        if ic_ret is None or ic_mut is None:
            return 0.0, 1.0
        ic_ret = np.abs(ic_ret)
        ic_mut = np.abs(ic_mut)
        
        # Check if we should add this factor to the pool
        if self.size < self.capacity:
            # Pool not full, add directly if correlation constraint is satisfied
            if ic_mut.size == 0 or np.max(ic_mut) <= self.ic_mut_threshold:
                self._add_factor(expr, value, ic_ret, ic_mut, embedding)
                print(f"[Pool Add] {expr}")
        else:
            # Pool is full, check if this factor is better than the worst one
            min_ic_idx = np.argmin(self.single_ics[:self.size])
            min_ic = self.single_ics[min_ic_idx]
            
            if ic_ret > min_ic and (ic_mut.size == 0 or np.max(ic_mut) <= self.ic_mut_threshold):
                # Add the new factor first (this will make size = capacity + 1)
                self._add_factor(expr, value, ic_ret, ic_mut, embedding)
                print(f"[Pool Add] {expr}")
                # Then remove the worst factor using _pop
                print(f"[Pool Pop] {self.exprs[np.argmin(self.single_ics[:self.size])]}")
                self._pop()
            else:
                print(f"[Pool Reject] {expr}")
        
        return ic_ret, (1 - np.max(ic_mut)) if ic_mut.size > 0 else 1.0
    
    def _add_factor(
        self,
        expr: Expression,
        value: Tensor,
        ic_ret: float,
        ic_mut: List[float],
        embedding: Optional[Tensor] = None
    ):
        # Call parent method to handle standard factor storage
        super()._add_factor(expr, value, ic_ret, ic_mut)
        # Store the embedding for the newly added factor
        n = self.size - 1  # size was incremented in parent method
        self.embeddings[n] = embedding
    

    def _pop(self) -> None:
        # Pop the factor with the lowest ic
        if self.size <= self.capacity:
            return
        idx = np.argmin(self.single_ics[:self.size])
        self._swap_idx(idx, self.capacity)
        self.size = self.capacity
    
    def _swap_idx(self, i, j) -> None:
        if i == j:
            return
        # Call parent method to handle standard factor swapping
        super()._swap_idx(i, j)
        # Swap embeddings
        self.embeddings[i], self.embeddings[j] = self.embeddings[j], self.embeddings[i]
    
    def _find_k_nearest_neighbors(self, query_embedding: Tensor, k: int, exclude_self: bool = True, distance_threshold: float = 1e-6) -> List[int]:
        """
        Find k nearest neighbors based on embedding similarity
        
        Args:
            query_embedding: The embedding to find neighbors for
            k: Number of neighbors to find
            exclude_self: Whether to exclude identical embeddings (distance ≈ 0)
            distance_threshold: Minimum distance to consider as different embeddings
            
        Returns:
            List of indices of k nearest neighbors
        """
        if self.size <= 1:
            # print(f"[SSL Debug] Pool size <= 1 ({self.size}), no neighbors available")
            return []
        
        distances = []
        valid_indices = []
        
        for i in range(self.size):
            if self.embeddings[i] is not None:
                # Calculate L2 distance
                dist = torch.norm(query_embedding - self.embeddings[i]).item()
                
                # Skip if this is likely the same embedding (distance ≈ 0)
                if exclude_self and dist < distance_threshold:
                    # print(f"[SSL Debug] Factor {i}: distance = {dist:.6f} (SKIPPED - too similar/identical)")
                    continue
                    
                distances.append(dist)
                valid_indices.append(i)
                # print(f"[SSL Debug] Factor {i}: distance = {dist:.6f}")
        
        if len(distances) == 0:
            # print(f"[SSL Debug] No valid embeddings found in pool (after excluding self)")
            return []
        
        # Get k smallest distances (nearest neighbors)
        k = min(k, len(distances))
        _, indices = torch.topk(torch.tensor(distances), k, largest=False)
        
        neighbor_indices = [valid_indices[idx] for idx in indices.tolist()]
        neighbor_distances = [distances[idx] for idx in indices.tolist()]
        
        # print(f"[SSL Debug] Found {len(neighbor_indices)} neighbors: {neighbor_indices}")
        # print(f"[SSL Debug] Neighbor distances: {[f'{d:.6f}' for d in neighbor_distances]}")
        
        return neighbor_indices
    
    def _compute_similarity_weights(self, query_embedding: Tensor, neighbor_indices: List[int]) -> Tensor:
        """
        Compute similarity weights using softmax with temperature
        
        Args:
            query_embedding: The query embedding
            neighbor_indices: Indices of neighbor factors
            
        Returns:
            Normalized similarity weights
        """
        if not neighbor_indices:
            # print(f"[SSL Debug] No neighbor indices provided")
            return torch.tensor([])
        
        # Calculate squared L2 distances
        distances = []
        similarity_scores = []
        for idx in neighbor_indices:
            if self.embeddings[idx] is not None:
                dist_squared = torch.norm(query_embedding - self.embeddings[idx]) ** 2
                similarity_score = -dist_squared / self.ssl_tau  # Negative for similarity
                distances.append(dist_squared.item())
                similarity_scores.append(similarity_score)
                # print(f"[SSL Debug] Factor {idx}: dist²={dist_squared.item():.4f}, similarity_score={similarity_score.item():.4f}")
        
        if not similarity_scores:
            # print(f"[SSL Debug] No valid similarity scores computed")
            return torch.tensor([])
        
        # Apply softmax to get normalized weights
        weights = F.softmax(torch.tensor(similarity_scores), dim=0)
        
        # print(f"[SSL Debug] Similarity weights: {[f'{w.item():.4f}' for w in weights]}")
        # print(f"[SSL Debug] Weight sum: {weights.sum().item():.4f}")
        
        return weights
    
    def _compute_consistency_loss(self, query_value: Tensor, neighbor_indices: List[int], weights: Tensor) -> float:
        """
        Compute consistency loss between query factor and its neighbors
        
        Args:
            query_value: Normalized factor values for the query factor
            neighbor_indices: Indices of neighbor factors
            weights: Similarity weights
            
        Returns:
            Consistency loss value
        """
        if not neighbor_indices or len(weights) == 0:
            # print(f"[SSL Debug] No neighbors or weights for consistency loss")
            return 0.0
        
        total_loss = 0.0
        individual_losses = []
        
        # print(f"[SSL Debug] Computing consistency loss with {len(neighbor_indices)} neighbors")
        # print(f"[SSL Debug] Query value shape: {query_value.shape}")
        
        for i, idx in enumerate(neighbor_indices):
            if idx < len(self.values) and self.values[idx] is not None:
                neighbor_value = self.values[idx]
                
                # Calculate MSE loss per cross-section and average
                diff_squared = (query_value - neighbor_value) ** 2
                mse_per_section = diff_squared.mean(dim=1)  # Average across stocks in each day
                avg_mse = mse_per_section.mean().item()  # Average across days
                
                weighted_loss = weights[i].item() * avg_mse
                total_loss += weighted_loss
                individual_losses.append(avg_mse)
                
                # print(f"[SSL Debug] Neighbor {idx}: MSE={avg_mse:.6f}, weight={weights[i].item():.4f}, weighted_loss={weighted_loss:.6f}")
        
        # print(f"[SSL Debug] Individual MSE losses: {[f'{loss:.6f}' for loss in individual_losses]}")
        # print(f"[SSL Debug] Total consistency loss: {total_loss:.6f}")
        
        return total_loss
    
    def compute_ssl_reward(self, expr: Expression, embedding: Optional[Tensor] = None) -> float:
        """
        Compute SSL (Self-Supervised Learning) reward based on structural consistency
        
        Args:
            expr: The expression to compute SSL reward for
            embedding: The structural embedding of the expression
            
        Returns:
            SSL reward (positive value, higher is better)
        """
        # print(f"\n[SSL Debug] ===== Computing SSL Reward =====")
        # print(f"[SSL Debug] Pool size: {self.size}, K: {self.ssl_k}, τ: {self.ssl_tau}")
        
        if embedding is None:
            # print(f"[SSL Debug] No embedding provided, SSL reward = 0.0")
            return 0.0
            
        if self.size <= 1:
            # print(f"[SSL Debug] Pool size <= 1, SSL reward = 0.0")
            return 0.0
        
        # print(f"[SSL Debug] Query embedding shape: {embedding.shape}")
        # print(f"[SSL Debug] Query embedding norm: {torch.norm(embedding).item():.4f}")
        
        # Find k nearest neighbors based on embedding similarity (excluding self)
        neighbor_indices = self._find_k_nearest_neighbors(embedding, self.ssl_k, exclude_self=True)
        
        if not neighbor_indices:
            # print(f"[SSL Debug] No neighbors found, SSL reward = 0.0")
            return 0.0
        
        # Compute similarity weights
        weights = self._compute_similarity_weights(embedding, neighbor_indices)
        
        if len(weights) == 0:
            # print(f"[SSL Debug] No weights computed, SSL reward = 0.0")
            return 0.0
        
        # Get normalized factor values
        query_value = self._normalize_by_day(expr.evaluate(self.data))
        
        # Compute consistency loss
        consistency_loss = self._compute_consistency_loss(query_value, neighbor_indices, weights)
        
        # Transform consistency loss to SSL reward
        ssl_reward = np.exp(-consistency_loss)
        # print(f"[SSL Debug] Final SSL reward: {ssl_reward:.6f}")
        # print(f"[SSL Debug] ===== SSL Reward Computation Done =====\n")
        
        return ssl_reward
    
    def try_new_expr_with_ssl(self, expr: Expression, embedding: Optional[Tensor] = None) -> Tuple[float, float, float]:
        """
        Compute both IC reward and SSL reward separately
        
        Args:
            expr: The expression to evaluate
            embedding: The structural embedding of the expression
            
        Returns:
            Tuple of (ic_reward, nov_reward, ssl_reward)
        """
        # print(f"\n[SSL Debug] ===== Computing IC + SSL Rewards =====")
        # print(f"[SSL Debug] Expression: {expr}")
        
        # Get IC reward first
        ic_reward, nov_reward = self.try_new_expr(expr, embedding)
        # print(f"[SSL Debug] IC reward: {ic_reward:.6f}")
        
        # Get SSL reward
        ssl_reward = 0.0
        if embedding is not None:
            ssl_reward = self.compute_ssl_reward(expr, embedding)
        else:
            # print(f"[SSL Debug] No embedding provided, SSL reward = 0.0")
            pass
        
        # print(f"[SSL Debug] Final rewards - IC: {ic_reward:.6f}, SSL: {ssl_reward:.6f}")
        # print(f"[SSL Debug] ===== IC + SSL Rewards Computation Done =====\n")
        
        return ic_reward, nov_reward, ssl_reward
    
    def debug_embedding_similarities(self, query_embedding: Tensor) -> None:
        """
        Debug method to analyze embedding similarities in the pool
        
        Args:
            query_embedding: The embedding to compare against pool embeddings
        """
        # print(f"\n[SSL Debug] ===== Embedding Similarity Analysis =====")
        # print(f"[SSL Debug] Pool size: {self.size}")
        # print(f"[SSL Debug] Query embedding shape: {query_embedding.shape}")
        # print(f"[SSL Debug] Query embedding norm: {torch.norm(query_embedding).item():.6f}")
        
        if self.size == 0:
            # print(f"[SSL Debug] Empty pool")
            return
            
        for i in range(self.size):
            if self.embeddings[i] is not None:
                dist = torch.norm(query_embedding - self.embeddings[i]).item()
                cosine_sim = F.cosine_similarity(query_embedding.unsqueeze(0), self.embeddings[i].unsqueeze(0)).item()
                
                status = "IDENTICAL" if dist < 1e-6 else "DIFFERENT"
                # print(f"[SSL Debug] Factor {i}: L2_dist={dist:.6f}, cosine_sim={cosine_sim:.6f} [{status}]")
            else:
                # print(f"[SSL Debug] Factor {i}: No embedding stored")
                pass
                
        # print(f"[SSL Debug] ===== Embedding Analysis Done =====\n")