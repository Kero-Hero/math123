"""
色域映射数据集模块
提供数据集管理、缓存和预处理功能
"""

import os
import pickle
import numpy as np
from typing import Tuple, Dict, List, Optional, Union, Any
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.color_conversion import rgb_to_lab, generate_color_space_volume
from data.sampler import ColorGamutSampler


class ColorMappingDataset:
    """色域映射数据集类"""
    
    def __init__(self, 
                 source_gamut: str = 'bt2020',
                 target_gamut: str = 'srgb',
                 cache_dir: str = './cache',
                 use_cache: bool = True):
        """
        初始化数据集
        
        Args:
            source_gamut: 源色域
            target_gamut: 目标色域  
            cache_dir: 缓存目录
            use_cache: 是否使用缓存
        """
        self.source_gamut = source_gamut
        self.target_gamut = target_gamut
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        
        # 数据缓存
        self._train_data = None
        self._val_data = None
        self._test_data = None
        
        # 创建缓存目录
        if use_cache:
            os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_path(self, dataset_type: str, n_samples: int, strategy: str) -> str:
        """获取缓存文件路径"""
        filename = f"{self.source_gamut}_to_{self.target_gamut}_{dataset_type}_{n_samples}_{strategy}.pkl"
        return os.path.join(self.cache_dir, filename)
    
    def _load_from_cache(self, cache_path: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """从缓存加载数据"""
        if not self.use_cache or not os.path.exists(cache_path):
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
                return data['source'], data['target']
        except Exception as e:
            print(f"缓存加载失败: {e}")
            return None
    
    def _save_to_cache(self, 
                      cache_path: str, 
                      source_data: np.ndarray, 
                      target_data: np.ndarray):
        """保存数据到缓存"""
        if not self.use_cache:
            return
        
        try:
            data = {
                'source': source_data,
                'target': target_data,
                'source_gamut': self.source_gamut,
                'target_gamut': self.target_gamut,
                'timestamp': np.datetime64('now')
            }
            
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
                
            print(f"数据已缓存到: {cache_path}")
            
        except Exception as e:
            print(f"缓存保存失败: {e}")
    
    def generate_dataset(self, 
                        n_samples: int,
                        strategy: str = 'perceptual',
                        dataset_type: str = 'train') -> Tuple[np.ndarray, np.ndarray]:
        """
        生成数据集
        
        Args:
            n_samples: 样本数量
            strategy: 采样策略
            dataset_type: 数据集类型 ('train', 'val', 'test')
            
        Returns:
            源数据, 目标数据
        """
        # 检查缓存
        cache_path = self._get_cache_path(dataset_type, n_samples, strategy)
        cached_data = self._load_from_cache(cache_path)
        
        if cached_data is not None:
            print(f"从缓存加载 {dataset_type} 数据: {n_samples} 样本")
            return cached_data
        
        # 生成新数据
        print(f"生成 {dataset_type} 数据: {n_samples} 样本 (策略: {strategy})")
        
        sampler = ColorGamutSampler(
            source_gamut=self.source_gamut,
            target_gamut=self.target_gamut,
            sampling_strategy=strategy
        )
        
        source_data, target_data = sampler.sample(n_samples)
        
        # 保存到缓存
        self._save_to_cache(cache_path, source_data, target_data)
        
        return source_data, target_data
    
    def get_train_data(self, 
                      n_samples: int = 50000,
                      strategy: str = 'perceptual') -> Tuple[np.ndarray, np.ndarray]:
        """获取训练数据"""
        if self._train_data is None:
            self._train_data = self.generate_dataset(n_samples, strategy, 'train')
        return self._train_data
    
    def get_validation_data(self, 
                           n_samples: int = 5000,
                           strategy: str = 'uniform') -> Tuple[np.ndarray, np.ndarray]:
        """获取验证数据"""
        if self._val_data is None:
            self._val_data = self.generate_dataset(n_samples, strategy, 'val')
        return self._val_data
    
    def get_test_data(self, 
                     n_samples: int = 2000,
                     strategy: str = 'uniform') -> Tuple[np.ndarray, np.ndarray]:
        """获取测试数据"""
        if self._test_data is None:
            self._test_data = self.generate_dataset(n_samples, strategy, 'test')
        return self._test_data
    
    def get_all_data(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """获取所有数据集"""
        return {
            'train': self.get_train_data(),
            'validation': self.get_validation_data(),
            'test': self.get_test_data()
        }
    
    def clear_cache(self):
        """清除缓存"""
        if not self.use_cache or not os.path.exists(self.cache_dir):
            return
        
        import glob
        cache_pattern = os.path.join(self.cache_dir, f"{self.source_gamut}_to_{self.target_gamut}_*.pkl")
        cache_files = glob.glob(cache_pattern)
        
        for cache_file in cache_files:
            try:
                os.remove(cache_file)
                print(f"已删除缓存: {cache_file}")
            except Exception as e:
                print(f"删除缓存失败: {e}")
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """获取数据集信息"""
        info = {
            'source_gamut': self.source_gamut,
            'target_gamut': self.target_gamut,
            'cache_dir': self.cache_dir,
            'use_cache': self.use_cache,
            'datasets': {}
        }
        
        # 检查已缓存的数据集
        if os.path.exists(self.cache_dir):
            import glob
            cache_pattern = os.path.join(self.cache_dir, f"{self.source_gamut}_to_{self.target_gamut}_*.pkl")
            cache_files = glob.glob(cache_pattern)
            
            for cache_file in cache_files:
                try:
                    with open(cache_file, 'rb') as f:
                        data = pickle.load(f)
                        filename = os.path.basename(cache_file)
                        
                        # 解析文件名获取信息
                        parts = filename.replace('.pkl', '').split('_')
                        if len(parts) >= 5:
                            dataset_type = parts[-3]
                            n_samples = int(parts[-2])
                            strategy = parts[-1]
                            
                            info['datasets'][filename] = {
                                'type': dataset_type,
                                'samples': n_samples,
                                'strategy': strategy,
                                'size_mb': os.path.getsize(cache_file) / (1024 * 1024),
                                'timestamp': str(data.get('timestamp', 'unknown'))
                            }
                            
                except Exception as e:
                    print(f"读取缓存信息失败 {cache_file}: {e}")
        
        return info


class BatchDataLoader:
    """批次数据加载器"""
    
    def __init__(self, 
                 source_data: np.ndarray,
                 target_data: np.ndarray,
                 batch_size: int = 1024,
                 shuffle: bool = True):
        """
        初始化数据加载器
        
        Args:
            source_data: 源数据
            target_data: 目标数据
            batch_size: 批次大小
            shuffle: 是否打乱数据
        """
        self.source_data = source_data
        self.target_data = target_data
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.n_samples = len(source_data)
        self.n_batches = (self.n_samples + batch_size - 1) // batch_size
        
        self.reset()
    
    def reset(self):
        """重置迭代器"""
        self.current_batch = 0
        
        if self.shuffle:
            indices = np.random.permutation(self.n_samples)
            self.source_data = self.source_data[indices]
            self.target_data = self.target_data[indices]
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current_batch >= self.n_batches:
            self.reset()
            raise StopIteration
        
        start_idx = self.current_batch * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.n_samples)
        
        batch_source = self.source_data[start_idx:end_idx]
        batch_target = self.target_data[start_idx:end_idx]
        
        self.current_batch += 1
        
        return batch_source, batch_target
    
    def __len__(self):
        return self.n_batches


def create_dataset(source_gamut: str = 'bt2020',
                  target_gamut: str = 'srgb',
                  cache_dir: str = './cache',
                  use_cache: bool = True) -> ColorMappingDataset:
    """
    创建色域映射数据集
    
    Args:
        source_gamut: 源色域
        target_gamut: 目标色域
        cache_dir: 缓存目录
        use_cache: 是否使用缓存
        
    Returns:
        数据集实例
    """
    return ColorMappingDataset(
        source_gamut=source_gamut,
        target_gamut=target_gamut,
        cache_dir=cache_dir,
        use_cache=use_cache
    )


def benchmark_dataset_generation(n_samples: int = 10000):
    """测试数据集生成性能"""
    import time
    
    print(f"数据集生成性能测试 ({n_samples} 样本)")
    print("-" * 50)
    
    strategies = ['uniform', 'perceptual', 'boundary', 'stratified']
    
    for strategy in strategies:
        print(f"测试策略: {strategy}")
        
        start_time = time.time()
        
        dataset = create_dataset('bt2020', 'srgb', use_cache=False)
        source_data, target_data = dataset.generate_dataset(n_samples, strategy, 'test')
        
        generation_time = time.time() - start_time
        
        print(f"  生成时间: {generation_time:.2f} 秒")
        print(f"  生成速度: {n_samples/generation_time:.0f} 样本/秒") 
        print(f"  数据形状: 源{source_data.shape}, 目标{target_data.shape}")
        print(f"  内存使用: {(source_data.nbytes + target_data.nbytes)/(1024*1024):.2f} MB")
        print()


if __name__ == "__main__":
    # 演示数据集功能
    print("色域映射数据集演示")
    
    # 创建数据集
    dataset = create_dataset('bt2020', 'srgb')
    
    # 获取数据
    train_source, train_target = dataset.get_train_data(1000)
    val_source, val_target = dataset.get_validation_data(200)
    
    print(f"训练数据: {train_source.shape} -> {train_target.shape}")
    print(f"验证数据: {val_source.shape} -> {val_target.shape}")
    
    # 数据加载器演示
    loader = BatchDataLoader(train_source, train_target, batch_size=64)
    
    print(f"批次数据加载器: {len(loader)} 批次")
    
    # 测试前几个批次
    for i, (batch_source, batch_target) in enumerate(loader):
        print(f"批次 {i}: {batch_source.shape} -> {batch_target.shape}")
        if i >= 2:  # 只显示前3个批次
            break
    
    # 数据集信息
    info = dataset.get_dataset_info()
    print("\n数据集信息:")
    for key, value in info.items():
        if key != 'datasets':
            print(f"  {key}: {value}")
    
    # 性能测试
    benchmark_dataset_generation(5000) 