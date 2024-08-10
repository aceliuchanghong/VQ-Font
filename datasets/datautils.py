import numpy as np


def cyclize(loader):
    """
    Cyclize loader
    在深度学习中，为了避免在一个 epoch 内重复加载数据集，可以使用类似的生成器将数据集循环利用
    """
    while True:
        for x in loader:
            yield x


def uniform_indice(end, n_sample, duplicate=False, st=None):
    """
    Sample from [0, end) with (almost) equidistant interval
    从 [0, end) 范围内等距抽样（近似等距），返回抽样的索引
    end：范围的结束值（不包含在内）。
    n_sample：要抽取的样本数量。
    duplicate：是否允许重复。如果为 False，且样本数量超过范围，样本数量会被设置为范围的长度。
    st：起始偏移值
    """
    if end <= 0:
        # 关于 np.int 的报错，可以使用 np.int32 或 np.int64 来替代
        # np.empty(0, dtype=np.int32) 创建了一个大小为 0 的空数组，元素类型为 32 位整数。由于数组大小为 0，因此数组中实际上没有任何元素。
        return np.empty(0, dtype=np.int32)

    # 如果不允许重复且样本数量大于范围 end，将样本数量设为 end，以避免超出范围。
    if not duplicate and n_sample > end:
        n_sample = end

    # NOTE with endpoint=False, np.linspace does not sample the `end` value
    # 使用 np.linspace 在 [0, end) 范围内生成 n_sample 个等间隔的值 endpoint=False 表示不包含 end 值
    indice = np.linspace(0, end, num=n_sample, dtype=np.int32, endpoint=False)
    if st is None and end:
        st = (end - 1 - indice[-1]) // 2
    return indice + st


def uniform_sample(population, n_sample, st=None):
    # 要选择的 Unicode 字符和字体的数量
    # 从 data_meta["valid"] 的不同部分 均匀抽样
    assert not isinstance(population, set), "population should have order"

    N = len(population)
    if n_sample is None:
        return population

    indice = uniform_indice(N, n_sample, st)

    """
    如果是 numpy 数组，直接用生成的索引数组 indice 进行索引并返回。
    如果是列表，使用索引进行列表元素的抽取并返回。
    如果是字符串，使用索引抽取字符并返回一个新的字符串。
    如果 population 是其他类型，抛出类型错误。
    """
    if isinstance(population, np.ndarray):
        return population[indice]
    elif isinstance(population, list):
        return [population[idx] for idx in indice]
    elif isinstance(population, str):
        return ''.join([population[idx] for idx in indice])
    else:
        raise TypeError(type(population))
