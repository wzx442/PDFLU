# import numpy as np
# import secrets

# """
# 初始化代码：
# 得到客户端的数量
# 为每个客户端分配一个自掩码 b_i
# 为客户端分配种子对 s_{i,j}
# 创建每个种子的秘密份额
# """

# # 定义一个 init_seed_pair 函数, 输入为 num_users, 生成 客户端i 和客户端j 之间的种子对 s_{i,j}
# # 种子对 s_{i,j} 是伪随机函数PRG的输入, 其中 s_{i,j} 是客户端i 和客户端j 之间的种子对, 且 s_{i,j} = s_{j,i}
# def init_seed_pair(num_users: int, t: int, k: int) -> np.ndarray:
#     """为每对客户端生成共享种子。使用二维数组存储, seed_pairs[i][j]表示客户端i和j之间的共享种子。
    
#     Args:
#         num_users (int): 客户端总数
#         t (int): 秘密共享阈值
#         k (int): 秘密共享邻居客户端数量
#     Returns:
#         np.ndarray: 二维数组，存储每对客户端之间的共享种子
#     """
#     # 设置种子长度
#     seed_length = 16
    
#     # 初始化二维数组
#     seed_pairs = np.zeros((num_users, num_users), dtype=np.int64)
    
#     # 为每对客户端生成共享种子
#     for i in range(num_users):
#         for j in range(i + 1, num_users):
#             # 使用密码学安全的随机数生成器
#             seed = secrets.randbits(seed_length)
#             # 存储种子对, 保持对称性
#             seed_pairs[i][j] = seed
#             seed_pairs[j][i] = seed
#     # 为每个种子对生成秘密份额

#     return seed_pairs


# def miller_rabin(n: int, k: int = 5) -> bool:
#     """Miller-Rabin素性测试（k次迭代，默认5次足够安全）"""
#     if n <= 1:
#         return False
#     elif n <= 3:
#         return True
#     elif n % 2 == 0:
#         return False
    
#     # 将n-1分解为d*2^s
#     d = n - 1
#     s = 0
#     while d % 2 == 0:
#         d //= 2
#         s += 1
    
#     # 测试k次
#     for _ in range(k):
#         a = secrets.randbelow(n - 2) + 2  # 随机选取a ∈ [2, n-1]
#         x = pow(a, d, n)
#         if x == 1 or x == n - 1:
#             continue
#         for _ in range(s - 1):
#             x = pow(x, 2, n)
#             if x == n - 1:
#                 break
#         else:
#             return False
#     return True

# def shamir_split(secret: int, threshold: int, num_shares: int, prime: int) -> list[tuple[int, int]]:
#     """
#     Shamir秘密共享拆分（将secret拆分为num_shares个份额，需threshold个份额恢复）
#     参数：
#         secret: 待拆分的秘密（< prime）
#         threshold: 恢复秘密的最小份额数（≥2）
#         num_shares: 生成的份额总数（≥threshold）
#         prime: 有限域质数（需>secret和num_shares）
#     返回：
#         份额列表，每个元素为(x, y)，x∈[1, num_shares]，y为份额值
#     """
#     if not miller_rabin(prime):
#         raise ValueError("prime必须是质数")
#     if secret >= prime:
#         raise ValueError("秘密必须小于质数")
#     if threshold < 2:
#         raise ValueError("阈值必须至少为2")
#     if num_shares < threshold:
#         raise ValueError("份额数必须≥阈值")
    
#     # 生成k-1个随机系数（安全随机）
#     coefficients = [secrets.randbelow(prime) for _ in range(threshold - 1)]
    
#     # 定义多项式f(x) = secret + a1*x + a2*x² + ... + a(k-1)*x^(k-1)
#     def _poly(x: int) -> int:
#         result = secret
#         power = x
#         for coeff in coefficients:
#             result = (result + coeff * power) % prime
#             power = (power * x) % prime
#         return result
    
#     # 生成份额（x从1到num_shares）
#     shares = [(i, _poly(i)) for i in range(1, num_shares + 1)]
#     return shares

# def shamir_recover(shares: list[tuple[int, int]], prime: int) -> int:
#     """从份额中恢复秘密（需至少threshold个份额）"""
#     if not miller_rabin(prime):
#         raise ValueError("prime必须是质数")
#     if len(shares) < 2:
#         raise ValueError("至少需要2个份额恢复秘密")
    
#     secret = 0
#     k = len(shares)
#     for j in range(k):
#         x_j, y_j = shares[j]
#         # 计算拉格朗日基多项式L_j(0)
#         lagrange = 1
#         for m in range(k):
#             if m == j:
#                 continue
#             x_m, _ = shares[m]
#             numerator = (-x_m) % prime  # 分子：-x_m
#             denominator = (x_j - x_m) % prime  # 分母：x_j - x_m
#             # 分母的模逆元（费马小定理：a^(p-2) mod p）
#             inv_denominator = pow(denominator, prime - 2, prime)
#             lagrange = (lagrange * numerator * inv_denominator) % prime
#         # 累加y_j * L_j(0)
#         secret = (secret + y_j * lagrange) % prime
#     return secret

# def generate_seeds(
#     num_clients: int,
#     threshold: int | None = None
# ) -> tuple[list[tuple[int, int]], tuple[list[tuple[int, int]], list[tuple[int, int]]]]:
#     """
#     生成双掩码盲化的种子对和秘密共享份额
#     参数：
#         num_clients: 客户端数量N
#         threshold: 恢复全局掩码的最小客户端数（默认N//2 + 1，即多数派）
#         prime: 有限域质数（默认10^18+3，已验证为质数）
#     返回：
#         client_seed_pairs: 每个客户端的种子对列表，格式[(m1_1, m2_1), (m1_2, m2_2), ...]
#         seed_shares: 全局掩码的秘密共享份额，格式(shares_M1, shares_M2)
#     """
#     # 默认参数处理
#     if threshold is None:
#         threshold = num_clients // 2 + 1  # 多数派阈值（如N=5→3，N=4→3）

#     prime = 10**18 + 3  # 预定义大质数（已验证为质数）
    
#     # 参数合法性检查
#     if threshold < 2:
#         raise ValueError("阈值必须≥2")
#     if num_clients < threshold:
#         raise ValueError("客户端数量必须≥阈值")
#     if prime <= num_clients:
#         raise ValueError("质数必须大于客户端数量")
#     if not miller_rabin(prime):
#         raise ValueError("prime必须是质数")
    
#     # 步骤1：生成互补全局掩码
#     M1 = secrets.randbelow(prime)  # 安全随机生成M1
#     M2 = (-M1) % prime  # M2 = -M1 mod prime（互补）
    
#     # 步骤2：拆分全局掩码为客户端份额
#     shares_M1 = shamir_split(M1, threshold, num_clients, prime)  # M1的份额
#     shares_M2 = [(x, (-y) % prime) for x, y in shares_M1]  # M2的份额（利用线性性，无需重复拆分）
    
#     # 步骤3：生成每个客户端的种子对（m1_i, m2_i）
#     client_seed_pairs = [(shares_M1[i][1], shares_M2[i][1]) for i in range(num_clients)]
    
#     # 步骤4：整理返回结果
#     seed_shares = (shares_M1, shares_M2)
#     return client_seed_pairs, seed_shares