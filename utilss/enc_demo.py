import secrets
import itertools
from pprint import pprint
import hashlib
import hmac
import random

# 为 256 位秘密选择一个足够大的素数 (NIST P-256 prime)
# p = 2^256 - 2^224 + 2^192 + 2^96 - 1
PRIME = 115792089210356248762697446949407573530086143415290314195533631308867097853951

class Shamir:
    """
    一个实现 Shamir 秘密共享方案的类。
    所有计算都在一个大的有限素数域中进行，以确保安全性和准确性。
    """
    def __init__(self, prime=PRIME):
        self.prime = prime

    def _evaluate_poly(self, poly, x):
        """在有限域中计算多项式在点 x 的值"""
        result = 0
        # 从最高次项开始计算，以提高效率 (Horner's method)
        for coeff in reversed(poly):
            result = (result * x + coeff) % self.prime
        return result

    def _extended_gcd(self, a, b):
        """扩展欧几里得算法，用于计算模逆"""
        x, y = 0, 1
        last_x, last_y = 1, 0
        while b != 0:
            quotient = a // b
            a, b = b, a % b
            last_x, x = x, last_x - quotient * x
            last_y, y = y, last_y - quotient * y
        return last_x, last_y

    def _mod_inverse(self, n):
        """计算 n 在模 prime 下的逆元"""
        # Python 3.8+ 支持 pow(n, -1, self.prime)
        try:
            return pow(n, -1, self.prime)
        except TypeError:
            # 兼容旧版本 Python 的回退方案
            inv, _ = self._extended_gcd(n, self.prime)
            return inv % self.prime

    def split(self, secret_int: int, num_shares: int, threshold: int):
        """
        将一个整数秘密分割成多个份额。

        Args:
            secret_int (int): 要分割的秘密，必须是整数且小于素数 PRIME。
            num_shares (int): 要生成的份额总数。
            threshold (int): 恢复秘密所需的最少份额数。

        Returns:
            list[tuple[int, int]]: 份额列表，每个份额是 (x, P(x)) 的形式。
        """
        if secret_int >= self.prime:
            raise ValueError("Secret is too large for the chosen prime field.")
        if threshold > num_shares:
            raise ValueError("Threshold cannot be greater than the number of shares.")

        # 1. 创建一个 k-1 次的多项式，其中 k 是阈值
        # P(x) = a_0 + a_1*x + a_2*x^2 + ... + a_{k-1}*x^{k-1}
        # a_0 就是我们的秘密
        coeffs = [secret_int]
        for _ in range(threshold - 1):
            # 随机选择其他系数
            coeffs.append(secrets.randbelow(self.prime))

        # 2. 在多项式上取 n 个点 (x, P(x)) 作为份额
        # 我们使用 x = 1, 2, 3, ... 作为求值点
        shares = []
        for i in range(1, num_shares + 1):
            x = i
            y = self._evaluate_poly(coeffs, x)
            shares.append((x, y))
        
        return shares

    def recover(self, shares: list[tuple[int, int]]):
        """
        使用一组份额通过拉格朗日插值法恢复秘密。

        Args:
            shares (list[tuple[int, int]]): 用于恢复的份额列表。

        Returns:
            int: 恢复出的原始秘密（整数形式）。
        """
        if not shares:
            raise ValueError("Cannot recover secret from an empty list of shares.")

        # 我们需要找到 P(0)，因为 P(0) = a_0 = secret
        # 拉格朗日插值公式在 x=0 处的值:
        # P(0) = Σ (y_j * l_j(0))
        # 其中 l_j(0) = Π (x_m / (x_m - x_j))  (m != j)
        
        secret = 0
        # 从份额中提取 x 和 y 坐标
        points = {s[0]: s[1] for s in shares}
        x_coords = list(points.keys())

        for j in x_coords:
            y_j = points[j]
            numerator = 1
            denominator = 1
            
            for m in x_coords:
                if m != j:
                    numerator = (numerator * m) % self.prime
                    denominator = (denominator * (m - j)) % self.prime
            
            # 计算 l_j(0) = numerator / denominator
            lagrange_basis = (numerator * self._mod_inverse(denominator)) % self.prime
            
            # 累加到最终结果
            term = (y_j * lagrange_basis) % self.prime
            secret = (secret + term) % self.prime
            
        return secret
    

def generate_seeds_and_shares(num_clients: int, threshold: int, shamir_handler: Shamir):
    """
    为双重掩码联邦学习生成种子和秘密共享份额，并返回四个独立的变量。

    Args:
        num_clients (int): 参与联邦学习的客户端数量。
        threshold (int): 重构一个秘密所需的最小份额数。
        shamir_handler (Shamir): 一个 Shamir 类的实例，用于处理秘密共享。

    Returns:
        tuple: 一个包含四个元素的元组:
            - self_mask_seeds (dict): 字典 {client_id: hex_seed}。
            - pairwise_seed_matrix (list[list[str]]): N*N 矩阵，存储成对种子的十六进制字符串。
            - self_mask_seed_shares (dict): 字典 {client_id: [shares...]}。
            - pairwise_seed_shares (dict): 字典 {(i, j): [shares...]}。
    """
    if not (1 < threshold <= num_clients):
        raise ValueError("阈值(threshold)必须大于1且小于等于客户端总数(num_clients)")

    # 1. 初始化四个独立的数据结构
    self_mask_seeds = {}
    pairwise_seed_matrix = [[None] * num_clients for _ in range(num_clients)]
    self_mask_seed_shares = {}
    pairwise_seed_shares = {}

    # 2. 生成自掩码种子及其份额
    print("--- 正在生成自掩码种子及其份额 ---")
    for client_id in range(num_clients):
        # 生成一个 16 位的整数种子
        self_mask_seed_int = secrets.randbits(16)
        self_mask_seeds[client_id] = self_mask_seed_int

        # 使用我们的 Shamir 实现来生成份额
        shares = shamir_handler.split(self_mask_seed_int, num_clients, threshold)
        self_mask_seed_shares[client_id] = shares
        print(f"为客户端 {client_id} 生成了自掩码种子和份额。")

    # 3. 生成成对掩码种子及其份额
    print("\n--- 正在生成成对掩码种子及其份额 ---")
    for client1_id, client2_id in itertools.combinations(range(num_clients), 2):
        # 生成一个 16 位的整数种子
        pairwise_seed_int = secrets.randbits(16)
        # hex_pairwise_seed = hex(pairwise_seed_int)
        
        # 存入矩阵
        pairwise_seed_matrix[client1_id][client2_id] = pairwise_seed_int
        pairwise_seed_matrix[client2_id][client1_id] = pairwise_seed_int

        # 生成份额
        shares = shamir_handler.split(pairwise_seed_int, num_clients, threshold)
        
        # 存储份额
        pair_key = tuple(sorted((client1_id, client2_id)))
        pairwise_seed_shares[pair_key] = shares
        print(f"为客户端对 {pair_key} 生成了成对掩码种子和份额。")

    # 4. 返回四个独立的变量
    return self_mask_seeds, pairwise_seed_matrix, self_mask_seed_shares, pairwise_seed_shares

### 第 3 步：使用示例和验证


def print_matrix(matrix):
    """辅助函数，用于美观地打印矩阵"""
    for row in matrix:
        formatted_row = [f"{s}" if s else "None" for s in row]
        print(formatted_row)


# 服务器收到带有索引的密文，服务器也知道客户端的总数
def sec_agg(online_clients_ls: list[int], num_clients: int, encrypted_data: list[int], self_seeds):
    """
    服务器聚合加密值
    输入：
        online_clients_ls: 在线客户端的id列表
        num_clients: 客户端总数
        encrypted_data: 加密数据列表
        self_seeds: 自掩码种子字典
    """
    # 服务器先把在线客户端的加密值聚合起来
    agg_cipher_from_survivors = sum(encrypted_data[i] for i in online_clients_ls)
    print(f"幸存者加密数据聚合值: {agg_cipher_from_survivors}")

    # 服务器减去幸存客户端的自掩码
    agg_self_masks_from_survivors = sum(self_seeds[i] for i in online_clients_ls)
    agg_cipher_from_survivors -= agg_self_masks_from_survivors
    print(f"幸存者加密数据聚合值减去幸存者自掩码: {agg_cipher_from_survivors}")

    # # 找到掉线客户端的id
    # drop_client_id_ls = [i for i in range(num_clients) if i not in online_clients_ls]

    # # 恢复掉线客户端的右部加密值，也就是双掩码的值
    # for drop_client_id in drop_client_id_ls:
    #     # 恢复掉线客户端的右部加密值，也就是双掩码的值
    #     recover_drop_client_pair_seed = 0
        
    #     # 关键修复点 1: 使用排序后的元组作为键
    #     for i in online_clients_ls:
    #         pair_key = tuple(sorted((drop_client_id, i)))
    #         shares_for_pair_recovery = pair_shares[pair_key][:threshold]
    #         recovered_pairwise_seed = shamir_handler.recover(shares_for_pair_recovery)

    #         # 检查恢复是否正确
    #         assert recovered_pairwise_seed == pair_matrix[drop_client_id][i]

    #         if i < drop_client_id:
    #             recover_drop_client_pair_seed -= recovered_pairwise_seed
    #         else: # i > drop_client_id
    #             recover_drop_client_pair_seed += recovered_pairwise_seed
    
    # print(f"  - 计算出未抵消的成对掩码总和为: {recover_drop_client_pair_seed}")

    # # 5. 计算最终结果
    # # 最终结果 = (幸存者密文和) - (幸存者自掩码和) + (未抵消的成对掩码)
    # final_result_dropout = agg_cipher_from_survivors + recover_drop_client_pair_seed
    return agg_cipher_from_survivors


def recover_drop(online_clients_ls: list[int], num_clients: int, agg_online_value, threshold: int, shamir_handler: Shamir, pair_matrix, pair_shares):
    """
    服务器恢复掉线客户端的右部加密值
    输入：
        online_clients_ls: 在线客户端的id列表
        num_clients: 客户端总数
        agg_online_value: 在线客户端的加密值聚合
        threshold: 恢复阈值
        shamir_handler: Shamir类实例
        pair_matrix: 成对掩码种子矩阵
        pair_shares: 成对掩码种子份额字典
    """

    # 找到掉线客户端的id
    drop_client_id_ls = [i for i in range(num_clients) if i not in online_clients_ls]

    # 恢复掉线客户端的右部加密值，也就是双掩码的值
    recover_drop_client_pair_seed = 0 # 未抵消的成对掩码总和
    for drop_client_id in drop_client_id_ls:
        
        # 使用排序后的元组作为键
        for i in online_clients_ls:
            pair_key = tuple(sorted((drop_client_id, i)))
            shares_for_pair_recovery = pair_shares[pair_key][:threshold]
            recovered_pairwise_seed = shamir_handler.recover(shares_for_pair_recovery)

            # 检查恢复是否正确
            assert recovered_pairwise_seed == pair_matrix[drop_client_id][i]

            if i < drop_client_id:
                recover_drop_client_pair_seed -= recovered_pairwise_seed
            else: # i > drop_client_id
                recover_drop_client_pair_seed += recovered_pairwise_seed
        
    
    print(f"  - 计算出未抵消的成对掩码总和为: {recover_drop_client_pair_seed}")

    # 5. 计算最终结果
    # 最终结果 = (幸存者密文和) + (未抵消的成对掩码)
    final_result_dropout = agg_online_value + recover_drop_client_pair_seed
    return final_result_dropout

if __name__ == "__main__":
    NUM_CLIENTS = 8
    THRESHOLD = 4
    print(f"初始化设置：客户端数量 = {NUM_CLIENTS}, 恢复阈值 = {THRESHOLD}\n")

    original_data = [1, 2, 3, 4, 5, 6, 7, 8]


    # 创建 Shamir 处理器实例
    shamir = Shamir()


    # 生成种子和份额
    self_seeds, pair_matrix, self_shares, pair_shares = generate_seeds_and_shares(NUM_CLIENTS, THRESHOLD, shamir)



    # 加密
    encrypted_data = {}
    for client_id in range(NUM_CLIENTS):
        # 客户端的 pairwise mask 是： sum_{j>i} PRG(seed_ij) - sum_{j<i} PRG(seed_ji)
        # 您的代码是 sum_{j<i} - sum_{j>i}，为了和您的逻辑保持一致，我这里也用您的版本
        # 注意：这只是一个符号约定，只要所有客户端都遵守，最终会抵消
        pairwise_mask = sum(pair_matrix[client_id][j] for j in range(client_id + 1, NUM_CLIENTS)) - sum(pair_matrix[client_id][j] for j in range(client_id))
        
        # 加密值 = 原始数据 + 自掩码 + 成对掩码
        encrypted_data[client_id] = original_data[client_id] + self_seeds[client_id] + pairwise_mask
        print(f"客户端 {client_id} 的加密值: {encrypted_data[client_id]}")

    dropped_client_id_ls = [0,1]
    online_clients_ls = [i for i in range(NUM_CLIENTS) if i not in dropped_client_id_ls]

    # print("\n--- 生成结果概览 ---")
    print("\n1. 自掩码种子 (self_mask_seeds):")
    pprint(self_seeds)
    
    print("\n2. 成对掩码种子矩阵 (pairwise_seed_matrix):")
    print_matrix(pair_matrix)




    # 服务器聚合在线客户端的加密值
    agg = sec_agg(online_clients_ls, NUM_CLIENTS, encrypted_data, self_seeds)

    # 计算在线客户端的原始数据聚合值
    plain_agg_from_survivors = sum(original_data[i] for i in online_clients_ls)


    # 加密值还缺少掉线客户端的双掩码，需要恢复
    final_result = recover_drop(online_clients_ls, NUM_CLIENTS, agg, THRESHOLD, shamir, pair_matrix, pair_shares)



    print(f"原始的在线用户聚合值：{plain_agg_from_survivors}，最终结果: {final_result}")


