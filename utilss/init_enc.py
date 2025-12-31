import secrets
import itertools
from pprint import pprint
import hashlib
import hmac
import random


PRIME = 115792089210356248762697446949407573530086143415290314195533631308867097853951

class Shamir:
    def __init__(self, prime=PRIME):
        self.prime = prime

    def _evaluate_poly(self, poly, x):
        result = 0
        for coeff in reversed(poly):
            result = (result * x + coeff) % self.prime
        return result

    def _extended_gcd(self, a, b):
        x, y = 0, 1
        last_x, last_y = 1, 0
        while b != 0:
            quotient = a // b
            a, b = b, a % b
            last_x, x = x, last_x - quotient * x
            last_y, y = y, last_y - quotient * y
        return last_x, last_y

    def _mod_inverse(self, n):
        try:
            return pow(n, -1, self.prime)
        except TypeError:
            inv, _ = self._extended_gcd(n, self.prime)
            return inv % self.prime

    def split(self, secret_int: int, num_shares: int, threshold: int):
        if secret_int >= self.prime:
            raise ValueError("Secret is too large for the chosen prime field.")
        if threshold > num_shares:
            raise ValueError("Threshold cannot be greater than the number of shares.")

        coeffs = [secret_int]
        for _ in range(threshold - 1):
            coeffs.append(secrets.randbelow(self.prime))

        shares = []
        for i in range(1, num_shares + 1):
            x = i
            y = self._evaluate_poly(coeffs, x)
            shares.append((x, y))
        
        return shares

    def recover(self, shares: list[tuple[int, int]]):
        if not shares:
            raise ValueError("Cannot recover secret from an empty list of shares.")
        
        secret = 0
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
            
            lagrange_basis = (numerator * self._mod_inverse(denominator)) % self.prime
            
            term = (y_j * lagrange_basis) % self.prime
            secret = (secret + term) % self.prime
            
        return secret
    

def generate_seeds_and_shares(num_clients: int, threshold: int, shamir_handler: Shamir):
    if not (1 < threshold <= num_clients):
        raise ValueError("(threshold)must <= (num_clients)")

    self_mask_seeds = {}
    pairwise_seed_matrix = [[None] * num_clients for _ in range(num_clients)]
    self_mask_seed_shares = {}
    pairwise_seed_shares = {}

    for client_id in range(num_clients):
        self_mask_seed_int = secrets.randbits(16)
        self_mask_seeds[client_id] = self_mask_seed_int

        shares = shamir_handler.split(self_mask_seed_int, num_clients, threshold)
        self_mask_seed_shares[client_id] = shares
    
    for client1_id, client2_id in itertools.combinations(range(num_clients), 2):
        pairwise_seed_int = secrets.randbits(16)
        
        pairwise_seed_matrix[client1_id][client2_id] = pairwise_seed_int
        pairwise_seed_matrix[client2_id][client1_id] = pairwise_seed_int

        shares = shamir_handler.split(pairwise_seed_int, num_clients, threshold)
        
        pair_key = tuple(sorted((client1_id, client2_id)))
        pairwise_seed_shares[pair_key] = shares
    return self_mask_seeds, pairwise_seed_matrix, self_mask_seed_shares, pairwise_seed_shares



def print_matrix(matrix):
    for row in matrix:
        formatted_row = [f"{s}" if s else "None" for s in row]
        print(formatted_row)


def sec_agg(online_clients_ls: list[int], num_clients: int, encrypted_data: list[int], self_seeds):
    agg_cipher_from_survivors = sum(encrypted_data[i] for i in online_clients_ls)
    agg_self_masks_from_survivors = sum(self_seeds[i] for i in online_clients_ls)
    agg_cipher_from_survivors -= agg_self_masks_from_survivors

    return agg_cipher_from_survivors


def recover_drop(online_clients_ls: list[int], num_clients: int, threshold: int, shamir_handler: Shamir, pair_matrix, pair_shares):
    drop_client_id_ls = [i for i in range(num_clients) if i not in online_clients_ls]


    recover_drop_client_pair_seed = 0 
    for drop_client_id in drop_client_id_ls:
        
        for i in online_clients_ls:
            pair_key = tuple(sorted((drop_client_id, i)))
            shares_for_pair_recovery = pair_shares[pair_key][:threshold]
            recovered_pairwise_seed = shamir_handler.recover(shares_for_pair_recovery)

            assert recovered_pairwise_seed == pair_matrix[drop_client_id][i]

            if i < drop_client_id:
                recover_drop_client_pair_seed -= recovered_pairwise_seed
            else: # i > drop_client_id
                recover_drop_client_pair_seed += recovered_pairwise_seed
        
    
    
    
    return recover_drop_client_pair_seed

def recover_drop_copy(online_clients_ls: list[int], num_clients: int, agg_online_value, threshold: int, shamir_handler: Shamir, pair_matrix, pair_shares):
    drop_client_id_ls = [i for i in range(num_clients) if i not in online_clients_ls]

    recover_drop_client_pair_seed = 0 
    for drop_client_id in drop_client_id_ls:
        
        for i in online_clients_ls:
            pair_key = tuple(sorted((drop_client_id, i)))
            shares_for_pair_recovery = pair_shares[pair_key][:threshold]
            recovered_pairwise_seed = shamir_handler.recover(shares_for_pair_recovery)

            assert recovered_pairwise_seed == pair_matrix[drop_client_id][i]

            if i < drop_client_id:
                recover_drop_client_pair_seed -= recovered_pairwise_seed
            else: # i > drop_client_id
                recover_drop_client_pair_seed += recovered_pairwise_seed
        
    
    final_result_dropout = agg_online_value + recover_drop_client_pair_seed
    return final_result_dropout