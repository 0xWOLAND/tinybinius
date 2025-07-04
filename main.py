from algebra.ff.gf2 import GF2
from algebra.extensions.gf2n import GF8
import hashlib

def fiat_shamir(transcript):
    h = hashlib.sha256(transcript).digest()
    return GF8([GF2((h[i//8] >> (i%8)) & 1) for i in range(3)])

def multilinear_eval(evals, point):
    for x in point:
        half = len(evals) // 2
        x8, one8 = GF8([x, GF2(0), GF2(0)]), GF8([GF2(1), GF2(0), GF2(0)])
        evals = [(one8 + x8) * evals[i] + x8 * evals[i + half] for i in range(half)]
    return evals[0]

def sumcheck_round(evals, transcript):
    half = len(evals) // 2
    sum_0 = sum(evals[:half], GF8([GF2(0), GF2(0), GF2(0)]))
    sum_1 = sum(evals[half:], GF8([GF2(0), GF2(0), GF2(0)]))
    transcript += str(sum_0).encode() + str(sum_1).encode()
    challenge = fiat_shamir(transcript)
    one8 = GF8([GF2(1), GF2(0), GF2(0)])
    folded = [(one8 + challenge) * evals[i] + challenge * evals[i + half] for i in range(half)]
    return folded, (sum_0, sum_1), challenge, transcript

def binius_prove(evals, eval_point):
    transcript, rounds, current = b"binius_proof", [], evals[:]
    for _ in eval_point:
        current, round_data, challenge, transcript = sumcheck_round(current, transcript)
        rounds.append((round_data, challenge))
    return {'rounds': rounds, 'final_eval': current[0]}

def binius_verify(proof, claimed_eval, eval_point):
    transcript, current_sum = b"binius_proof", claimed_eval
    for round_data, challenge in proof['rounds']:
        sum_0, sum_1 = round_data
        transcript += str(sum_0).encode() + str(sum_1).encode()
        if challenge != fiat_shamir(transcript) or sum_0 + sum_1 != current_sum:
            return False
        one8 = GF8([GF2(1), GF2(0), GF2(0)])
        current_sum = (one8 + challenge) * sum_0 + challenge * sum_1
    return current_sum == proof['final_eval']

def main():
    evals = [GF8([GF2(i%2), GF2((i//2)%2), GF2((i//4)%2)]) for i in range(8)]
    claimed_sum = sum(evals, GF8([GF2(0), GF2(0), GF2(0)]))
    proof = binius_prove(evals, [GF2(0), GF2(0), GF2(0)])
    print(f"Valid: {binius_verify(proof, claimed_sum, [GF2(0), GF2(0), GF2(0)])}")
    print(f"Invalid: {binius_verify(proof, GF8([GF2(1), GF2(1), GF2(1)]), [GF2(0), GF2(0), GF2(0)])}")

if __name__ == "__main__":
    main()
