import random, math

def sim_battle(attack, defense):
    while attack > 0 and defense > 0:
        a_dice = min(4, attack)
        d_dice = min(3, defense)
        a_rolls = sorted((random.randint(1, 6) for _ in range(a_dice)), reverse=True)
        d_rolls = sorted((random.randint(1, 6) for _ in range(d_dice)), reverse=True)

        for a, d in zip(a_rolls, d_rolls):
            if a > d:
                defense -= 1
            else:
                attack -= 1
            if attack == 0 or defense == 0:
                break

    return attack != 0, max(attack, defense)

def score_to_probability(score, k=1.0):
    if score <= 0:
        return 0.0
    return score / (score + k)