import utils, math

def evaluate_expansion(terrain, resource_map, y, x, C_N, player, players):
    neighbors = [
        (ny, nx)
        for ny, nx in [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]
        if 0 <= ny < C_N and 0 <= nx < C_N and terrain[ny, nx] == 0
    ]

    if not neighbors or terrain[y, x] != player:
        return -float('inf')  # can't expand from here

    # Score each possible target neighbor
    scores = {}
    for ny, nx in neighbors:
        base_score = resource_map[ny, nx]
        enemy_neighbors = sum(
            1 for nny, nnx in [(ny - 1, nx), (ny + 1, nx), (ny, nx - 1), (ny, nx + 1)]
            if 0 <= nny < C_N and 0 <= nnx < C_N and terrain[nny, nnx] not in (0, player)
        )
        scores[(ny, nx)] = base_score * 2 - enemy_neighbors / 2

    max_score = max(scores.values())
    best_neighbors = [pos for pos, s in scores.items() if s == max_score]

    if len(best_neighbors) == 1:
        bny, bnx = best_neighbors[0]
        second_neighbors = [
            (ny, nx)
            for ny, nx in [(bny - 1, bnx), (bny + 1, bnx), (bny, bnx - 1), (bny, bnx + 1)]
            if 0 <= ny < C_N and 0 <= nx < C_N and terrain[ny, nx] == 0
        ]

        scores = {}
        for ny, nx in second_neighbors:
            base_score = resource_map[ny, nx]
            enemy_neighbors = sum(
                1 for nny, nnx in [(ny - 1, nx), (ny + 1, nx), (ny, nx - 1), (ny, nx + 1)]
                if 0 <= nny < C_N and 0 <= nnx < C_N and terrain[nny, nnx] not in (0, player)
            )
            scores[(ny, nx)] = base_score - enemy_neighbors / 2

        if scores:
            max_score += (sum(scores.values()) / len(scores)) * 1.5 + len(scores) / 4

    income_scale = 2 - utils.score_to_probability(players[player]['income'], k=20.0)

    cap_y, cap_x = players[player]['capital']
    dist_to_cap = abs(y - cap_y) + abs(x - cap_x)
    capital_factor = 1 + (1 / max(1, dist_to_cap)) * 5

    return max_score * income_scale * capital_factor

def evaluate_attack(military_map, resource_map, terrain, player, y, x, ny, nx, players):
    """
    Evaluate an attack from (y,x) to (ny,nx).
    Returns a score: higher is better.
    """
    attacker = military_map[y, x]
    defender = military_map[ny, nx]
    defending_player = terrain[ny, nx]

    resource_score = resource_map[ny, nx]

    C_N = military_map.shape[0]
    neighbors = [(ny-1,nx),(ny+1,nx),(ny,nx-1),(ny,nx+1)]
    enemy_neighbors = sum(
        1 for ty,tx in neighbors
        if 0 <= ty < C_N and 0 <= tx < C_N and terrain[ty, tx] not in (0, player)
    )

    if defender == 0:
        score = (attacker / 0.1 * 3.6) + resource_score * 4 - enemy_neighbors / 3
        return score

    score = (attacker / defender * 3.6) + resource_score / 2 - enemy_neighbors / 3

    cap_y, cap_x = players[player]['capital']
    if defending_player in players:
        ncap_y, ncap_x = players[defending_player]['capital']
        dist_to_ncap = abs(ny - ncap_y) + abs(nx - ncap_x)
        score *= 1 + max(0, (7 - dist_to_ncap) * 0.1)

    dist_to_cap = abs(ny - cap_y) + abs(nx - cap_x)
    score *= 1 + max(0, (7 - dist_to_cap) * 0.5)

    uncertainty = 1 / (1 + abs(attacker - defender))
    score *= (1 - 0.5 * uncertainty)

    return score

def evaluate_training(military_map, resource_map, terrain, player, y, x, players):
    """
    Return a score for training a troop at tile (y, x) for the AI player.
    Higher score = better place to train.
    """
    C_N = military_map.shape[0]
    current_troops = military_map[y, x]
    base_resource = resource_map[y, x]

    ratio = (players[player]['army'] + 1) / max(players[player]['income'], 1)
    ratio = min(ratio, 1.0)  # cap at equilibrium

    army_pressure = 1.5 * math.cos(ratio * math.pi / 2)

    threat_radius = 5
    enemy_near_cap = 0
    cap_y, cap_x = players[player]['capital']

    for dy in range(-threat_radius, threat_radius + 1):
        for dx in range(-threat_radius, threat_radius + 1):
            ny, nx = cap_y + dy, cap_x + dx
            if 0 <= ny < C_N and 0 <= nx < C_N:
                if terrain[ny, nx] not in (0, 1, player):
                    if not terrain[ny, nx].startswith('tribe'):
                        distance = abs(dy) + abs(dx)
                        if distance == 0:
                            distance = 0.6
                        enemy_near_cap += military_map[ny, nx] / (distance * 2)

    dist_to_cap = max(0.5, abs(y - cap_y) + abs(x - cap_x))

    capital_bonus = max(1, (1 + 1 / dist_to_cap) * (0.1 + enemy_near_cap))

    closest_enemy_cap_dist = C_N * 2
    for p, pdata in players.items():
        if p == player or pdata['income'] <= 0:
            continue
        ey, ex = pdata['capital']
        dist = abs(ey - y) + abs(ex - x)
        if dist < closest_enemy_cap_dist:
            closest_enemy_cap_dist = dist

    capital_bonus *= 1 + (1 - math.sin(closest_enemy_cap_dist / ((2*C_N) / math.pi) - 1.57)) * 3

    neighbors = [(y-1,x),(y+1,x),(y,x-1),(y,x+1)]
    enemy_threat = sum(
        military_map[ny, nx]
        for ny, nx in neighbors
        if 0 <= ny < C_N and 0 <= nx < C_N and terrain[ny, nx] not in (0, player)
    )

    if enemy_threat == 0:
        score = base_resource * 0.3 / (current_troops + 1)
    else:
        score = enemy_threat + base_resource * 0.5 - current_troops * 0.4

    safe_stocks = max(players[player]['stocks'], 0)
    score += math.log1p(safe_stocks) * 0.5

    income_scale = utils.score_to_probability(players[player]['income'], k=35.0)
    score *= 0.1 + income_scale

    return max(score, 0) * army_pressure + capital_bonus

def evaluate_disband(military_map, resource_map, terrain, player, y, x, players):
    """
    Return a score for disbanding a troop at tile (y, x).
    Higher score = better place to disband.
    """
    C_N = military_map.shape[0]
    current_troops = military_map[y, x]
    base_resource = resource_map[y, x]

    if current_troops <= 0:
        return 0

    ratio = (players[player]['army'] + 1) / max(players[player]['income'], 1)
    ratio = min(ratio, 1.0)

    army_pressure = 1.5 * math.sin(ratio * math.pi / 2)

    if (y, x) == players[player]['capital']:
        capital_malus = 2.0
    else:
        capital_malus = 1.0

    neighbors = [(y-1,x),(y+1,x),(y,x-1),(y,x+1)]
    enemy_threat = sum(
        military_map[ny, nx]
        for ny, nx in neighbors
        if 0 <= ny < C_N and 0 <= nx < C_N and terrain[ny, nx] not in (0, player)
    )

    if enemy_threat == 0:
        score = current_troops * 0.6 - base_resource * 0.3
    else:
        score = current_troops * 0.2 - enemy_threat * 0.8

    if players[player]['stocks'] < 0:
        score += abs(players[player]['stocks']) * 0.4

    income_scale = utils.score_to_probability(players[player]['income'], k=35.0)
    score *= 1.1 - income_scale

    return max(score, 0) * army_pressure / capital_malus