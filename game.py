import numpy as np
import random
import pygame
import sys
from perlin_noise import PerlinNoise
import bot, utils
from collections import defaultdict

C_N = 20
terrain = np.zeros((C_N, C_N), dtype=object)
resource_map = np.ones((C_N, C_N), dtype=int)
military_map = np.zeros((C_N, C_N), dtype=int)
movements = []

turn_hold = False
bot_hold = False

n_players = 2

S_SIZE = 881
CELL_SIZE = round((8 * S_SIZE) / (9 * C_N + 1))
BORDER_SIZE = CELL_SIZE // 8

S_SIZE = C_N * CELL_SIZE + (C_N + 1) * BORDER_SIZE
x_offset = 400

x_size = S_SIZE+x_offset*2

mountain_ranges = 6
range_size = (6, 30)

pygame.init()

if S_SIZE != CELL_SIZE * C_N + BORDER_SIZE * (C_N+1):
    print(f"The size of S_SIZE is invalid ({S_SIZE} ({CELL_SIZE} * {C_N} = {CELL_SIZE * C_N} + {BORDER_SIZE * (C_N+1)} = {CELL_SIZE * C_N + BORDER_SIZE * (C_N+1)})")
    pygame.quit()
    sys.exit()

screen = pygame.display.set_mode((S_SIZE+x_offset*2, S_SIZE))
pygame.display.set_caption("Napoleon")

CAPITAL_STAR = pygame.image.load("assets/star.png").convert_alpha()
CAPITAL_STAR = pygame.transform.smoothscale(CAPITAL_STAR, (CELL_SIZE * 0.5, CELL_SIZE * 0.5))

corners = [(0,0), (C_N-1,C_N-1), (C_N-1,0), (0,C_N-1)]
font = pygame.font.Font(None, 32)
tag_font = pygame.font.Font(None, 16)
big_font = pygame.font.Font(None, 84)

player_amount_rect = pygame.Rect(x_size / 2 - x_size / 6, 100, x_size / 3, 150)
remove_player_rect = pygame.Rect(x_size / 2 - x_size / 6 + 25, 125, 100, 100)
add_player_rect = pygame.Rect(x_size / 2 + x_size / 6 - 125, 125, 100, 100)

red_rect = pygame.Rect(x_size / 2 - x_size / 6, 100 + 170, x_size / 13, x_size / 13)
blue_rect = pygame.Rect(x_size / 2 - x_size / 6 + x_size / 13 + x_size / (13*9), 100 + 170, x_size / 13, x_size / 13)
green_rect = pygame.Rect(x_size / 2 - x_size / 6 + (x_size / 13 + x_size / (13*9)) * 2, 100 + 170, x_size / 13, x_size / 13)
yellow_rect = pygame.Rect(x_size / 2 - x_size / 6 + (x_size / 13 + x_size / (13*9)) * 3, 100 + 170, x_size / 13, x_size / 13)

confirm_rect = pygame.Rect(x_size / 2 - x_size / 6, S_SIZE - 250, x_size / 3, 150)

playing = [0, 0, 0, 0]

playing_codes = {
    0: 'PL',
    1: 'BOT',
    2: 'AI*',
}

def draw_optionscreen():
    pygame.draw.rect(screen, (200,200,200), player_amount_rect, border_radius=30)
    pygame.draw.rect(screen, (150, 150, 150), remove_player_rect, border_radius=20)
    pygame.draw.rect(screen, (150, 150, 150), add_player_rect, border_radius=20)

    pygame.draw.rect(screen, (200, 0, 0), red_rect, border_radius=20)
    pygame.draw.rect(screen, (0, 0, 200) if n_players > 1 else (0, 0, 50), blue_rect, border_radius=20)
    pygame.draw.rect(screen, (0, 200, 0) if n_players > 2 else (0, 50, 0), green_rect, border_radius=20)
    pygame.draw.rect(screen, (200, 200, 0) if n_players > 3 else (50, 50, 0), yellow_rect, border_radius=20)

    pygame.draw.rect(screen, (0, 200, 100), confirm_rect, border_radius=30)

    txt = big_font.render("-", True, (100,100,100))
    txt_rect = txt.get_rect(center=remove_player_rect.center)
    screen.blit(txt, txt_rect)

    txt = big_font.render("+", True, (100,100,100))
    txt_rect = txt.get_rect(center=add_player_rect.center)
    screen.blit(txt, txt_rect)

    txt = big_font.render(str(n_players), True, (100,100,100))
    txt_rect = txt.get_rect(center=player_amount_rect.center)
    screen.blit(txt, txt_rect)

    txt = big_font.render(playing_codes[playing[0]], True, (255, 150, 150))
    txt_rect = txt.get_rect(center=red_rect.center)
    screen.blit(txt, txt_rect)

    txt = big_font.render(playing_codes[playing[1]], True, (150, 150, 255) if n_players > 1 else (75, 75, 127))
    txt_rect = txt.get_rect(center=blue_rect.center)
    screen.blit(txt, txt_rect)

    txt = big_font.render(playing_codes[playing[2]], True, (150, 255, 150) if n_players > 2 else (75, 127, 75))
    txt_rect = txt.get_rect(center=green_rect.center)
    screen.blit(txt, txt_rect)

    txt = big_font.render(playing_codes[playing[3]], True, (255, 255, 150) if n_players > 3 else (127, 127, 75))
    txt_rect = txt.get_rect(center=yellow_rect.center)
    screen.blit(txt, txt_rect)

    txt = big_font.render('Play', True, (150, 255, 200))
    txt_rect = txt.get_rect(center=confirm_rect.center)
    screen.blit(txt, txt_rect)

clock = pygame.time.Clock()
choosing = True
while choosing:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            choosing = False
            pygame.quit()
            break
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                if remove_player_rect.collidepoint(event.pos):
                    n_players = max(n_players - 1, 1)
                elif add_player_rect.collidepoint(event.pos):
                    n_players = min(n_players + 1, 4)

                elif red_rect.collidepoint(event.pos):
                    playing[0] = (playing[0] + 1 ) % 3
                elif blue_rect.collidepoint(event.pos):
                    playing[1] = (playing[1] + 1 ) % 3
                elif green_rect.collidepoint(event.pos):
                    playing[2] = (playing[2] + 1 ) % 3
                elif yellow_rect.collidepoint(event.pos):
                    playing[3] = (playing[3] + 1 ) % 3

                elif confirm_rect.collidepoint(event.pos):
                    choosing = False
                    break
    draw_optionscreen()
    pygame.display.flip()
    clock.tick(60)

colors = {
    0: {'color': (100, 200, 100)},
    1: {'color': (100, 100, 100)},
}

available_players = {
    "red": {'color': (200, 0, 0),
            'bkg_color': (50, 20, 20),
            'stocks': 20,
            'income': 0,
            'army': 0,
            'capital': None},
    "blue": {'color': (0, 0, 200),
             'bkg_color': (20, 20, 50),
             'stocks': 20,
             'income': 0,
             'army': 0,
             'capital': None},
    "green": {'color': (0, 200, 0),
             'bkg_color': (20, 50, 20),
             'stocks': 20,
             'income': 0,
             'army': 0,
             'capital': None},
    "yellow": {'color': (200, 200, 0),
             'bkg_color': (50, 50, 20),
             'stocks': 20,
             'income': 0,
             'army': 0,
             'capital': None},
}

players = {}

available_keys = list(available_players.keys())

for i in range(min(4, n_players)):
    key = available_keys[i]
    players[key] = available_players[key].copy()

tribes = {}

res_colors = {
    0: [(48, 34, 10)],
    1: [(81, 56, 13)],
    2: [(122, 111, 13)],
    3: [(112, 137, 11)],
    4: [(35, 193, 0)],
    'outbound': [(4, 232, 148)],
}

mil_amount_colors = {
 0: [(0, 0, 0)],
 1: [(18, 18, 28)],
 2: [(36, 36, 56)],
 3: [(54, 54, 85)],
 4: [(63, 63, 113)],
 5: [(90, 90, 141)],
 6: [(108, 108, 170)],
 7: [(126, 126, 198)],
 8: [(150, 150, 226)],
 9: [(171, 171, 255)],
 'outbound': [(189, 189, 255)],
}

colors.update(players)

vis_mode = 0
overlay = None

expand_price = 10
troop_price = 20
troop_move_price = 10
troop_upkeep = 1
mil_actions = False

#=== UI ELEMENTS === O

next_turn_rect = pygame.Rect(x_size - 5 - x_offset/2, S_SIZE - 5 - x_offset/6, x_offset / 2, x_offset / 6)
coins_rect = pygame.Rect(x_size - 5 - x_offset/3, 5, x_offset / 3, x_offset / 7)
army_rect = pygame.Rect(x_size - 15 - x_offset/3 * 2, 5, x_offset / 3, x_offset / 7)
coords_rect = pygame.Rect(20, 10, x_offset / 2, x_offset / 7)
prod_rect = pygame.Rect(30 + x_offset / 2, 10, x_offset / 3, x_offset / 7)
expand_rect = pygame.Rect(20, 40 + x_offset / 7, x_offset / 1.3, x_offset / 7)
mil_actions_rect = pygame.Rect(20, 50 + (x_offset / 7) * 2, x_offset / 1.3, x_offset / 7)
slider_rect = pygame.Rect(40, 60 + (x_offset / 7) * 3, x_offset / 1.5, x_offset / 7)
make_mil_rect = pygame.Rect(40, 70 + (x_offset / 7) * 4, x_offset / 1.5, x_offset / 7)
move_mil_rect = pygame.Rect(40, 80 + (x_offset / 7) * 5, x_offset / 1.5, x_offset / 7)

#=== UI ELEMENTS === C

def generate_resources(resource_map, scale=10):
    seed = random.randint(0, 10000)
    noise = PerlinNoise(octaves=4, seed=seed)


    for y in range(C_N):
        for x in range(C_N):
            nx = x / scale
            ny = y / scale

            v = noise([nx, ny])
            v = (v + 1) / 2

            if v > 0.8:
                resource_map[y, x] = 5
            elif v > 0.65:
                resource_map[y, x] = 4
            elif v > 0.55:
                resource_map[y, x] = 3
            elif v > 0.45:
                resource_map[y, x] = 2
            elif v > 0.3:
                resource_map[y, x] = 1
            else:
                resource_map[y, x] = 0

generate_resources(resource_map)

from collections import deque

def is_map_connected(terrain):
    """Check if all passable cells (terrain != 1) are connected using BFS"""
    visited = set()
    C_N = terrain.shape[0]

    for y in range(C_N):
        for x in range(C_N):
            if terrain[y, x] != 1:
                start = (y, x)
                break
        else:
            continue
        break
    else:
        return False

    queue = deque([start])
    visited.add(start)
    while queue:
        y, x = queue.popleft()
        for ny, nx in [(y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)]:
            if 0 <= ny < C_N and 0 <= nx < C_N:
                if terrain[ny, nx] != 1 and (ny, nx) not in visited:
                    visited.add((ny, nx))
                    queue.append((ny, nx))

    free_cells = sum(terrain[y, x] != 1 for y in range(C_N) for x in range(C_N))
    return len(visited) == free_cells

for _ in range(mountain_ranges):
    x = random.randint(0, C_N - 1)
    y = random.randint(0, C_N - 1)
    steps = random.randint(*range_size)

    for _ in range(steps):
        terrain[y, x] = 1
        dx, dy = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
        x_new = max(0, min(C_N - 1, x + dx))
        y_new = max(0, min(C_N - 1, y + dy))

        terrain[y_new, x_new] = 1
        if not is_map_connected(terrain):
            terrain[y_new, x_new] = 0
        else:
            x, y = x_new, y_new

def claim_tile(y, x, new_owner):
    old_owner = terrain[y, x]
    terrain[y, x] = new_owner

    if old_owner in players:
        players[old_owner]["income"] -= resource_map[y, x] * (1 + (players[old_owner]['capital'] == (y, x)))

    if new_owner in players:
        players[new_owner]["income"] += resource_map[y, x] * (1 + (players[new_owner]['capital'] == (y, x)))

def create_tribe(y, x):
    global tribes, military_map, terrain
    tribe_name = f'tribe{x}_{y}'
    tribes[tribe_name] = {
        "color": (random.randint(160, 255), random.randint(200, 255), random.randint(160, 255)),
        "army": random.randint(1, resource_map[y, x] + 2)
    }
    claim_tile(y, x, tribe_name)
    military_map[y, x] = tribes[tribe_name]["army"]
    colors.update(tribes)

def generate_tribes():
    for y in range(C_N):
        for x in range(C_N):
            if terrain[y, x] == 0:
                if (resource_map[y, x] + 0.5) * 0.1 > random.random():
                    create_tribe(y, x)

generate_tribes()

def free_neighbors(y, x, terrain):
    H, W = terrain.shape
    count = 0
    for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
        ny, nx = y + dy, x + dx
        if 0 <= ny < H and 0 <= nx < W:
            if terrain[ny, nx] == 0:
                count += 1
    return count

def find_start_cell(corner):
    H, W = terrain.shape
    cy, cx = corner

    if terrain[cy, cx] == 0:
        if free_neighbors(cy, cx, terrain) >= 2:
            return (cy, cx)

    for radius in range(1, max(H,W)):
        for dy in range(-radius, radius+1):
            for dx in range(-radius, radius+1):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < H and 0 <= nx < W:
                    if terrain[ny, nx] == 0:
                        if free_neighbors(ny, nx, terrain) >= 2:
                            return (ny, nx)
    return None

for pid, corner in zip(players, corners):
    cell = find_start_cell(corner)
    if cell:
        y, x = cell
        players[pid]["capital"] = (y, x)
        claim_tile(y, x, pid)
    else:
        raise RuntimeError(f"Cannot place player {pid}")

resource_map[terrain == 1] = 0

player_order = list(players.keys())
player_turn = 0
selected = None
selected_troop = None
sel_troop_amount = 0

def next_turn():
    global player_turn
    global selected
    global selected_troop
    global turn_hold

    turn_hold = True

    selected = None
    selected_troop = None
    current = player_order[player_turn]

    if terrain[players[current]["capital"]] != current:
        collapse_land(current)
    players[current]["stocks"] -= players[current]["army"]

    if players[current]["stocks"] < 0:
        deficit = -players[current]["stocks"]

        troop_positions = []
        for y in range(C_N):
            for x in range(C_N):
                if terrain[y, x] == current and military_map[y, x] > 0:
                    troop_positions.extend([(y, x)] * military_map[y, x])

        random.shuffle(troop_positions)

        kills = 0
        for i in range(min(deficit, len(troop_positions))):
            if random.random() < 0.2:
                y, x = troop_positions[i]
                military_map[y, x] -= 1
                players[current]["army"] -= 1
                kills += 1

        players[current]["stocks"] = 0

        if kills > 0:
            print(f"Player {current} lost {kills} troops due to upkeep deficit ({deficit})")
        players[current]["stocks"] = 0

    players[current]["stocks"] += players[current]["income"]
    player_turn = (player_turn + 1) % len(player_order)

    by_target = defaultdict(list)

    for movement in movements:
        by_target[movement['target']].append(movement)

    for target, grouped_movements in by_target.items():
        move_troop(grouped_movements)

    movements.clear()

    turn_hold = False

def collapse_land(pid):
    owned_tiles = [(y, x) for y in range(C_N) for x in range(C_N) if terrain[y, x] == pid]

    for y, x in owned_tiles:
        distances = []
        for oy, ox in owned_tiles:
            distances.append(abs(oy - y) + abs(ox - x))
        max_dist = max(distances) or 1
        dist = abs(y - players[pid]['capital'][0]) + abs(x - players[pid]['capital'][1])

        collapse_prob = dist / max_dist * 0.3

        if random.random() < collapse_prob:
            if random.random() < 0.8:
                claim_tile(y, x, 0)
            else:
                create_tribe(y, x)

def expand_territory():
    current = player_order[player_turn]
    if not selected:
        return None

    y, x = selected
    if terrain[y, x] != current:
        return None

    if players[current]["stocks"] < expand_price:
        return None

    neighbors = [
        (y-1, x), (y+1, x),
        (y, x-1), (y, x+1)
    ]

    free_neighbors = [
        (ny, nx)
        for ny, nx in neighbors
        if 0 <= ny < C_N and 0 <= nx < C_N and terrain[ny, nx] == 0
    ]

    if not free_neighbors:
        return None

    max_resource = max(resource_map[ny, nx] for ny, nx in free_neighbors)

    best_neighbors = [
        (ny, nx)
        for ny, nx in free_neighbors
        if resource_map[ny, nx] == max_resource
    ]

    ny, nx = random.choice(best_neighbors)

    players[current]["stocks"] -= expand_price
    claim_tile(ny, nx, current)

    return ny, nx

def train_troop(amount=1):
    current = player_order[player_turn]
    amount = min(amount, players[current]["stocks"] // troop_price)

    if amount < 1:
        return False

    if not selected:
        return False

    y, x = selected
    if not terrain[y, x] == current:
        return False

    if players[current]["stocks"] < troop_price * amount:
        return False

    players[current]["stocks"] -= troop_price * amount

    players[current]["army"] += amount

    military_map[y, x] += amount

    return True

def disband_troop(amount=1):
    current = player_order[player_turn]

    if amount < 1:
        return False

    if not selected:
        return False

    y, x = selected
    if not terrain[y, x] == current:
        return False

    if not military_map[y, x] > 0:
        return False

    for movement in movements:
        if movement['origin'] == (y, x):
            movements.remove(movement)

    players[current]["stocks"] += (troop_price * amount) // 2

    players[current]["army"] -= amount

    military_map[y, x] -= amount

    return True

def select_troop(select=True):
    global selected_troop
    global sel_troop_amount
    if select:
        y, x = selected
        current = player_order[player_turn]
        if terrain[y, x] != current:
            return False
        selected_troop = selected
    else:
        selected_troop = None
        sel_troop_amount = 0
        return True
    return True

def create_movement(troop_amount=None):
    current = player_order[player_turn]
    global selected_troop

    used_troops = 0

    if not selected_troop:
        return False
    if not selected:
        return False

    y, x = selected_troop
    sy, sx = selected

    for movement in movements:
        if movement['origin'] == (y, x):
            used_troops += movement['troop_amount']

    if not troop_amount:
        troop_amount = min(military_map[y, x] - used_troops, players[current]["stocks"]//troop_move_price)
        if not troop_amount:
            return False

    if military_map[y, x] < troop_amount or not (troop_amount > 0):
        return False

    if players[current]["stocks"] < troop_move_price * troop_amount:
        return False

    neighbors = [
        (y - 1, x), (y + 1, x),
        (y, x - 1), (y, x + 1)
    ]

    valid_neighbors = [
        (ny, nx)
        for ny, nx in neighbors
        if 0 <= ny < C_N and 0 <= nx < C_N and terrain[ny, nx] != 1
    ]

    if selected not in valid_neighbors:
        return False

    players[current]["stocks"] -= troop_move_price * troop_amount

    movement = {'origin':(y, x),
                'target': (sy, sx),
                'troop_amount': troop_amount,
                'attacker': current,}

    movements.append(movement)

    selected_troop = None
    return True

def move_troop(movements: list):
    if not movements:
        return False

    # All movements share these
    sy, sx = movements[0]['target']
    attacker = movements[0]['attacker']

    total_attackers = 0
    valid_movements = []

    # 1. Validate movements and sum attackers
    for movement in movements:
        y, x = movement['origin']
        troop_amount = movement['troop_amount']

        if military_map[y, x] < troop_amount:
            continue

        neighbors = [
            (y-1, x), (y+1, x),
            (y, x-1), (y, x+1)
        ]

        if (sy, sx) not in [
            (ny, nx)
            for ny, nx in neighbors
            if 0 <= ny < C_N and 0 <= nx < C_N and terrain[ny, nx] != 1
        ]:
            continue

        total_attackers += troop_amount
        valid_movements.append(movement)

    if total_attackers == 0:
        return False

    # 2. Resolve combat or movement
    if terrain[sy, sx] not in (attacker, 0):
        defender = terrain[sy, sx]

        players[attacker]["army"] -= total_attackers
        try:
            players[defender]["army"] -= military_map[sy, sx]
        except KeyError:
            tribes[defender]["army"] -= military_map[sy, sx]

        att_win, surviving = utils.sim_battle(
            total_attackers,
            military_map[sy, sx]
        )

        # Remove troops from origins
        for movement in valid_movements:
            y, x = movement['origin']
            military_map[y, x] -= movement['troop_amount']

        military_map[sy, sx] = surviving

        if att_win:
            players[attacker]["army"] += surviving
            claim_tile(sy, sx, attacker)
        else:
            try:
                players[defender]["army"] += surviving
            except KeyError:
                tribes[defender]["army"] += surviving

    elif terrain[sy, sx] == attacker:
        # Move within territory
        for movement in valid_movements:
            y, x = movement['origin']
            military_map[y, x] -= movement['troop_amount']
        military_map[sy, sx] += total_attackers

    else:
        # Colonization
        for movement in valid_movements:
            y, x = movement['origin']
            military_map[y, x] -= movement['troop_amount']
        military_map[sy, sx] += total_attackers
        claim_tile(sy, sx, attacker)

    return True

def get_tile_from_click(pos):
    mx, my = pos
    mx -= x_offset

    if mx < 0 or my < 0:
        return None
    x = mx // (CELL_SIZE + BORDER_SIZE)
    y = my // (CELL_SIZE + BORDER_SIZE)

    if x >= C_N or y >= C_N:
        return None

    return y, x

def draw_loop(map, colors, text_vals=False, overlay=None):
    rect = pygame.Rect(x_offset, 0, S_SIZE, S_SIZE)
    pygame.draw.rect(screen, (0,0,0), rect)

    for movement in movements:
        rect = pygame.Rect((CELL_SIZE + BORDER_SIZE) * movement['origin'][1] + x_offset + BORDER_SIZE // 2, (CELL_SIZE + BORDER_SIZE) * movement['origin'][0] + BORDER_SIZE // 2, (CELL_SIZE + BORDER_SIZE), (CELL_SIZE + BORDER_SIZE))
        pygame.draw.rect(screen, (255, 200, 200), rect)

        rect = pygame.Rect((CELL_SIZE + BORDER_SIZE) * movement['target'][1] + x_offset + BORDER_SIZE / 2, (CELL_SIZE + BORDER_SIZE) * movement['target'][0] + BORDER_SIZE / 2, (CELL_SIZE + BORDER_SIZE), (CELL_SIZE + BORDER_SIZE))
        pygame.draw.rect(screen, (100, 35, 30), rect)

    if selected_troop:
        rect = pygame.Rect((CELL_SIZE + BORDER_SIZE) * selected_troop[1] + x_offset + BORDER_SIZE / 2, (CELL_SIZE + BORDER_SIZE) * selected_troop[0] + BORDER_SIZE / 2, (CELL_SIZE + BORDER_SIZE), (CELL_SIZE + BORDER_SIZE))
        pygame.draw.rect(screen, (30, 100, 35), rect)

    if selected:
        rect = pygame.Rect((CELL_SIZE + BORDER_SIZE) * selected[1] + x_offset + BORDER_SIZE / 2, (CELL_SIZE + BORDER_SIZE) * selected[0] + BORDER_SIZE / 2, (CELL_SIZE + BORDER_SIZE), (CELL_SIZE + BORDER_SIZE))
        pygame.draw.rect(screen, (200,220,255), rect)

    if text_vals:
        for y in range(C_N):
            for x in range(C_N):
                v = map[y, x]
                color = colors[v][0] if v in colors else colors['outbound'][0]
                rect = pygame.Rect(BORDER_SIZE + (CELL_SIZE + BORDER_SIZE) * x + x_offset, BORDER_SIZE + (CELL_SIZE + BORDER_SIZE) * y, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, color, rect)

                txt = font.render(str(v), True, (255, 255, 255) if terrain[y, x] == 0 else (30,30,30))
                txt_rect = txt.get_rect(center=rect.center)
                screen.blit(txt, txt_rect)

    else:
        if overlay is not None:
            for y in range(C_N):
                for x in range(C_N):
                    v = map[y, x]
                    rx = BORDER_SIZE + (CELL_SIZE + BORDER_SIZE) * x + x_offset
                    ry = BORDER_SIZE + (CELL_SIZE + BORDER_SIZE) * y
                    if v in colors:
                        pygame.draw.rect(
                            screen,
                            colors[v]['color'],
                            (rx, ry, CELL_SIZE, CELL_SIZE)
                        )

                    if terrain[y, x] in players:
                        if (y, x) == players[terrain[y, x]]['capital']:
                            screen.blit(CAPITAL_STAR, (rx, ry))

                    t = overlay[y, x]
                    if t != 0:
                        r = pygame.Rect(rx + BORDER_SIZE, ry + 2/6*CELL_SIZE, CELL_SIZE * 3/4, CELL_SIZE * 3/8)
                        bkg = players.get(terrain[y, x], {}).get('bkg_color', (67, 95, 99))
                        pygame.draw.rect(screen, (0, 0, 20), r, border_radius=5)
                        pygame.draw.rect(screen, bkg, r, border_radius=5, width=2)

                        txt = tag_font.render(str(t), True, (150, 210, 220))
                        txt_rect = txt.get_rect(center=r.center)
                        screen.blit(txt, txt_rect)

        else:
            for y in range(C_N):
                for x in range(C_N):
                    v = map[y, x]
                    rx = BORDER_SIZE + (CELL_SIZE + BORDER_SIZE) * x + x_offset
                    ry = BORDER_SIZE + (CELL_SIZE + BORDER_SIZE) * y
                    if v in colors:
                        pygame.draw.rect(
                            screen,
                            colors[v]['color'],
                            (rx, ry, CELL_SIZE, CELL_SIZE)
                        )

                    if terrain[y, x] in players:
                        if (y, x) == players[terrain[y, x]]['capital']:
                            screen.blit(CAPITAL_STAR, (rx, ry))

def draw_map():
    if vis_mode == 0:
        draw_loop(terrain, colors, overlay=overlay)
    elif vis_mode == 1:
        draw_loop(resource_map, res_colors, True)
    elif vis_mode == 2:
        draw_loop(military_map, mil_amount_colors, True)

def draw_screen():
    current = player_order[player_turn]
    screen.fill(players[current]['bkg_color'] if playing[player_turn] not in (1, 2) else (50,50,50))

    # ------

    pygame.draw.rect(screen, (20,10,30), next_turn_rect, border_radius=15)

    txt = font.render("Next turn >>", True, (200, 150, 200))
    txt_rect = txt.get_rect(center=next_turn_rect.center)
    screen.blit(txt, txt_rect)

    # ------

    pygame.draw.rect(screen, (80, 70, 10), coins_rect, border_radius=15)

    txt = font.render(f'{players[current]['stocks']} (+{players[current]['income']})', True, (200, 200, 150))
    txt_rect = txt.get_rect(center=coins_rect.center)
    screen.blit(txt, txt_rect)

    draw_map()

    # ------

    pygame.draw.rect(screen, (10, 70, 80), army_rect, border_radius=15)

    txt = font.render(f'{players[current]['army']} (-{players[current]['army']})', True, (150, 170, 200))
    txt_rect = txt.get_rect(center=army_rect.center)
    screen.blit(txt, txt_rect)

    draw_map()

    # ------

    if selected:
        y, x = selected
        owned = terrain[y, x] == current

        # ------
        txt_col = (50, 50, 50) if owned else (120, 120, 120)
        base_rect_col = (150, 150, 150) if owned else (50,50,50)
        prod_rect_col = (150, 150, 130) if owned else (50,50,50)
        expand_rect_col = (120, 150, 130) if owned else (50,50,50)
        mil_actions_rect_col = (60, 100, 50) if owned else (50,50,50)
        slider_rect_col = (20, 20, 30) if owned else (50, 50, 50)
        make_mil_rect_col = (150, 120, 130) if owned else (50,50,50)
        move_mil_rect_col = (120, 130, 150) if owned else (50,50,50)

        # ------
        pygame.draw.rect(screen, base_rect_col, coords_rect, border_radius=10)
        txt = font.render(f'({x}, {y})', True, txt_col)
        txt_rect = txt.get_rect(center=coords_rect.center)
        screen.blit(txt, txt_rect)

        # ------
        pygame.draw.rect(screen, prod_rect_col, prod_rect, border_radius=10)
        txt = font.render(f'+{resource_map[y, x]}', True, txt_col)
        txt_rect = txt.get_rect(center=prod_rect.center)
        screen.blit(txt, txt_rect)

        # ------

        pygame.draw.rect(screen, expand_rect_col, expand_rect, border_radius=10)
        txt = font.render(f'Expand ({expand_price})', True, txt_col)
        txt_rect = txt.get_rect(center=expand_rect.center)
        screen.blit(txt, txt_rect)

        # ------
        pygame.draw.rect(screen, mil_actions_rect_col, mil_actions_rect, border_radius=10)
        txt = font.render(f'Manage Troops ({military_map[y, x]}){'v' if mil_actions else '>'}', True, txt_col)
        txt_rect = txt.get_rect(center=mil_actions_rect.center)
        screen.blit(txt, txt_rect)

        if mil_actions:
            pygame.draw.rect(screen, slider_rect_col, slider_rect, border_radius=10)
            txt_rect = txt.get_rect(center=make_mil_rect.center)
            screen.blit(txt, txt_rect)

            pygame.draw.rect(screen, make_mil_rect_col, make_mil_rect, border_radius=10)
            txt = font.render(f'Train Troop ({troop_price})', True, txt_col)
            txt_rect = txt.get_rect(center=make_mil_rect.center)
            screen.blit(txt, txt_rect)

            pygame.draw.rect(screen, move_mil_rect_col, move_mil_rect, border_radius=10)
            txt = font.render(f'Move Troops ({troop_move_price}/troop)', True, txt_col)
            txt_rect = txt.get_rect(center=move_mil_rect.center)
            screen.blit(txt, txt_rect)

def bot_turn(player):
    """Bot turn for a player using existing game functions."""

    global selected
    global selected_troop
    global bot_hold

    bot_hold = True

    selected = None
    selected_troop = None
    used_troops = np.zeros((C_N, C_N), dtype=int)
    attempted_attacks = set()

    tiles = [(y, x) for y in range(C_N) for x in range(C_N)]
    random.shuffle(tiles)

    made_action = True
    while made_action:
        made_action = False

        # 1 - Train Troops: pick best tiles first
        best_tile = None
        best_score = -float('inf')

        for y, x in tiles:
            if terrain[y, x] == player:
                score = bot.evaluate_training(military_map, resource_map, terrain, player, y, x, players)
                if score > best_score:
                    best_score = score
                    best_tile = (y, x)

        if best_tile and best_score > 0:
            prob = utils.score_to_probability(best_score)

            if random.random() < prob:
                selected = best_tile
                if train_troop():
                    made_action = True
                    print(
                        f"Bot {player} trained troop at {best_tile} "
                        f"(score {best_score:.2f}, prob {prob:.2f})"
                    )

            selected = None

        # 2 - Attack / Move Troops
        attack_candidates = []

        for y, x in tiles:
            if terrain[y, x] == player and 1 < military_map[y, x] > used_troops[y, x]:
                neighbors = [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]
                for ny, nx in neighbors:
                    if 0 <= ny < C_N and 0 <= nx < C_N and terrain[ny, nx] != player:
                        score = bot.evaluate_attack(military_map, resource_map, terrain, player, y, x, ny, nx, players)
                        attack_candidates.append({
                            'origin_max_troops': military_map[y, x],
                            'origin': (y, x),
                            'target': (ny, nx),
                            'score': score
                        })

        attack_candidates.sort(key=lambda c: c['score'], reverse=True)

        for candidate in attack_candidates:
            origin = candidate['origin']
            target = candidate['target']

            if (origin, target) in attempted_attacks:
                continue

            available = military_map[origin] - used_troops[origin]
            max_troops = min(available, players[player]['stocks'] // troop_move_price)

            if ((max_troops == available) or max_troops > military_map[target] + 1) and max_troops > 0:
                candidate['prob'] = utils.score_to_probability(candidate['score'])
                if random.random() < candidate['prob']:
                    selected = origin

                    if select_troop(True):
                        selected = target
                        if create_movement(max_troops):
                            used_troops[origin] += max_troops
                            made_action = True
                            print(f"Bot {player} moved {max_troops} troops from {origin} to {target} "
                                  f"(score {candidate['score']}, prob {candidate['prob']:.2f}), max_troops: {candidate['origin_max_troops']}")

                    attempted_attacks.add((origin, target))

                selected_troop = None
                selected = None

        # 3 - Expansion
        expand_candidates = []
        for y, x in tiles:
            expand_candidates.append({
                'from': (y, x),
                'score': bot.evaluate_expansion(terrain, resource_map, y, x, C_N, player, players)
            })

        if expand_candidates and players[player]['stocks'] >= expand_price:
            best_exp = max(expand_candidates, key=lambda e: e['score'])
            prob = utils.score_to_probability(best_exp['score'], k=2)
            if random.random() < prob:
                selected = best_exp['from']
                if expand_territory():
                    made_action = True
                    print(f"Bot {player} expanded from {best_exp['from']} (score {best_exp['score']}, prob {prob:.2f})")

        # 4 - Disbanding troops, to avoid useless wasting of money
        best_tile = None
        best_score = -float('inf')
        for y, x in tiles:
            if terrain[y, x] == player:
                score = bot.evaluate_disband(military_map, resource_map, terrain, player, y, x, players)
                if score > best_score:
                    best_score = score
                    best_tile = (y, x)

        if best_tile and best_score > 0:
            prob = utils.score_to_probability(best_score)

            if random.random() < prob:
                selected = best_tile
                if disband_troop():
                    made_action = True
                    print(
                        f"Bot {player} disbanded troop at {best_tile} "
                        f"(score {best_score:.2f}, prob {prob:.2f})"
                    )

            selected = None

    bot_hold = False

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            pygame.quit()
            break

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_n:
                vis_mode = 1 if vis_mode != 1 else 0
            elif event.key == pygame.K_m:
                vis_mode = 2 if vis_mode != 2 else 0

            elif event.key == pygame.K_k:
                overlay = resource_map if overlay is not resource_map else None
            elif event.key == pygame.K_l:
                overlay = military_map if overlay is not military_map else None

        if not playing[player_turn] in (1, 2):
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_e:
                    expand_territory()
                elif event.key == pygame.K_r:
                    train_troop()
                elif event.key == pygame.K_b:
                    disband_troop()
                elif event.key == pygame.K_t:
                    if selected_troop == selected:
                        select_troop(False)
                    elif not selected_troop:
                        select_troop(True)
                    else:
                        create_movement()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    if next_turn_rect.collidepoint(event.pos):
                        next_turn()
                    elif expand_rect.collidepoint(event.pos):
                        expand_territory()
                    elif mil_actions_rect.collidepoint(event.pos):
                        mil_actions = not mil_actions
                    elif make_mil_rect.collidepoint(event.pos):
                        train_troop()
                    elif move_mil_rect.collidepoint(event.pos):
                        if selected_troop == selected:
                            select_troop(False)
                        elif not selected_troop:
                            select_troop(True)
                        else:
                            create_movement()
                    else:
                        selected = get_tile_from_click(event.pos)
                elif event.button == 3:
                    selected = get_tile_from_click(event.pos)
                    if selected_troop == selected:
                        select_troop(False)
                    elif not selected_troop:
                        select_troop(True)
                    else:
                        create_movement()

    current_player = player_order[player_turn]
    if playing[player_turn] in (1, 2):
        bot_turn(current_player)
        next_turn()


    draw_screen()
    pygame.display.flip()
    clock.tick(60)