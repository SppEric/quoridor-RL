import numpy as np
from collections import deque

# Global variables
selfn = 0
selfn_ = 0
multi = 0
num_walls = 0

def setup(size):
    global selfn, selfn_, num_walls
    selfn_ = size - 1
    selfn = 2 * size - 1
    num_walls = 2 * selfn_ * selfn_

def get_item(arr, x, y):
    return arr[x, y]

def print_2d(arr):
    for row in arr:
        print(' '.join(map(str, row)))

def possible_walls(block):
    available = np.zeros(128, dtype=bool)
    counter = 0

    # Horizontal walls
    for x in range(1, selfn, 2):
        for y in range(0, selfn - 2, 2):
            conditions = [
                not get_item(block, x, y),
                not get_item(block, x, y + 2),
                (not get_item(block, x - 1, y + 1) or not get_item(block, x + 1, y + 1)) or (
                    x - 3 >= 0 and x + 3 < selfn and
                    get_item(block, x - 1, y + 1) and get_item(block, x - 3, y + 1) and
                    get_item(block, x + 1, y + 1) and get_item(block, x + 3, y + 1))
            ]
            if all(conditions):
                available[counter] = True
            counter += 1

    # Vertical walls
    for x in range(2, selfn, 2):
        for y in range(1, selfn, 2):
            conditions = [
                not get_item(block, x, y),
                not get_item(block, x - 2, y),
                (not get_item(block, x - 1, y - 1) or not get_item(block, x - 1, y + 1)) or (
                    y - 3 >= 3 and y + 3 < selfn and
                    get_item(block, x - 1, y - 1) and get_item(block, x - 1, y - 3) and
                    get_item(block, x - 1, y + 1) and get_item(block, x - 1, y + 3))
            ]
            if all(conditions):
                available[counter] = True
            counter += 1

    return available

def clean_next_visits(queue, x, y):
    queue.clear()
    queue.append((x, y))

def set_item3(arr, x, y, val1, val2):
    index = (x * selfn + y) * 2
    arr[index] = val1
    arr[index + 1] = val2

def add_next_visits(queue, x, y, x_target, block, visited, parent):
    visited[x, y] = 1

    directions = [
        (-2, 0), (2, 0), (0, -2), (0, 2)
    ]
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < selfn and 0 <= ny < selfn and not get_item(block, x + dx // 2, y + dy // 2) and not get_item(visited, nx, ny):
            queue.append((nx, ny))
            set_item3(parent, nx, ny, x, y)
            visited[nx, ny] = 1
            if nx == x_target:
                clean_next_visits(queue, nx, ny)
                return

def bitset_to_pylist(data):
    return list(data)

def index_of_action(a, x, y):
    if a == 8:
        return (x // 2) * selfn_ + (y // 2) + 1
    if a == 9:
        return (selfn_ * selfn_) + (y + 1) // 2 + ((x // 2) - 1) * selfn_

def walls_not_in_path(x, y, parent):
    not_in_path = np.ones(num_walls, dtype=bool)
    while parent[x, y, 0] != -1:
        x_, y_ = parent[x, y]
        if y == y_:
            ax = (x + x_) // 2
            if y < selfn - 1:
                not_in_path[index_of_action(8, ax, y) - 1] = False
            if y - 2 >= 0:
                not_in_path[index_of_action(8, ax, y - 2) - 1] = False
        if x == x_:
            ay = (y + y_) // 2
            if x >= 2:
                not_in_path[index_of_action(9, x, ay) - 1] = False
            if x + 2 <= selfn - 1:
                not_in_path[index_of_action(9, x + 2, ay) - 1] = False
        x, y = x_, y_
    return not_in_path

def find_path_and_non_blocking_walls(x, y, x_target, wall, block):
    non_blocking_walls = np.zeros(num_walls, dtype=bool)
    block = block.copy()
    wx, wy = divmod(wall, selfn_)
    if wall < (num_walls // 2):
        wx = wx * 2 + 1
        wy = wy * 2
        block[wx, wy + 2] = 1
    else:
        wx = (wx - (num_walls // 2) // selfn_) * 2 + 2
        wy = wy * 2 + 1
        block[wx - 2, wy] = 1
    block[wx, wy] = 1

    parent = np.full((selfn, selfn, 2), -1)
    visited = np.zeros((selfn, selfn), dtype=bool)
    queue = deque([(x, y)])
    parent[x, y] = (-1, -1)
    while queue:
        cx, cy = queue.popleft()
        if cx == x_target:
            return walls_not_in_path(cx, cy, parent)
        add_next_visits(queue, cx, cy, x_target, block, visited, parent)
    return non_blocking_walls

def legal_walls(buffer):
    block = np.zeros((selfn, selfn), dtype=int)
    x, y, x_, y_ = 0, 0, 0, 0
    pointer = 0
    for i in range(4):
        for j in range(selfn):
            for k in range(selfn):
                value = buffer[pointer]
                if i == 0 and value == 1:
                    x, y = j, k
                if i == 1 and value == 1:
                    x_, y_ = j, k
                if i == 2:
                    block[j, k] = value
                if i == 3:
                    block[j, k] += value
                pointer += 1

    all_walls = possible_walls(block)
    legal_walls = np.zeros(num_walls, dtype=bool)
    tried_walls = np.zeros(num_walls, dtype=bool)
    next_try = 0
    while next_try < num_walls:
        if next_try < num_walls and (not all_walls[next_try] or tried_walls[next_try]):
            next_try += 1
            continue
        if next_try >= num_walls:
            break
        trial_result = find_path_and_non_blocking_walls(x, y, 0, next_try, block) & \
                       find_path_and_non_blocking_walls(x_, y_, selfn - 1, next_try, block) & \
                       all_walls
        legal_walls |= trial_result
        tried_walls[next_try] = True
        next_try += 1
    return legal_walls

# Example usage of legalWalls:
buffer = np.random.randint(0, 2, selfn * selfn * 4)
lw = legal_walls(buffer)
print(lw)
# Example usage and setup
setup(5)  # Initialize with a size of 5
block = np.zeros((selfn, selfn), dtype=int)
print_2d(block)
available = possible_walls(block)
print(available)

# You can now import this file in other Python scripts and use the functions directly.
