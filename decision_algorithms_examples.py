from collections import deque
import heapq
from math import inf


def bfs_shortest_path(graph, start, target):
    """
    Shortest path in an unweighted graph.

    Time complexity: O(V + E)
    V = number of vertices, E = number of edges.
    """
    queue = deque([start])
    previous = {start: None}

    while queue:
        node = queue.popleft()
        if node == target:
            break

        for neighbor in graph[node]:
            if neighbor not in previous:
                previous[neighbor] = node
                queue.append(neighbor)

    return rebuild_path(previous, start, target)


def dijkstra(graph, start, target):
    """
    Shortest path in a weighted graph with no negative edge weights.

    Time complexity: O((V + E) log V) with a priority queue.
    V = number of vertices, E = number of edges.
    """
    distances = {node: inf for node in graph}
    previous = {node: None for node in graph}
    distances[start] = 0
    priority_queue = [(0, start)]

    while priority_queue:
        current_distance, node = heapq.heappop(priority_queue)
        if current_distance > distances[node]:
            continue
        if node == target:
            break

        for neighbor, weight in graph[node]:
            candidate = current_distance + weight
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                previous[neighbor] = node
                heapq.heappush(priority_queue, (candidate, neighbor))

    return distances[target], rebuild_path(previous, start, target)


def bellman_ford(nodes, edges, start):
    """
    Shortest paths in a weighted graph that may contain negative edges.

    Time complexity: O(V * E)
    V = number of vertices, E = number of edges.
    """
    distances = {node: inf for node in nodes}
    previous = {node: None for node in nodes}
    distances[start] = 0

    for _ in range(len(nodes) - 1):
        updated = False
        for source, target, weight in edges:
            if distances[source] == inf:
                continue
            candidate = distances[source] + weight
            if candidate < distances[target]:
                distances[target] = candidate
                previous[target] = source
                updated = True
        if not updated:
            break

    for source, target, weight in edges:
        if distances[source] != inf and distances[source] + weight < distances[target]:
            raise ValueError("negative cycle detected")

    return distances, previous


def zero_one_knapsack(items, capacity):
    """
    Pick each item at most once to maximize value without exceeding capacity.

    items is a list of (name, weight, value).

    Time complexity: O(N * W)
    N = number of items, W = knapsack capacity.
    """
    table = [[0] * (capacity + 1) for _ in range(len(items) + 1)]

    for row, (_, weight, value) in enumerate(items, start=1):
        for current_capacity in range(capacity + 1):
            without_item = table[row - 1][current_capacity]
            with_item = -inf
            if weight <= current_capacity:
                with_item = table[row - 1][current_capacity - weight] + value
            table[row][current_capacity] = max(without_item, with_item)

    chosen = []
    current_capacity = capacity
    for row in range(len(items), 0, -1):
        if table[row][current_capacity] != table[row - 1][current_capacity]:
            name, weight, value = items[row - 1]
            chosen.append((name, weight, value))
            current_capacity -= weight

    chosen.reverse()
    return table[-1][capacity], chosen, table


def rebuild_path(previous, start, target):
    if target not in previous or previous[target] is None and target != start:
        return []

    path = []
    current = target
    while current is not None:
        path.append(current)
        current = previous[current]
    path.reverse()

    return path if path and path[0] == start else []


def demo_bfs():
    graph = {
        "A": ["B", "C"],
        "B": ["A", "D", "E"],
        "C": ["A", "F"],
        "D": ["B"],
        "E": ["B", "F"],
        "F": ["C", "E"],
    }
    path = bfs_shortest_path(graph, "A", "F")
    print("BFS shortest path A -> F:", " -> ".join(path))
    print("Time complexity: O(V + E)")


def demo_dijkstra():
    graph = {
        "A": [("B", 5), ("C", 4)],
        "B": [("A", 5), ("D", 7), ("E", 2)],
        "C": [("A", 4), ("E", 10)],
        "D": [("B", 7), ("F", 6)],
        "E": [("B", 2), ("C", 10), ("F", 8)],
        "F": [("D", 6), ("E", 8)],
    }
    distance, path = dijkstra(graph, "A", "F")
    print("Dijkstra shortest path A -> F:", " -> ".join(path), "cost =", distance)
    print("Time complexity: O((V + E) log V)")


def demo_bellman_ford():
    nodes = ["A", "B", "C", "D"]
    edges = [
        ("A", "B", 4),
        ("A", "C", 5),
        ("B", "C", -2),
        ("B", "D", 6),
        ("C", "D", 3),
    ]
    distances, previous = bellman_ford(nodes, edges, "A")
    path = rebuild_path(previous, "A", "D")
    print("Bellman-Ford shortest path A -> D:", " -> ".join(path), "cost =", distances["D"])
    print("Time complexity: O(V * E)")


def demo_knapsack():
    items = [
        ("laptop", 3, 2000),
        ("camera", 1, 1500),
        ("book", 2, 600),
        ("jacket", 2, 1000),
    ]
    capacity = 4
    best_value, chosen, _ = zero_one_knapsack(items, capacity)
    total_weight = sum(weight for _, weight, _ in chosen)
    names = ", ".join(name for name, _, _ in chosen)

    print("0/1 knapsack best value:", best_value)
    print("Chosen items:", names)
    print("Total weight:", total_weight, "/", capacity)
    print("Time complexity: O(N * W)")


if __name__ == "__main__":
    demo_bfs()
    demo_dijkstra()
    demo_bellman_ford()
    demo_knapsack()
