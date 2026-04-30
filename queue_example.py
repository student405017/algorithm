from collections import deque


class Queue:
    def __init__(self):
        self.items = deque()

    def is_empty(self):
        return len(self.items) == 0

    def enqueue(self, item):
        self.items.append(item)

    def dequeue(self):
        if self.is_empty():
            return None
        return self.items.popleft()

    def peek(self):
        if self.is_empty():
            return None
        return self.items[0]

    def size(self):
        return len(self.items)

    def show(self):
        print("Queue:", list(self.items))


if __name__ == "__main__":
    queue = Queue()

    queue.enqueue("A")
    queue.enqueue("B")
    queue.enqueue("C")
    queue.show()

    print("Front item:", queue.peek())
    print("Queue size:", queue.size())

    print("Removed item:", queue.dequeue())
    queue.show()

    print("Removed item:", queue.dequeue())
    print("Removed item:", queue.dequeue())
    print("Queue is empty:", queue.is_empty())
