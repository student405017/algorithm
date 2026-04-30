class Stack:
    def __init__(self):
        self.items = []

    def is_empty(self):
        return len(self.items) == 0

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if self.is_empty():
            return None
        return self.items.pop()

    def peek(self):
        if self.is_empty():
            return None
        return self.items[-1]

    def size(self):
        return len(self.items)

    def show(self):
        print("Stack:", self.items)


if __name__ == "__main__":
    stack = Stack()

    stack.push("A")
    stack.push("B")
    stack.push("C")
    stack.show()

    print("最上面的元素:", stack.peek())
    print("Stack 大小:", stack.size())

    print("取出元素:", stack.pop())
    stack.show()

    print("取出元素:", stack.pop())
    print("取出元素:", stack.pop())
    print("Stack 是否為空:", stack.is_empty())
