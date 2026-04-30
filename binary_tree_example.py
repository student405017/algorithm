from collections import deque


class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None


class BinaryTree:
    def __init__(self, root=None):
        self.root = root

    def preorder(self, node):
        if node is None:
            return []
        return [node.value] + self.preorder(node.left) + self.preorder(node.right)

    def inorder(self, node):
        if node is None:
            return []
        return self.inorder(node.left) + [node.value] + self.inorder(node.right)

    def postorder(self, node):
        if node is None:
            return []
        return self.postorder(node.left) + self.postorder(node.right) + [node.value]

    def level_order(self):
        if self.root is None:
            return []

        result = []
        queue = deque([self.root])

        while queue:
            current = queue.popleft()
            result.append(current.value)

            if current.left is not None:
                queue.append(current.left)
            if current.right is not None:
                queue.append(current.right)

        return result

    def height(self, node):
        if node is None:
            return 0
        left_height = self.height(node.left)
        right_height = self.height(node.right)
        return max(left_height, right_height) + 1


if __name__ == "__main__":
    # Tree structure:
    #
    #        A
    #       / \
    #      B   C
    #     / \   \
    #    D   E   F

    root = TreeNode("A")
    root.left = TreeNode("B")
    root.right = TreeNode("C")
    root.left.left = TreeNode("D")
    root.left.right = TreeNode("E")
    root.right.right = TreeNode("F")

    tree = BinaryTree(root)

    print("Preorder:", tree.preorder(tree.root))
    print("Inorder:", tree.inorder(tree.root))
    print("Postorder:", tree.postorder(tree.root))
    print("Level order:", tree.level_order())
    print("Tree height:", tree.height(tree.root))
