from queue import PriorityQueue
from typing import Dict, List


class _Node(object):
    """node of huffman tree
    """

    def __init__(self, num: int, weight: int):
        self._num = num
        self._weight = weight
        self._left_child = None
        self._right_child = None

    def add_child(self, type: int, child):
        """add child for node

        Args:
            type: 0 if left child, 1 if right child
            child: node of child
        """
        if type == 0:
            self._left_child = child
        else:
            self._right_child = child

    def num(self) -> int:
        return self._num

    def left_child(self):
        return self._left_child

    def right_child(self):
        return self._right_child

    def weight(self) -> int:
        return self._weight

    def __lt__(self, o):
        if self.weight() <= o.weight():
            return True
        return False


class HuffmanTree(object):
    """huffman tree

    left child of every node following with 0
    right child of every node following with 1
    tree must have at least one node
    """

    def __init__(self, labels: Dict[int, int] = None):
        """initialize huffman tree

        Args:
            labels: maps from label to occurrence time
        """
        self._cnt = 0
        self._num: Dict[int, int] = {}
        self._path: Dict[int, List[int]] = {}

        if labels is None:
            return

        nodes = PriorityQueue()
        for x in labels:
            self._num[x] = self._cnt
            nodes.put(_Node(self._cnt, labels[x]))
            self._cnt += 1

        while nodes.qsize() >= 2:
            a = nodes.get()
            b = nodes.get()

            c = _Node(self._cnt, a.weight() + b.weight())
            self._cnt += 1

            c.add_child(0, a)
            c.add_child(1, b)

            nodes.put(c)

        self._root = nodes.get()
        self._DFS(-1, self._root)

    def _DFS(self, par: int, current: _Node):
        if current is None:
            return
        if par == -1:
            self._path[current.num()] = [current.num()]
        else:
            self._path[current.num()] = self._path[par] + [current.num()]

        self._DFS(current.num(), current.left_child())
        self._DFS(current.num(), current.right_child())

    def get(self, label: int) -> (List[int], List[int]):
        """get path of node in huffman tree

        Args:
            label: label of wanted node

        Returns:
            positive node in which node get left
            negative node in which node get right
        """
        pos_nodes = []
        neg_nodes = []
        current = self._root

        for i in range(1, len(self._path[self._num[label]])):
            x = self._path[self._num[label]][i]
            if current.left_child().num() == x:
                pos_nodes.append(self._path[self._num[label]][i-1])
                current = current.left_child()
            else:
                neg_nodes.append(self._path[self._num[label]][i-1])
                current = current.right_child()

        return pos_nodes, neg_nodes

    def dump(self, path: str):
        """dump parameters of huffman tree

        Args:
            path: path to dump parameters
        """
        with open(path, 'w') as f:
            f.write(str(self._root.num()) + '\n')
            f.write(str(self._cnt) + '\n')

            rec: List[str] = ["" for i in range(self._cnt)]
            stack: List[_Node] = [self._root]
            while len(stack) != 0:
                x = stack[-1]
                stack.pop()

                if x is None:
                    continue

                left_child = x.left_child()
                right_child = x.right_child()
                if left_child is None:
                    left_child = _Node(-1, 0)
                if right_child is None:
                    right_child = _Node(-1, 0)

                rec[x.num()] = "%d %d %d" % (
                    x.weight(), left_child.num(), right_child.num())
                stack.append(x.left_child())
                stack.append(x.right_child())
            for s in rec:
                f.write(s + '\n')

            for label in self._num:
                f.write("%s %d\n" % (label, self._num[label]))

    def load(self, path: str):
        """load parameters from given file

        Args:
            path: path of parameters
        """
        nodes = {}
        nodes[-1] = None
        childs = []
        with open(path, 'r') as f:
            line_count = 0
            for line in f:
                current = line.strip()

                if line_count == 0:
                    self._root = int(current)
                elif line_count == 1:
                    self._cnt = int(current)
                elif line_count <= self._cnt + 1:
                    param = current.split()
                    cnt = line_count - 2
                    nodes[cnt] = _Node(cnt, int(param[0]))
                    childs.append(
                        [cnt, int(param[1]), int(param[2])])
                else:
                    param = current.split()
                    self._num[int(param[0])] = int(param[1])
                line_count += 1

        for p in childs:
            par = nodes[p[0]]
            par.add_child(0, nodes[p[1]])
            par.add_child(1, nodes[p[2]])

    def __len__(self):
        return self._cnt
