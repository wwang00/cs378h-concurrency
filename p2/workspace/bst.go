package main

// Node : node in BST
type Node struct {
	V    int
	L, R *Node
}

// BST : binary search tree
type BST struct {
	Root *Node
}

// insert a value v not already in t
func (t *BST) insert(v int) {
	if t.Root == nil {
		t.Root = &Node{v, nil, nil}
		return
	}
	curr := t.Root
	for {
		if v < curr.V {
			if curr.L == nil {
				curr.L = &Node{v, nil, nil}
				return
			}
			curr = curr.L
		} else if v > curr.V {
			if curr.R == nil {
				curr.R = &Node{v, nil, nil}
				return
			}
			curr = curr.R
		} else {
			return
		}
	}
}

func (t *BST) hash() uint64 {
	var hash uint64 = 1
	var stack Stack
	curr := t.Root
	for curr != nil || stack.Size > 0 {
		if curr == nil {
			curr = stack.pop()
			val := uint64(curr.V) + 2
			hash = (hash*val + val) % 4222234741
			curr = curr.R
		} else {
			stack.push(curr)
			curr = curr.L
		}
	}
	return hash
}
