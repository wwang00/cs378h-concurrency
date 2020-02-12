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

func hash(t *BST) uint64 {
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

func equals(a, b *BST) bool {
	var stackA Stack
	var stackB Stack
	currA := a.Root
	currB := b.Root
	var lastA *Node = nil
	var lastB *Node = nil
	for (currA != nil || stackA.Size > 0) || (currB != nil || stackB.Size > 0) {
		if lastA == nil {
			if currA == nil {
				lastA = stackA.pop()
				currA = lastA.R
			} else {
				stackA.push(currA)
				currA = currA.L
			}
		}
		if lastB == nil {
			if currB == nil {
				lastB = stackB.pop()
				currB = lastB.R
			} else {
				stackB.push(currB)
				currB = currB.L
			}
		}
		if lastA != nil && lastB != nil {
			if lastA.V != lastB.V {
				return false
			}
			lastA = nil
			lastB = nil
		}
	}
	return true
}
