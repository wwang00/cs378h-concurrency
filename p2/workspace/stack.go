package main

import "log"

// Stack : LIFO stack for Nodes
type Stack struct {
	V    []*Node
	Size int
}

func (s *Stack) push(v *Node) {
	if s.Size == len(s.V) {
		s.V = append(s.V, v)
	} else {
		s.V[s.Size] = v
	}
	s.Size++
}

func (s *Stack) pop() *Node {
	if s.Size == 0 {
		log.Fatalln("popped empty stack")
	}
	s.Size--
	return s.V[s.Size]
}
