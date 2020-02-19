package main

import "sync"

// BoundedBuffer : blocking buffer for TreePairs
type BoundedBuffer struct {
	mutex           sync.Mutex
	empty           sync.Cond
	full            sync.Cond
	vals            []*TreePair
	cap, used, l, r int
}

func (bb *BoundedBuffer) init(cap int) {
	bb.empty = *sync.NewCond(&bb.mutex)
	bb.full = *sync.NewCond(&bb.mutex)
	bb.vals = make([]*TreePair, cap)
	bb.cap = cap
}

func (bb *BoundedBuffer) push(tp *TreePair) {
	defer bb.mutex.Unlock()

	bb.mutex.Lock()
	for bb.used == bb.cap {
		bb.full.Wait()
	}
	bb.vals[bb.r] = tp
	bb.r++
	bb.r %= bb.cap
	bb.used++
	bb.empty.Signal()
}

func (bb *BoundedBuffer) pop() *TreePair {
	defer bb.mutex.Unlock()

	bb.mutex.Lock()
	for bb.used == 0 {
		bb.empty.Wait()
	}
	result := bb.vals[bb.l]
	bb.l++
	bb.l %= bb.cap
	bb.used--
	bb.full.Signal()
	return result
}
