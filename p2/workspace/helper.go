package main

import (
	"sync"
)

func computeHashesSequential() {
	for i, tree := range trees {
		hashes[i] = hash(&tree)
	}
}

func computeHashesParallel(t int, wg *sync.WaitGroup) {
	defer wg.Done()

	start := t * items
	if start >= N {
		return
	}
	end := start + items
	if end > N {
		end = N
	}
	for i := start; i < end; i++ {
		hashes[i] = hash(&trees[i])
	}
}

func processHashesSequential() {
	for i, tree := range trees {
		h := hash(&tree)
		arr := treesByHash.Maps[0][h]
		arr = append(arr, i)
		treesByHash.Maps[0][h] = arr
	}
}

func processHashesParallelChan(t int, mapCh []chan *HashPair, wg *sync.WaitGroup) {
	defer wg.Done()

	start := t * items
	if start >= N {
		return
	}
	end := start + items
	if end > N {
		end = N
	}
	for i := start; i < end; i++ {
		h := hash(&trees[i])
		mapCh[h%uint64(dataWorkers)] <- &HashPair{h, i}
	}
}

func processHashesParallelLock(t int, wg *sync.WaitGroup) {
	defer wg.Done()

	start := t * items
	if start >= N {
		return
	}
	end := start + items
	if end > N {
		end = N
	}
	for i := start; i < end; i++ {
		h := hash(&trees[i])
		bucket := h % uint64(dataWorkers)
		treesByHash.Mutex.Lock()
		arr := treesByHash.Maps[bucket][h]
		arr = append(arr, i)
		treesByHash.Maps[bucket][h] = arr
		treesByHash.Mutex.Unlock()
	}
}

func compareTreesParallel(workBuf *BoundedBuffer, wg *sync.WaitGroup) {
	defer wg.Done()

	var compareWg sync.WaitGroup
	for {
		tp := workBuf.pop()
		if tp == nil {
			return
		}
		valCh := make(chan *int)
		compareWg.Add(1)
		go compareTwoTrees(tp, true, valCh, &compareWg)
		compareWg.Add(1)
		go compareTwoTrees(tp, false, valCh, &compareWg)
		compareWg.Wait()
	}
}

func compareTwoTrees(tp *TreePair, sendFirst bool, valCh chan *int, wg *sync.WaitGroup) {
	defer wg.Done()

	var stack Stack
	var other *int
	var curr *Node
	if sendFirst {
		curr = trees[tp.A].Root
	} else {
		curr = trees[tp.B].Root
	}
	for curr != nil || stack.Size > 0 {
		if curr == nil {
			curr = stack.pop()
			if sendFirst {
				valCh <- &curr.V
				other = <-valCh
			} else {
				other = <-valCh
				valCh <- &curr.V
			}
			if other == nil || *other != curr.V {
				return
			}
			curr = curr.R
		} else {
			stack.push(curr)
			curr = curr.L
		}
	}
	if sendFirst {
		valCh <- nil
		other = <-valCh
	} else {
		other = <-valCh
		valCh <- nil
		if other == nil {
			treesEqual[tp.A][tp.B] = true
			treesEqual[tp.B][tp.A] = true
		}
	}
}

func insertHashes(mapCh chan *HashPair, t int, wg *sync.WaitGroup) {
	defer wg.Done()

	for {
		hp := <-mapCh
		if hp == nil {
			return
		}
		h := hp.K
		arr := treesByHash.Maps[t][h]
		arr = append(arr, hp.V)
		treesByHash.Maps[t][h] = arr
	}
}
