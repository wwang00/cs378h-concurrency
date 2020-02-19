package main

import (
	"sync"
)

func computeHashes() {
	for i, tree := range trees {
		hashes[i] = hash(&tree)
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

	for {
		tp := workBuf.pop()
		if tp == nil {
			return
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
		if h%uint64(dataWorkers) != uint64(t) {
			panic("data worker hash bucket mismatch")
		}
		arr := treesByHash.Maps[t][h]
		arr = append(arr, hp.V)
		treesByHash.Maps[t][h] = arr
	}
}
