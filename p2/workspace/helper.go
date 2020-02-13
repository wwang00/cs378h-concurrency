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
		treesByHash[h] = append(treesByHash[h], i)
	}
}

func processHashesParallelChan(t int, mapCh chan<- *HashPair, wg *sync.WaitGroup) {
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
		mapCh <- &HashPair{hash(&trees[i]), i}
	}
}

func processHashesParallelLock(t int, mutex *sync.Mutex, wg *sync.WaitGroup) {
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
		mutex.Lock()
		treesByHash[h] = append(treesByHash[h], i)
		mutex.Unlock()
	}
}

func insertHashes(mapCh <-chan *HashPair) {
	var hp *HashPair
	for {
		hp = <-mapCh
		if hp == nil {
			return
		}
		hash := hp.K
		treesByHash[hash] = append(treesByHash[hash], hp.V)
	}
}
