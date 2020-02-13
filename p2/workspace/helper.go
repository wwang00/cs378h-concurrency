package main

import (
	"sync"
)

func computeHashes(start int, end int, wg *sync.WaitGroup) {
	if wg != nil {
		defer wg.Done()
	}
	for i := start; i < end; i++ {
		hashes[i] = hash(&trees[i])
	}
}
