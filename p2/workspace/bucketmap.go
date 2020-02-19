package main

import "sync"

// BucketMap : list of map from tree hash to list of tree ID
type BucketMap struct {
	Mutex sync.Mutex
	Maps  []map[uint64][]int
}

func (bm *BucketMap) init(b int) {
	for i := 0; i < b; i++ {
		bm.Maps = append(bm.Maps, make(map[uint64][]int))
	}
}
