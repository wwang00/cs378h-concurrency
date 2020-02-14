package main

// BucketMap : list of map[uint64][]int
type BucketMap struct {
	Maps []map[uint64][]int
}

func (bm *BucketMap) init(b int) {
	for i := 0; i < b; i++ {
		bm.Maps = append(bm.Maps, make(map[uint64][]int))
	}
}
