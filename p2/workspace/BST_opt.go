package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"
)

// N : number of trees
var N int
var items int
var trees []BST
var hashes []uint64
var treesByHash BucketMap
var treesEqual [][]bool

var hashWorkers int
var dataWorkers int
var compWorkers int

func main() {
	// parse args
	hashWorkersFlag := flag.Int("hash-workers", 0, "num threads calculating hash")
	dataWorkersFlag := flag.Int("data-workers", 0, "num threads modifying hash map")
	compWorkersFlag := flag.Int("comp-workers", 0, "num threads comparing trees")
	fin := flag.String("input", "", "input file")
	flag.Parse()
	hashWorkers = *hashWorkersFlag
	dataWorkers = *dataWorkersFlag
	compWorkers = *compWorkersFlag

	// read input and build trees
	file, _ := os.Open(*fin)
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.Split(scanner.Text(), " ")
		var tree BST
		for _, elem := range line {
			val, _ := strconv.Atoi(elem)
			tree.insert(val)
		}
		trees = append(trees, tree)
		N++
	}

	// init data structures
	hashes = make([]uint64, N)
	treesByHash.init(dataWorkers)
	treesEqual = make([][]bool, N)
	for i := range treesEqual {
		treesEqual[i] = make([]bool, N)
	}

	// compute hashes
	startHashTime := time.Now()

	if hashWorkers == 1 {
		if dataWorkers == 0 {
			computeHashes()
			fmt.Printf("hashTime: %d\n", time.Since(startHashTime).Microseconds())
			return
		}
		processHashesSequential()
	} else {
		items = N / hashWorkers
		if items*hashWorkers != N {
			items++
		}
		var wg sync.WaitGroup
		if dataWorkers == hashWorkers {
			for t := 0; t < hashWorkers; t++ {
				wg.Add(1)
				go processHashesParallelLock(t, &wg)
			}
			wg.Wait()
		} else {
			var insertWg sync.WaitGroup
			mapChs := make([]chan *HashPair, dataWorkers)
			for t := 0; t < dataWorkers; t++ {
				mapCh := make(chan *HashPair)
				mapChs[t] = mapCh
				insertWg.Add(1)
				go insertHashes(mapCh, t, &insertWg)
			}
			for t := 0; t < hashWorkers; t++ {
				wg.Add(1)
				go processHashesParallelChan(t, mapChs, &wg)
			}
			wg.Wait()
			for t := 0; t < dataWorkers; t++ {
				mapChs[t] <- nil
			}
			insertWg.Wait()
		}
	}

	fmt.Printf("hashGroupTime: %d\n", time.Since(startHashTime).Microseconds())

	for _, bucket := range treesByHash.Maps {
		for hash, ids := range bucket {
			fmt.Printf("%d:", hash)
			for _, id := range ids {
				fmt.Printf(" %d", id)
			}
			fmt.Println()
		}
	}

	// compare within groups
	startCompareTime := time.Now()

	if compWorkers == 0 {
		return
	} else if compWorkers > 1 {
		var wg sync.WaitGroup
		var workBuf BoundedBuffer
		workBuf.init(compWorkers)
		for i := 0; i < compWorkers; i++ {
			wg.Add(1)
			go compareTreesParallel(&workBuf, &wg)
		}
		// send work
		// send nils
		wg.Wait()
	} else {
		for _, bucket := range treesByHash.Maps {
			for _, ids := range bucket {
				for i := 0; i < len(ids)-1; i++ {
					for j := i + 1; j < len(ids); j++ {
						id1 := ids[i]
						id2 := ids[j]
						if equals(&trees[id1], &trees[id2]) {
							treesEqual[id1][id2] = true
							treesEqual[id2][id1] = true
						}
					}
				}
			}
		}
	}

	fmt.Printf("compareTreeTime: %d\n", time.Since(startCompareTime).Microseconds())

	group := 0
	seen := make([]bool, N)
	for i := 0; i < N-1; i++ {
		if seen[i] {
			continue
		}
		found := false
		for j := i + 1; j < N; j++ {
			if treesEqual[i][j] {
				if !found {
					found = true
					seen[i] = true
					fmt.Printf("group %d: %d", group, i)
					group++
				}
				seen[j] = true
				fmt.Printf(" %d", j)
			}
		}
		if found {
			fmt.Println()
		}
	}
}
