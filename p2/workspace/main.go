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
var trees []BST
var hashes []uint64
var treesByHash map[uint64][]int
var treesEqual [][]bool

func main() {
	N = 0
	// parse args
	hashWorkersFlag := flag.Int("hash-workers", 0, "num threads calculating hash")
	dataWorkersFlag := flag.Int("data-workers", 0, "num threads modifying hash map")
	compWorkersFlag := flag.Int("comp-workers", 0, "num threads comparing trees")
	fin := flag.String("input", "", "input file")
	flag.Parse()
	hashWorkers := *hashWorkersFlag
	dataWorkers := *dataWorkersFlag
	compWorkers := *compWorkersFlag

	// read input and build trees
	file, _ := os.Open(*fin)
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.Split(scanner.Text(), " ")
		tree := BST{nil}
		for _, elem := range line {
			val, _ := strconv.Atoi(elem)
			tree.insert(val)
		}
		trees = append(trees, tree)
		N++
	}

	// compute hashes
	hashes = make([]uint64, N)

	startHashTime := time.Now()

	if hashWorkers > 1 {
		items := N / hashWorkers
		if items*hashWorkers != N {
			items++
		}
		var wg sync.WaitGroup
		for t := 0; t < hashWorkers; t++ {
			start := t * items
			if start >= N {
				continue
			}
			end := start + items
			if end > N {
				end = N
			}
			wg.Add(1)
			go computeHashes(start, end, &wg)
			wg.Wait()
		}
	} else {
		computeHashes(0, N, nil)
	}

	fmt.Printf("hashTime: %d\n", time.Since(startHashTime).Microseconds())

	// make hash groups
	treesByHash = make(map[uint64][]int)

	startGroupTime := time.Now()

	if dataWorkers > 1 {

	} else {
		for i, hash := range hashes {
			treesByHash[hash] = append(treesByHash[hash], i)
		}
	}

	fmt.Printf("hashGroupTime: %d\n", time.Since(startGroupTime).Microseconds())

	for hash, ids := range treesByHash {
		fmt.Printf("%d:", hash)
		for _, id := range ids {
			fmt.Printf(" %d", id)
		}
		fmt.Println()
	}

	// compare within groups
	treesEqual = make([][]bool, N)
	for i := range treesEqual {
		treesEqual[i] = make([]bool, N)
	}

	startCompareTime := time.Now()

	if compWorkers > 1 {

	} else {
		for _, ids := range treesByHash {
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
