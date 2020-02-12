package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"
)

func main() {
	N := 0
	var trees []BST
	var hashes []uint64
	var treesByHash map[uint64][]int
	var treeGroups [][]bool
	// parse args
	// hashWorkers := flag.Int("hash-workers", 0, "num threads calculating hash")
	// dataWorkers := flag.Int("data-workers", 0, "num threads modifying hash map")
	// compWorkers := flag.Int("comp-workers", 0, "num threads comparing trees")
	fin := flag.String("input", "", "input file")
	flag.Parse()

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

	for i, tree := range trees {
		hashes[i] = hash(&tree)
	}

	fmt.Printf("hashTime: %d\n", time.Since(startHashTime).Microseconds())

	// make hash groups
	treesByHash = make(map[uint64][]int)

	startGroupTime := time.Now()

	for i := 0; i < N; i++ {
		hash := hashes[i]
		treesByHash[hash] = append(treesByHash[hash], i)
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
	treeGroups = make([][]bool, N)
	for i := range treeGroups {
		treeGroups[i] = make([]bool, N)
	}

	startCompareTime := time.Now()

	for _, ids := range treesByHash {
		for i := 0; i < len(ids)-1; i++ {
			for j := i + 1; j < len(ids); j++ {
				id1 := ids[i]
				id2 := ids[j]
				if equals(&trees[id1], &trees[id2]) {
					if id1 < id2 {
						treeGroups[id1][id2] = true
					} else {
						treeGroups[id2][id1] = true
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
			if treeGroups[i][j] {
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
