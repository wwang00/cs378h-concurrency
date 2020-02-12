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
		hashes[i] = tree.hash()
	}

	fmt.Println("hashTime: ", time.Since(startHashTime).Microseconds())

	// make groups
	treesByHash = make(map[uint64][]int)

	startGroupTime := time.Now()

	for i := 0; i < N; i++ {
		hash := hashes[i]
		treesByHash[hash] = append(treesByHash[hash], i)
	}

	fmt.Print("hashGroupTime: ", time.Since(startGroupTime).Microseconds())

	for hash, ids := range treesByHash {
		fmt.Print(hash, ":")
		for id := range ids {
			fmt.Print(" ", id)
		}
		fmt.Println()
	}
}
