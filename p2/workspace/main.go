package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"
)

func main() {
	N := 0
	var trees []BST
	var hashes []uint64
	// parse args
	// hashWorkers := flag.Int("hash-workers", 0, "num threads calculating hash")
	// dataWorkers := flag.Int("data-workers", 0, "num threads modifying hash map")
	// compWorkers := flag.Int("comp-workers", 0, "num threads comparing trees")
	fin := flag.String("input", "", "input file")
	flag.Parse()

	// build trees
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
	for i, tree := range trees {
		hashes[i] = tree.hash()
		fmt.Println(hashes[i])
	}
}
