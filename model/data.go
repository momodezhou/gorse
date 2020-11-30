// Copyright 2020 gorse Project Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package model

import (
	"bufio"
	"github.com/zhenghaoz/gorse/base"
	"github.com/zhenghaoz/gorse/storage"
	"log"
	"os"
	"strings"
)

const batchSize = 1000

// DataSet contains preprocessed data structures for recommendation models.
type DataSet struct {
	UserIndex     base.Index
	ItemIndex     base.Index
	FeedbackUsers []int
	FeedbackItems []int
	UserFeedback  [][]int
	ItemFeedback  [][]int
	Negatives     [][]int
}

// NewMapIndexDataset creates a data set.
func NewMapIndexDataset() *DataSet {
	set := new(DataSet)
	// Create index
	set.UserIndex = base.NewMapIndex()
	set.ItemIndex = base.NewMapIndex()
	// Initialize slices
	set.FeedbackUsers = make([]int, 0)
	set.FeedbackItems = make([]int, 0)
	set.UserFeedback = make([][]int, 0)
	set.ItemFeedback = make([][]int, 0)
	return set
}

func NewDirectIndexDataset() *DataSet {
	dataset := new(DataSet)
	// Create index
	dataset.UserIndex = base.NewDirectIndex()
	dataset.ItemIndex = base.NewDirectIndex()
	// Initialize slices
	dataset.FeedbackUsers = make([]int, 0)
	dataset.FeedbackItems = make([]int, 0)
	dataset.UserFeedback = make([][]int, 0)
	dataset.ItemFeedback = make([][]int, 0)
	dataset.Negatives = make([][]int, 0)
	return dataset
}

func (dataset *DataSet) AddUser(userId string) {
	dataset.UserIndex.Add(userId)
	userIndex := dataset.UserIndex.ToNumber(userId)
	for userIndex >= len(dataset.UserFeedback) {
		dataset.UserFeedback = append(dataset.UserFeedback, make([]int, 0))
	}
}

func (dataset *DataSet) AddItem(itemId string) {
	dataset.ItemIndex.Add(itemId)
	itemIndex := dataset.ItemIndex.ToNumber(itemId)
	for itemIndex >= len(dataset.ItemFeedback) {
		dataset.ItemFeedback = append(dataset.ItemFeedback, make([]int, 0))
	}
}

func (dataset *DataSet) AddFeedback(userId, itemId string, insertUserItem bool) {
	if insertUserItem {
		dataset.UserIndex.Add(userId)
	}
	if insertUserItem {
		dataset.ItemIndex.Add(itemId)
	}
	userIndex := dataset.UserIndex.ToNumber(userId)
	itemIndex := dataset.ItemIndex.ToNumber(itemId)
	if userIndex != base.NotId && itemIndex != base.NotId {
		dataset.FeedbackUsers = append(dataset.FeedbackUsers, userIndex)
		dataset.FeedbackItems = append(dataset.FeedbackItems, itemIndex)
		for itemIndex >= len(dataset.ItemFeedback) {
			dataset.ItemFeedback = append(dataset.ItemFeedback, make([]int, 0))
		}
		dataset.ItemFeedback[itemIndex] = append(dataset.ItemFeedback[itemIndex], userIndex)
		for userIndex >= len(dataset.UserFeedback) {
			dataset.UserFeedback = append(dataset.UserFeedback, make([]int, 0))
		}
		dataset.UserFeedback[userIndex] = append(dataset.UserFeedback[userIndex], itemIndex)
	}
}

func (dataset *DataSet) SetNegatives(userId string, negatives []string) {
	userIndex := dataset.UserIndex.ToNumber(userId)
	if userIndex != base.NotId {
		for userIndex >= len(dataset.Negatives) {
			dataset.Negatives = append(dataset.Negatives, make([]int, 0))
		}
		dataset.Negatives[userIndex] = make([]int, 0, len(negatives))
		for _, itemId := range negatives {
			itemIndex := dataset.ItemIndex.ToNumber(itemId)
			if itemIndex != base.NotId {
				dataset.Negatives[userIndex] = append(dataset.Negatives[userIndex], itemIndex)
			}
		}
	}
}

func (dataset *DataSet) Count() int {
	return len(dataset.FeedbackUsers)
}

// UserCount returns the number of UserFeedback.
func (dataset *DataSet) UserCount() int {
	return dataset.UserIndex.Len()
}

// ItemCount returns the number of ItemFeedback.
func (dataset *DataSet) ItemCount() int {
	return dataset.ItemIndex.Len()
}

func createSliceOfSlice(n int) [][]int {
	x := make([][]int, n)
	for i := range x {
		x[i] = make([]int, 0)
	}
	return x
}

func (dataset *DataSet) Split(numTestUsers int, seed int64) (*DataSet, *DataSet) {
	trainSet, testSet := new(DataSet), new(DataSet)
	trainSet.UserIndex, testSet.UserIndex = dataset.UserIndex, dataset.UserIndex
	trainSet.ItemIndex, testSet.ItemIndex = dataset.ItemIndex, dataset.ItemIndex
	trainSet.UserFeedback, testSet.UserFeedback = createSliceOfSlice(dataset.UserCount()), createSliceOfSlice(dataset.UserCount())
	trainSet.ItemFeedback, testSet.ItemFeedback = createSliceOfSlice(dataset.ItemCount()), createSliceOfSlice(dataset.ItemCount())
	rng := base.NewRandomGenerator(seed)
	if numTestUsers == dataset.UserCount() {
		for userIndex := 0; userIndex < dataset.UserCount(); userIndex++ {
			k := rng.Intn(len(dataset.UserFeedback[userIndex]))
			testSet.FeedbackUsers = append(testSet.FeedbackUsers, userIndex)
			testSet.FeedbackItems = append(testSet.FeedbackItems, dataset.UserFeedback[userIndex][k])
			testSet.UserFeedback[userIndex] = append(testSet.UserFeedback[userIndex], dataset.UserFeedback[userIndex][k])
			testSet.ItemFeedback[dataset.UserFeedback[userIndex][k]] = append(testSet.ItemFeedback[dataset.UserFeedback[userIndex][k]], userIndex)
			for i, itemIndex := range dataset.UserFeedback[userIndex] {
				if i != k {
					trainSet.FeedbackUsers = append(trainSet.FeedbackUsers, userIndex)
					trainSet.FeedbackItems = append(trainSet.FeedbackItems, itemIndex)
					trainSet.UserFeedback[userIndex] = append(trainSet.UserFeedback[userIndex], itemIndex)
					trainSet.ItemFeedback[itemIndex] = append(trainSet.ItemFeedback[itemIndex], userIndex)
				}
			}
		}
	} else {
		testUsers := rng.Sample(dataset.UserCount(), numTestUsers)
		for _, userIndex := range testUsers {
			k := rng.Intn(len(dataset.UserFeedback[userIndex]))
			testSet.FeedbackUsers = append(testSet.FeedbackUsers, userIndex)
			testSet.FeedbackItems = append(testSet.FeedbackItems, dataset.UserFeedback[userIndex][k])
			testSet.UserFeedback[userIndex] = append(testSet.UserFeedback[userIndex], dataset.UserFeedback[userIndex][k])
			testSet.ItemFeedback[dataset.UserFeedback[userIndex][k]] = append(testSet.ItemFeedback[dataset.UserFeedback[userIndex][k]], userIndex)
			for i, itemIndex := range dataset.UserFeedback[userIndex] {
				if i != k {
					trainSet.FeedbackUsers = append(trainSet.FeedbackUsers, userIndex)
					trainSet.FeedbackItems = append(trainSet.FeedbackItems, itemIndex)
					trainSet.UserFeedback[userIndex] = append(trainSet.UserFeedback[userIndex], itemIndex)
					trainSet.ItemFeedback[itemIndex] = append(trainSet.ItemFeedback[itemIndex], userIndex)
				}
			}
		}
	}
	return trainSet, testSet
}

// GetID the i-th record by <user ID, item ID, rating>.
func (dataset *DataSet) GetID(i int) (string, string) {
	userIndex, itemIndex := dataset.GetIndex(i)
	return dataset.UserIndex.ToName(userIndex), dataset.ItemIndex.ToName(itemIndex)
}

// GetIndex gets the i-th record by <user index, item index, rating>.
func (dataset *DataSet) GetIndex(i int) (int, int) {
	return dataset.FeedbackUsers[i], dataset.FeedbackItems[i]
}

// LoadDataFromCSV loads Data from a CSV file. The CSV file should be:
//   [optional header]
//   <userId 1> <sep> <itemId 1> <sep> <rating 1> <sep> <extras>
//   <userId 2> <sep> <itemId 2> <sep> <rating 2> <sep> <extras>
//   <userId 3> <sep> <itemId 3> <sep> <rating 3> <sep> <extras>
//   ...
// For example, the `u.Data` from MovieLens 100K is:
//  196\t242\t3\t881250949
//  186\t302\t3\t891717742
//  22\t377\t1\t878887116
func LoadDataFromCSV(fileName string, sep string, hasHeader bool) *DataSet {
	dataset := NewMapIndexDataset()
	// Open file
	file, err := os.Open(fileName)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	// Read CSV file
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		// Ignore header
		if hasHeader {
			hasHeader = false
			continue
		}
		fields := strings.Split(line, sep)
		// Ignore empty line
		if len(fields) < 2 {
			continue
		}
		dataset.AddFeedback(fields[0], fields[1], true)
	}
	return dataset
}

func LoadDataFromDatabase(database storage.Database) (*DataSet, error) {
	dataset := NewMapIndexDataset()
	cursor := ""
	var err error
	// pull users
	for {
		var users []storage.User
		cursor, users, err = database.GetUsers(cursor, batchSize)
		if err != nil {
			return nil, err
		}
		for _, user := range users {
			dataset.AddUser(user.UserId)
		}
		if cursor == "" {
			break
		}
	}
	// pull items
	for {
		var items []storage.Item
		cursor, items, err = database.GetItems(cursor, batchSize)
		if err != nil {
			return nil, err
		}
		for _, item := range items {
			dataset.AddItem(item.ItemId)
		}
		if cursor == "" {
			break
		}
	}
	// pull database
	for {
		var feedback []storage.Feedback
		cursor, feedback, err = database.GetFeedback(cursor, batchSize)
		if err != nil {
			return nil, err
		}
		for _, v := range feedback {
			dataset.AddFeedback(v.UserId, v.ItemId, false)
		}
		if cursor == "" {
			break
		}
	}
	return dataset, nil
}
