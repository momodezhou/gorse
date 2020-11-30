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
package main

import (
	log "github.com/sirupsen/logrus"
	"github.com/spf13/cobra"
	"github.com/zhenghaoz/gorse/config"
	"github.com/zhenghaoz/gorse/model"
	"github.com/zhenghaoz/gorse/storage"
)

func tune(cmd *cobra.Command, args []string) {
	modelName := args[0]
	var m model.Model
	var exist bool
	if m, exist = models[modelName]; !exist {
		log.Fatalf("Unknown m %s\n", modelName)
	}
	// Load data
	var trainSet, testSet *model.DataSet
	if cmd.PersistentFlags().Changed("load-builtin") {
		name, _ := cmd.PersistentFlags().GetString("load-builtin")
		trainSet, testSet = model.LoadDataFromBuiltIn(name)
		log.Printf("Load built-in dataset %s\n", name)
	} else if cmd.PersistentFlags().Changed("load-csv") {
		name, _ := cmd.PersistentFlags().GetString("load-csv")
		sep, _ := cmd.PersistentFlags().GetString("csv-sep")
		header, _ := cmd.PersistentFlags().GetBool("csv-header")
		numTestUsers, _ := cmd.PersistentFlags().GetInt("n-test-users")
		seed, _ := cmd.PersistentFlags().GetInt("random-state")
		data := model.LoadDataFromCSV(name, sep, header)
		trainSet, testSet = data.Split(numTestUsers, int64(seed))
	} else {
		log.Println("Load default dataset ml-100k")
		// Load config
		cfg, _, err := config.LoadConfig("/home/zhenghaoz/.gorse/cli.toml")
		if err != nil {
			log.Fatal(err)
		}
		// Open database
		database, err := storage.Open(cfg.Database.Path)
		if err != nil {
			log.Fatal(err)
		}
		defer database.Close()
		// Load data
		data, err := model.LoadDataFromDatabase(database)
		if err != nil {
			log.Fatal(err)
		}
		numTestUsers, _ := cmd.PersistentFlags().GetInt("n-test-users")
		seed, _ := cmd.PersistentFlags().GetInt("random-state")
		trainSet, testSet = data.Split(numTestUsers, int64(seed))
	}
	_, _ = trainSet, testSet
	_ = m
}
