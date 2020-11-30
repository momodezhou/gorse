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
	"fmt"
	"github.com/olekukonko/tablewriter"
	log "github.com/sirupsen/logrus"
	"github.com/spf13/cobra"
	"github.com/zhenghaoz/gorse/config"
	"github.com/zhenghaoz/gorse/model"
	"github.com/zhenghaoz/gorse/storage"
	"os"
	"time"
)

/* Models */

/* Flags for parameters */

const (
	intFlag     = 0
	float64Flag = 1
)

type paramFlag struct {
	Type int
	Key  model.ParamName
	Name string
	Help string
}

var paramFlags = []paramFlag{
	{float64Flag, model.Lr, "lr", "Learning rate"},
	{float64Flag, model.Reg, "reg", "Regularization strength"},
	{intFlag, model.NEpochs, "set-n-epochs", "Number of epochs"},
	{intFlag, model.NFactors, "n-factors", "Number of factors"},
	{float64Flag, model.InitMean, "init-mean", "Mean of gaussian initial parameters"},
	{float64Flag, model.InitStdDev, "init-std", "Standard deviation of gaussian initial parameters"},
	{float64Flag, model.Weight, "weight", "Weight of negative samples in ALS."},
}

func test(cmd *cobra.Command, args []string) {
	modelName := args[0]
	m := model.NewModel(modelName, nil)
	// Load data
	var trainSet, testSet *model.DataSet
	if cmd.PersistentFlags().Changed("load-builtin") {
		name, _ := cmd.PersistentFlags().GetString("load-builtin")
		log.Infof("Load built-in dataset %s\n", name)
		trainSet, testSet = model.LoadDataFromBuiltIn(name)
	} else if cmd.PersistentFlags().Changed("load-csv") {
		name, _ := cmd.PersistentFlags().GetString("load-csv")
		sep, _ := cmd.PersistentFlags().GetString("csv-sep")
		header, _ := cmd.PersistentFlags().GetBool("csv-header")
		numTestUsers, _ := cmd.PersistentFlags().GetInt("n-test-users")
		seed, _ := cmd.PersistentFlags().GetInt("random-state")
		log.Infof("Load csv file %v", name)
		data := model.LoadDataFromCSV(name, sep, header)
		trainSet, testSet = data.Split(numTestUsers, int64(seed))
	} else {

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
		log.Infof("Load database %v", cfg.Database.Path)
		data, err := model.LoadDataFromDatabase(database)
		if err != nil {
			log.Fatal(err)
		}
		numTestUsers, _ := cmd.PersistentFlags().GetInt("n-test-users")
		seed, _ := cmd.PersistentFlags().GetInt("random-state")
		trainSet, testSet = data.Split(numTestUsers, int64(seed))
	}
	// Load hyper-parameters
	params := make(model.Params)
	for _, paramFlag := range paramFlags {
		if cmd.PersistentFlags().Changed(paramFlag.Name) {
			switch paramFlag.Type {
			case intFlag:
				value, _ := cmd.PersistentFlags().GetInt(paramFlag.Name)
				params[paramFlag.Key] = value
			case float64Flag:
				value, _ := cmd.PersistentFlags().GetFloat64(paramFlag.Name)
				params[paramFlag.Key] = value
			}
		}
	}
	log.Printf("Load hyper-parameters %v\n", params)
	m.SetParams(params)
	// Load runtime options
	fitConfig := &config.FitConfig{}
	fitConfig.Verbose, _ = cmd.PersistentFlags().GetInt("verbose")
	fitConfig.Jobs, _ = cmd.PersistentFlags().GetInt("jobs")
	fitConfig.TopK, _ = cmd.PersistentFlags().GetInt("top-k")
	fitConfig.Candidates, _ = cmd.PersistentFlags().GetInt("n-negatives")
	// Cross validation
	start := time.Now()
	score := m.Fit(trainSet, testSet, fitConfig)
	elapsed := time.Since(start)
	// Render table
	table := tablewriter.NewWriter(os.Stdout)
	table.SetHeader([]string{"NDCG@10", "Precision@10", "Recall@10"})
	table.Append([]string{
		fmt.Sprintf("%v", score.NDCG),
		fmt.Sprintf("%v", score.Precision),
		fmt.Sprintf("%v", score.Recall),
	})
	table.Render()
	log.Printf("Complete cross validation (%v)\n", elapsed)
}
