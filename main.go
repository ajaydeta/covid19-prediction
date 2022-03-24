package main

import (
	"bytes"
	"fmt"
	"log"
	"os"

	"github.com/go-gota/gota/dataframe"
	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/evaluation"
	"github.com/sjwhitworth/golearn/knn"
)

func main() {
	csvfile, err := os.Open("datasheet/covid_19_clean_complete.csv")
	if err != nil {
		log.Fatal(err)
	}
	df := dataframe.ReadCSV(csvfile, dataframe.HasHeader(true))
	selected := df.Select([]string{"Confirmed", "Deaths", "Recovered"})
	var by bytes.Buffer
	selected.WriteCSV(&by, dataframe.WriteHeader(true))

	rawData, err := base.ParseCSVToInstancesFromReader(bytes.NewReader(by.Bytes()), true)
	if err != nil {
		panic(err)
	}
	fmt.Println(rawData)

	cls := knn.NewKnnClassifier("euclidean", "linear", 5)

	trainData, testData := base.InstancesTrainTestSplit(rawData, 0.7)
	cls.Fit(trainData)

	predictions, err := cls.Predict(testData)
	if err != nil {
		panic(err)
	}

	confusionMat, err := evaluation.GetConfusionMatrix(testData, predictions)
	if err != nil {
		panic(fmt.Sprintf("Unable to get confusion matrix: %s", err.Error()))
	}
	fmt.Println(evaluation.GetSummary(confusionMat))
}
