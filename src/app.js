import { NaiveBayes } from './naiveBayes.js'
import fs from 'fs'
import CsvParser from 'csv-parser'

const naiveBayes = new NaiveBayes()

// Read Data

const results = []

await fs.createReadStream('iris.csv') // banknote_authentication.csv iris.csv
  .pipe(CsvParser())
  .on('data', (data) => results.push(data))
  .on('end', () => {
    formatCsvData(results)
  })

/**
 * Formats CSV data to labels array and data array (output).
 *
 * @param {object} data - Data from CSV file.
 */
const formatCsvData = (data) => {
  const output = []
  const labelsObj = {}
  const labelsArray = []

  for (let i = 0; i < data.length; i++) { // Iterate all csv rows
    const keys = Object.keys(data[i])
    const numOfKeys = Object.keys(data[i]).length

    const formatedData = []

    for (let j = 0; j < numOfKeys; j++) { // Iterate all keys
      if (j < numOfKeys - 1) { // Attribute value
        formatedData.push(parseFloat(data[i][keys[j]]))
      } else { // Class/label
        const label = data[i][keys[j]]

        // Add label and value if it does not exist
        if (!labelsObj[label]) {
          const numOfLabels = labelsArray.length === 0 ? 1 : Object.keys(labelsObj).length + 1
          labelsObj[label] = numOfLabels
        }

        labelsArray.push(labelsObj[label])
      }
    }
    output.push(formatedData)
  }
  runNaiveBayes(output, labelsArray)
}

/**
 * Run Naïve Bayes methods.
 *
 * @param {Array} data - Data to train and predict.
 * @param {Array} labelsArray - Category labels.
 */
const runNaiveBayes = (data, labelsArray) => {
  naiveBayes.fit(data, labelsArray)

  const preds = naiveBayes.predict(data)

  const accuracyScore = naiveBayes.accuracyScore(preds, labelsArray)

  console.log('Accuracy: ' + (accuracyScore * 100).toFixed(2) + '%')
}
