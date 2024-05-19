/**
 * Class represents Naïve Bayes algorithm.
 */
export class NaiveBayes {
  #data

  /**
   * Train model on example input and labels.
   *
   * @param {Array} x - Array list of input examples.
   * @param {Array} y - Array with labels.
   */
  fit (x, y) {
    // Divide the dataset into each category
    this.#formatData(x, y)

    // Calculate the mean value of each attribute in each category
    this.#meanEachAttributeCategory()

    // Calculate standard deviation of each attribute in each category
    this.#standardDeviation()
  }

  /**
   * Create data object with categories and attribute values.
   *
   * @param {Array} attributeValues - Attribute values.
   * @param {Array} labels - Category labels.
   */
  #formatData (attributeValues, labels) {
    this.#data = {} // Init data object

    // Add object for each category
    let numOfUniqueLabels = 0
    for (let i = 0; i < labels.length; i++) {
      if (labels[i] > numOfUniqueLabels) {
        numOfUniqueLabels += 1
        this.#data[labels[i]] = {}
      }
    }

    const numOfAttributes = attributeValues[0].length

    // Add arrays for sorted attributes.
    for (let i = 1; i <= numOfUniqueLabels; i++) {
      this.#data[i].sortedAttributes = []

      for (let j = 0; j < numOfAttributes; j++) {
        this.#data[i].sortedAttributes.push([])
      }
    }

    // Add values of each attribute and category into separate arrays
    for (let i = 0; i < attributeValues.length; i++) {
      const category = labels[i]

      for (let j = 0; j < numOfAttributes; j++) {
        this.#data[category].sortedAttributes[j].push(attributeValues[i][j])
      }
    }
  }

  /**
   * Calculate mean for each attribute in each category.
   */
  #meanEachAttributeCategory () {
    for (const category in this.#data) {
      this.#data[category].mean = []

      const attributesValues = this.#data[category].sortedAttributes
      const numOfCategories = attributesValues.length

      // Iterate array with attribute value arrays
      for (let i = 0; i < numOfCategories; i++) {
        const attributeValues = attributesValues[i]

        let attributeSum = 0
        for (let j = 0; j < attributeValues.length; j++) {
          attributeSum += attributeValues[j]
        }

        const attributeMean = attributeSum / attributeValues.length

        this.#data[category].mean.push(attributeMean)
      }
    }
  }

  /**
   * Calculate standard deviation for each attribute in each category.
   */
  #standardDeviation () {
    for (const category in this.#data) {
      this.#data[category].stDev = []

      const attributes = this.#data[category].sortedAttributes

      const attributesMeanValue = this.#data[category].mean

      for (let i = 0; i < attributes.length; i++) {
        // Squared differences
        let squaredDifferences = 0
        for (let j = 0; j < attributes[i].length; j++) {
          squaredDifferences += Math.pow(attributes[i][j] - attributesMeanValue[i], 2)
        }

        // Average of squared differences
        const variance = squaredDifferences / attributes[i].length

        // Standard deviation
        const standardDeviation = Math.sqrt(variance)

        this.#data[category].stDev.push(standardDeviation)
      }
    }
  }

  /**
   * Returns a list of predictions.
   *
   * @param {Array} x - List of examples to predict.
   * @returns {Array} - Predictions.
   */
  predict (x) {
    const categoryPredictions = [] // Predicted categories of array x

    // Iterate all inputs
    for (let i = 0; i < x.length; i++) {
      // Calculate PDF values for each attribute in the input
      const pdfPredictions = this.#calculatePDFPredictions(x, i)

      // Convert pdfPredictions to ln
      const pdfPredictionsLog = this.#convertPDFPredictionsToLog(pdfPredictions)

      // sum of log pdfPredictions
      const sumLogPDF = this.#sumLogPDF(pdfPredictionsLog)

      // Convert log sum to original form
      const expSumLogPDF = this.#convertLogToOriginal(sumLogPDF)

      // Normalize probabilities
      const normalizedProbabilities = this.#normalizeProbabilities(expSumLogPDF)

      // Find highest probability index
      const maxValueIndex = this.#findHighestProbabilityIndex(normalizedProbabilities)

      categoryPredictions.push(maxValueIndex)
    }

    return categoryPredictions
  }

  /**
   * Calculate PDF for attributes of an input.
   *
   * @param {Array} x - Inputs to predict.
   * @param {number} index - Array index of current input to predict.
   * @returns {Array} - Attribute predictions of input.
   */
  #calculatePDFPredictions (x, index) {
    const pdfPredictions = []
    for (let j = 0; j < x[index].length; j++) { // Iterate attributes
      const attributePredictions = []
      for (const category in this.#data) { // check pdf with the same attribute in each category
        const xInput = x[index][j]
        const meanTraining = this.#data[category].mean[j]
        const stdTraining = this.#data[category].stDev[j]

        const pdf = this.#pdf(xInput, meanTraining, stdTraining)

        attributePredictions.push(pdf)
      }
      pdfPredictions.push(attributePredictions)
    }
    return pdfPredictions
  }

  /**
   * Gaussian Probability Density Function.
   *
   * @param {number} xInput - Attribute input to predict.
   * @param {number} meanTraining - Mean of trained attribute.
   * @param {number} stdTraining - Standard deviation of trained attribute.
   * @returns {number} - PDF value.
   */
  #pdf (xInput, meanTraining, stdTraining) {
    return (1 / (Math.sqrt(2 * Math.PI) * stdTraining)) * Math.exp(-((xInput - meanTraining) ** 2) / (2 * stdTraining ** 2))
  }

  /**
   * Convert PDF to log.
   *
   * @param {Array} pdfPredictions - PDF predictions.
   * @returns {Array} - Predictions converted to log.
   */
  #convertPDFPredictionsToLog (pdfPredictions) {
    for (let j = 0; j < pdfPredictions.length; j++) {
      for (let h = 0; h < pdfPredictions[j].length; h++) {
        pdfPredictions[j][h] = Math.log(pdfPredictions[j][h])
      }
    }

    return pdfPredictions
  }

  /**
   * Sum PDF log for each attribute.
   *
   * @param {Array} pdfPredictionsLog - PDF log for each attribute.
   * @returns {Array} - Sum of PDF log for each attribute.
   */
  #sumLogPDF (pdfPredictionsLog) {
    const sumLogPDF = []
    const numOfAttributes = pdfPredictionsLog[0].length

    for (let j = 0; j < pdfPredictionsLog.length; j++) {
      for (let h = 0; h < numOfAttributes; h++) {
        if (sumLogPDF[h] === undefined) {
          sumLogPDF[h] = pdfPredictionsLog[j][h]
          continue
        }
        sumLogPDF[h] += pdfPredictionsLog[j][h]
      }
    }

    return sumLogPDF
  }

  /**
   * Convert log probabilities to original form.
   *
   * @param {Array} sumLogPDF - List of log probabilities.
   * @returns {Array} - List of original form probabilities.
   */
  #convertLogToOriginal (sumLogPDF) {
    const expSumLogPdf = []
    for (let j = 0; j < sumLogPDF.length; j++) {
      const expSum = Math.exp(sumLogPDF[j])
      expSumLogPdf.push(expSum)
    }

    return expSumLogPdf
  }

  /**
   * Normalize probabilities.
   *
   * @param {Array} expSumLogPDF - List of probabilities.
   * @returns {Array} - List of normalized probabilities
   */
  #normalizeProbabilities (expSumLogPDF) {
    let pSum = 0
    for (let j = 0; j < expSumLogPDF.length; j++) {
      pSum += expSumLogPDF[j]
    }

    const normalizedProbabilities = []
    for (let j = 0; j < expSumLogPDF.length; j++) {
      const pNorm = expSumLogPDF[j] / (pSum)
      normalizedProbabilities.push(pNorm)
    }

    return normalizedProbabilities
  }

  /**
   * Find highest probability index of list.
   *
   * @param {Array} normalizedProbabilities - List of normalized probabilities.
   * @returns {number} - Highest probability index.
   */
  #findHighestProbabilityIndex (normalizedProbabilities) {
    let maxValueIndex = 0
    for (let j = 0; j < normalizedProbabilities.length; j++) {
      if (normalizedProbabilities[j] > normalizedProbabilities[maxValueIndex]) {
        maxValueIndex = j
      }
    }

    return maxValueIndex
  }

  /**
   * Returns accuracy score of predictions.
   *
   * @param {Array} predictions - Array of integers with predictions
   * @param {Array} y - Array with labels
   * @returns {number} - accuracy score
   */
  accuracyScore (predictions, y) {
    const totalPredictionsCount = predictions.length
    const correctlyClassified = this.#countCorrectlyClassifiedPredictions(predictions, y, totalPredictionsCount)
    const accuracyScore = correctlyClassified / totalPredictionsCount

    return accuracyScore
  }

  /**
   * Count correctly classified predictions.
   *
   * @param {Array} predictions - List of predictions.
   * @param {Array} y - List of categories.
   * @param {number} totalPredictionsCount - Number of predictions.
   * @returns {number} - Number of correctly.
   */
  #countCorrectlyClassifiedPredictions (predictions, y, totalPredictionsCount) {
    let correctlyClassified = 0
    for (let i = 0; i < totalPredictionsCount; i++) {
      if (y[i] === (predictions[i] + 1)) {
        correctlyClassified += 1
      }
    }

    return correctlyClassified
  }
}
