class LogisticRegression {
  constructor(learningRate = 0.1, numIterations = 100) {
    this.learningRate = learningRate;
    this.numIterations = numIterations;
    this.weights = [];
    this.bias = 0;
  }

  train(features, labels) {
    // Initialize weights and bias
    this.weights = new Array(features[0].length).fill(0);

    // Gradient descent
    for (let i = 0; i < this.numIterations; i++) {
      const scores = this.predict(features);
      const gradients = this.computeGradients(features, labels, scores);

      // Update weights and bias
      for (let j = 0; j < this.weights.length; j++) {
        this.weights[j] -= this.learningRate * gradients[j];
      }
      this.bias -= this.learningRate * gradients[this.weights.length];
    }
  }

  predict(features) {//output i*1
    const scores = [];

    for (let i = 0; i < features.length; i++) {
      let score = this.bias;

      for (let j = 0; j < this.weights.length; j++) {
        score += features[i][j] * this.weights[j];
      } 

      scores.push(this.sigmoid(score));
    }

    return scores;
  }

  computeGradients(features, labels, scores) {
    const gradients = new Array(this.weights.length + 1).fill(0);

    for (let i = 0; i < features.length; i++) {
      const error = labels[i] - scores[i];

      for (let j = 0; j < this.weights.length; j++) {
        gradients[j] += error * features[i][j];
      }

      gradients[this.weights.length] += error;
    }

    return gradients;
  }

  sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
  }
}
