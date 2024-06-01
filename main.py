import preprocessing, kernel


if __name__ == "__main__":
    X, Y, tickers = preprocessing.generate_variables()
    accuracy = kernel.train(X, Y, tickers)
    print("Model Accuracy = {:.4f}".format(accuracy))
