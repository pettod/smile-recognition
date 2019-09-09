from sklearn.ensemble import AdaBoostClassifier

from src.utils.datahelpers import split_data, load_labels, load_imgs


def main():
    root = 'data/genki4k/'
    imgs = load_imgs(root)
    labels = load_labels(root)
    X_train, X_test, y_train, y_test = split_data(imgs, labels)    
    model = AdaBoostClassifier()
    model.fit(X_train, y_train)
    print(model.predict(X_test))



if __name__ == '__main__':
    main()
