import nltk


def main():
    print("performing setup")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)


if __name__ == "__main__":
    main()
