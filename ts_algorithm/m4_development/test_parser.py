import argparse

def main()-> None:
    """
    test argparse
    """
    parser = argparse.ArgumentParser(description="Define whether to run train or test dataset")

    parser.add_argument("--test", dest="run_test",
                        action='store_true', default=False,
                        help='run on test data set')

    args = parser.parse_args()

    if args.run_test:
        print("Run test")
    else:
        print("Run train")

if __name__ == "__main__":
    main()
