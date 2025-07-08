# Script to count zeros as target regression values

if __name__ == "__main__":
    train_count = 0
    valid_count = 0
    test_count = 0
    files = ["train.txt", "valid.txt", "test.txt"]
    counts = [train_count, valid_count, test_count]
    for i in range(len(files)):
        with open(files[i]) as file:
            for line in file:
                value = float(line.strip().split()[-1])
                if value == 0:
                    print("FOUND A ZERO!")
                    counts[i] += 1

    print("Train count:", counts[0])
    print("Valid count:", counts[1])
    print("Test count: ", counts[2])
