# counts number of classes in multilabel classification dataset


if __name__ == "__main__":
    
    # load train data
    with open("train.txt") as file:
        for line in file:
            line_split = line.strip().split()[1:]
            print("NUM CLASSES:", len(line_split))
            break
        